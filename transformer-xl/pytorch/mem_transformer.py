import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits

import os, json
from h_matrix import H_Matrix_1D, H_Matrix_Masks_1D, get_layer_indices, FishPP_Head

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


# L2
class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            # print(mems.size())
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        print(attn_prob.size())
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

    def _parallelogram_mask(self, h, w, left=False):
        # print('1')
        mask = torch.ones((h, w)).byte()
        # mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x


class RelPartialLearnableMultiHeadAttn_FishPP(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 
                 mask_levels=3,
                 global_heads=2,
                 global_proj_type='full',
                 non_linear=0,
                 non_linear_bias=1,
                 masks=None,
                 ):
        super(RelPartialLearnableMultiHeadAttn_FishPP, self).__init__()
        
        assert d_model == d_head * n_head, '?? HEADS and DIM'
        self.n_head = n_head
        self.global_heads = global_heads
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        
        _dims = FishPP_Head.get_dims(
            heads=self.n_head,
            global_heads=self.global_heads,
            head_dim=d_head,
        )
        print('<ATTN> [FISHPP] dims:', _dims)
        self.qk_dim = _dims['qk_dim']
        self.v_dim = _dims['v_dim']
        
        # self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        # self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.qkv_net = nn.Linear(d_model, _dims['qkv_dim'], bias=False)
        self.r_net = nn.Linear(self.d_model, _dims['qk_dim'], bias=False)

        self.masks = masks
        self.mask_levels = mask_levels
        self.global_proj_type = global_proj_type
        self.head_ratio = int(self.n_head // self.global_heads)
        if self.global_proj_type == 'mix':
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.n_head * self.global_heads, device='cuda'))
        elif self.global_proj_type == 'full':
            assert self.n_head % self.global_heads == 0
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.n_head, device='cuda'))
        else:
            assert self.n_head % self.global_heads == 0
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.head_ratio, device='cuda'))


        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
    
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        
        # mems
        if mems is not None:
            cat = torch.cat([mems, w], 0)
        else:
            cat = w
        
        if self.pre_lnorm:
            cat = self.layer_norm(cat)
        
        w_heads = self.qkv_net(cat)
        # r [N, 1, H*D]
        r_head_k = self.r_net(r)
        
        # w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_heads[..., : self.qk_dim]
        w_head_k = w_heads[..., self.qk_dim : self.qk_dim * 2]
        w_head_v = w_heads[..., self.qk_dim * 2 : ]
        
        if mems is not None:
            w_head_q = w_head_q[-qlen:]

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.global_heads, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.global_heads, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)               # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.global_heads, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        # r_w_bias [GH, D] (given from L0 class)
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        # [Nq, B, GH, D] x [Nk, B, GH, D] -> [Nq, Nk, B, GH]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        
        # r_r_bias [GH, D] (given from L0 class)
        rr_head_q = w_head_q + r_r_bias
        # [Nq, B, GH, D] x [Nk, GH, D] -> [Nq, Nk, B, GH]
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
        
        
        # FISHPP: attn_score [Nq, Nk, B, GH] -> attn [B, GH, Nq, Nk]
        Nq, Nk, B, _ = attn_score.shape
        # print('attn_score', attn_score.size()) # 
        # print(Nq, Nk, B, _, mems)
        # assert Nq == Nk, f'{Nq} != {Nk}'
        # assert 0
        # N = Nq
        attn = attn_score.permute(2, 3, 0, 1).contiguous()
        
        # masks [1, 1, N, N, Level] -> [1, 1, N, N, HR / H / GH*H]
        mask_weights = self.masks @ self.mask_proj
        
        if self.global_proj_type == 'mix':
            mask_weights = mask_weights.reshape([1, N, N, self.global_heads, self.n_head])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, H]
        elif self.global_proj_type == 'full':
            mask_weights = mask_weights.reshape([1, N, N, self.global_heads, self.head_ratio])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, HR]
        
        # attn [B, GH, N, N] x masks [1, 1/GH, N, N, HR/H] -> [B, GH, N, N, HR/H]
        attn = attn[..., None] * mask_weights
        
        # TODO: implement non_linear
        if self.global_proj_type == 'mix':
            # [B, GH, N, N, H] -> [B, N, N, H]
            attn = torch.sum(attn, dim=-4, keepdim=False)
            # [B, N, N, H] -> [B, H, N, N]
            attn = attn.permute(0, 3, 1, 2).contiguous()
        else:
            # [B, GH, N, N, HR] -> [B, GH, HR, N, N] -> [B, H, N, N]
            attn = attn.permute(0, 1, 4, 2, 3).reshape(B, self.n_head, N, N).contiguous()
        
        # print('r', r.size())
        # print('w_heads', w_heads.size()) # [N, B, qkv_dim]
        # print('w_head_q', w_head_q.size()) # [N, B, GH, D]
        # print('w_head_k', w_head_k.size()) # [N, B, GH, D]
        # print('w_head_v', w_head_v.size()) # [N, B, H, D]
        # print('r_head_k', r_head_k.size()) # [N, GH, D]
        # print('mask_weights', mask_weights.size()) # 
        # print('attn_score', attn_score.size()) # 
        # print('attn', attn.size()) # 
        # assert 0
        
        # attn: [B, H, N, N] -> attn_score: [N, N, B, H]
        attn_score = attn.permute(2, 3, 0, 1).contiguous()
        # continue with local head attn
        
        
        
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

    def _parallelogram_mask(self, h, w, left=False):
        # print('1')
        mask = torch.ones((h, w)).byte()
        # mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x


# L1
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                pre_lnorm,
                fishpp=0,
                # fish_kwargs={},
                **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        # TODO: split flow and implement fishpp class
        if fishpp:
            self.dec_attn = RelPartialLearnableMultiHeadAttn_FishPP(
                n_head, d_model, d_head, dropout,
                **kwargs,
            )
        else:
            self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=pre_lnorm)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


# L2_2
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

# L2_2
class MultiHeadAttn_FishPP(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False,
                 mask_levels=3,
                 global_heads=2,
                 global_proj_type='full',
                 non_linear=0,
                 non_linear_bias=1,
                 masks=None,
                 token_count=160,
                 ):
        super(MultiHeadAttn_FishPP, self).__init__()
        
        assert masks is not None
        assert d_model == d_head * n_head, '?? HEADS and DIM'
        self.non_linear = non_linear
        self.non_linear_bias = non_linear_bias
        self.global_proj_type = global_proj_type
        self.global_heads = global_heads
        self.mask_levels = mask_levels
        self.num_heads = n_head
        self.masks = masks
        self.token_count = token_count
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        
        
        
        _dims = FishPP_Head.get_dims(
            heads=n_head,
            global_heads=global_heads,
            head_dim=d_head,
        )
        # print('<ATTN> [FISHPP] dims:', _dims)
        self.qk_dim = _dims['qk_dim']
        self.v_dim = _dims['v_dim']
        print(f'<ATTN> [FISHPP] head[{global_heads}->{n_head}] mask_levels[{mask_levels}] global_proj_type[{global_proj_type}]')
        
        # self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        # self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        
        self.q_net = nn.Linear(d_model, _dims['qk_dim'], bias=False)
        self.k_net = nn.Linear(d_model, _dims['qk_dim'], bias=False)
        self.v_net = nn.Linear(d_model, _dims['v_dim'], bias=False)
        
        self.head_ratio = int(self.n_head // self.global_heads)
        if self.global_proj_type == 'mix':
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.n_head * self.global_heads, device='cuda'))
        elif self.global_proj_type == 'full':
            assert self.n_head % self.global_heads == 0
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.n_head, device='cuda'))
        else:
            assert self.n_head % self.global_heads == 0
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.head_ratio, device='cuda'))
        
    
    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        assert mems is None, f'debug to check for mems: {mems.size()}'
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        
        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)
        
        Nq = h.size(0)
        Nk = c.size(0)
        B = h.size(1)
        assert B == c.size(1)
        
        head_q = self.q_net(h)
        head_k = self.k_net(c)
        head_v = self.v_net(c)

        head_q = head_q.view(Nq, B, self.global_heads, self.d_head)
        head_k = head_k.view(Nk, B, self.global_heads, self.d_head)
        head_v = head_v.view(Nk, B, self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head] / [Nq, Nk, B, GH]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
        
        # FISHPP: attn_score [Nq, Nk, B, H]
        attn_score
        
        
        # FISHPP: attn_score [Nq, Nk, B, GH] -> attn [B, GH, Nq, Nk]
        Nq, Nk, B, _ = attn_score.shape
        N = Nq
        # print('attn_score', attn_score.size()) # 
        # print(Nq, Nk, B, _, mems)
        # assert Nq == Nk == self.token_count, f'mismatch: Nq[{Nq}] Nk[{Nk}] token_count[{self.token_count}]'
        assert Nq == Nk, f'mismatch: Nq[{Nq}] Nk[{Nk}]'
        # [Nq, Nk, B, GH] -> [B, GH, Nq, Nk]
        attn = attn_score.permute(2, 3, 0, 1).contiguous()
        
        # masks [1, 1, N, N, Level] -> [1, 1, N, N, HR / H / GH*H]
        mask_weights = self.masks @ self.mask_proj
        
        if self.global_proj_type == 'mix':
            mask_weights = mask_weights.reshape([1, self.token_count, self.token_count, self.global_heads, self.n_head])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, H]
        elif self.global_proj_type == 'full':
            mask_weights = mask_weights.reshape([1, self.token_count, self.token_count, self.global_heads, self.head_ratio])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, HR]
        
        mask_weights = mask_weights[:, :, :Nq, :Nk]
        # attn [B, GH, N, N] x masks [1, 1/GH, N, N, HR/H] -> [B, GH, N, N, HR/H]
        attn = attn[..., None] * mask_weights
        
        # TODO: implement non_linear
        if self.global_proj_type == 'mix':
            # [B, GH, N, N, H] -> [B, N, N, H]
            attn = torch.sum(attn, dim=-4, keepdim=False)
            # [B, N, N, H] -> [B, H, N, N]
            attn = attn.permute(0, 3, 1, 2).contiguous()
        else:
            # [B, GH, N, N, HR] -> [B, GH, HR, N, N] -> [B, H, N, N]
            attn = attn.permute(0, 1, 4, 2, 3).reshape(B, self.n_head, N, N).contiguous()
        
        # print('r', r.size())
        # print('w_heads', w_heads.size()) # [N, B, qkv_dim]
        # print('w_head_q', w_head_q.size()) # [N, B, GH, D]
        # print('w_head_k', w_head_k.size()) # [N, B, GH, D]
        # print('w_head_v', w_head_v.size()) # [N, B, H, D]
        # print('r_head_k', r_head_k.size()) # [N, GH, D]
        # print('mask_weights', mask_weights.size()) # 
        # print('attn_score', attn_score.size()) # 
        # print('attn', attn.size()) # 
        # assert 0
        
        # attn: [B, H, N, N] -> attn_score: [N, N, B, H]
        attn_score = attn.permute(2, 3, 0, 1).contiguous()
        # continue with local attn_score [Nq, Nk, B, H]
        
        
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# L1_2
class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, fishpp=0, **kwargs):
        super(DecoderLayer, self).__init__()
        
        if fishpp:
            self.dec_attn = MultiHeadAttn_FishPP(n_head, d_model, d_head, dropout, **kwargs)
        else:
            self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

# L0
class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1,
                 
                 fishpp=0,
                 mask_type='h1d',
                 mask_levels=4,
                 non_linear=0,
                 non_linear_bias=1,
                 
                 global_heads=3,
                 global_proj_type='mix',
                 
                 layer_limit=None,
                 layer_offset=0,
                 ):
        
        self.qlen_logs = {}
        
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, 
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        
        if fishpp:
            self.global_heads = global_heads
            fish_layer_indices = get_layer_indices(
                limit=layer_limit,
                offset=layer_offset,
                count=n_layer,
            )
            print('layers', fish_layer_indices)
            _masks, _ = H_Matrix_Masks_1D.get_masks(
                mask_type=mask_type,
                n=tgt_len,
                mask_levels=mask_levels,
            )
            self.masks = torch.tensor(
                _masks,
                dtype=torch.float32, device='cuda', requires_grad=False,
            )
            assert _ == mask_levels, f'mask_levels is modified [{mask_levels}->{_}] in H_Matrix_Masks_1D.get_masks | not supposed to happen'
        else:
            self.global_heads = n_head
            fish_layer_indices = []
            self.masks = None

        assert attn_type == 2, f'attn_type[{attn_type}] | only the default value [2] is allowed for fish++'
        self.attn_type = attn_type
        
        # TODO: implement non_linear
        assert not non_linear
        L1_fish_kwargs = dict(
            fishpp=1,
            mask_levels=mask_levels,
            global_heads=global_heads,
            global_proj_type=global_proj_type,
            non_linear=non_linear,
            non_linear_bias=non_linear_bias,
            masks=self.masks,
            token_count=tgt_len,
        )
        
        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            assert 0
            # for i in range(n_layer):
            #     if fishpp and i in fish_layer_indices:
            #         # TODO: implement L1 class fish kwargs passing
            #         self.layers.append(
            #             RelPartialLearnableDecoderLayer(
            #                 n_head, d_model, d_head, d_inner, dropout,
            #                 tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
            #                 dropatt=dropatt, pre_lnorm=pre_lnorm,
            #                 **L1_fish_kwargs,
            #             )
            #         )
            #     else:
            #         self.layers.append(
            #             RelPartialLearnableDecoderLayer(
            #                 n_head, d_model, d_head, d_inner, dropout,
            #                 tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
            #                 dropatt=dropatt, pre_lnorm=pre_lnorm,
            #             )
            #         )
        elif attn_type == 1: # learnable embeddings
            assert 0
            # for i in range(n_layer):
            #     self.layers.append(
            #         RelLearnableDecoderLayer(
            #             n_head, d_model, d_head, d_inner, dropout,
            #             tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
            #             dropatt=dropatt, pre_lnorm=pre_lnorm)
            #     )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        **(L1_fish_kwargs if fishpp and i in fish_layer_indices else {}),
                    )
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            assert 0
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.global_heads, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.global_heads, self.d_head))
        elif self.attn_type == 1: # learnable
            assert 0
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            # assert 0
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            assert 0
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        
        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
            # dec_attn_mask = (torch.triu(all_ones, 1+mlen)
            #         + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]
            # dec_attn_mask = torch.triu(
            #     word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        hids = []
        if self.attn_type == 0: # default
            assert 0
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            assert 0
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    # print(mems_i.size())
                    # print(pos_emb.size())
                    mems_i += pos_emb[:mlen]
                    # assert 0
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            assert 0
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True, 
                            d_embed=d_embed, div_val=div_val, 
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, 
                            cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
