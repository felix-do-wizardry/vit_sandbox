# TODO: add universal fishpp mechanism
import time, json, os

# %%
# updated Experiment 220425_1130
class Experiment:
    def __init__(self,
                model='unknown',
                fishpp=True,
                mask_type='dist',
                mask_levels=3,
                global_heads_str='3',
                global_proj_str='m',
                bs=128,
                gpu=1,
                accumulation_steps=1,
                time_stamp=None,
                non_linear=False,
                non_linear_bias=True,
                layer_limit=-1,
                stage_limit=-1,
                ):
        if time_stamp is None:
            self.time_stamp = time.strftime('%y%m%d_%H%M%S')
        else:
            self.time_stamp = str(time_stamp)
        
        assert isinstance(bs, int) and bs >= 1
        assert isinstance(gpu, int) and gpu >= 1
        assert isinstance(accumulation_steps, int) and accumulation_steps >= 0
        if accumulation_steps < 2:
            accumulation_steps = 1
        self.accumulation_steps = accumulation_steps
        self.gpu = gpu
        self.bs = bs
        self.bs_eff = bs * gpu * accumulation_steps
        
        self.mask_type = str(mask_type)
        self.mask_levels = str(mask_levels)
        self.global_heads_str = str(global_heads_str)
        self.global_proj_str = str(global_proj_str)
        
        self.non_linear = bool(non_linear)
        self.non_linear_bias = bool(non_linear_bias)
        self.name_nl = ''
        if self.non_linear:
            if self.non_linear_bias:
                self.name_nl = 'nl'
            else:
                self.name_nl = 'nlb'
        
        self.layer_limit = layer_limit
        self.stage_limit = stage_limit
        assert isinstance(self.layer_limit, int)
        assert isinstance(self.stage_limit, int)
        assert self.layer_limit <= 0 or self.stage_limit <= 0, '??'
        self.name_layer = ''
        if self.layer_limit > 0:
            self.name_layer = f'l{self.layer_limit}'
        if self.stage_limit > 0:
            self.name_layer = f's{self.stage_limit}'
        
        self.fishpp = bool(fishpp)
        self.name_bs = f'{self.bs}x{self.gpu}{"" if self.accumulation_steps < 2 else "x"+str(self.accumulation_steps)}'
        self.model = str(model)
        if self.fishpp:
            self.name_type = f'{self.mask_type}{self.mask_levels}_g{self.global_heads_str}{self.global_proj_str}'
            if self.name_nl:
                self.name_type = f'{self.name_type}_{self.name_nl}'
            if self.name_layer:
                self.name_type = f'{self.name_type}_{self.name_layer}'
        else:
            self.name_type = 'baseline'
        
        self.name = f'{self.model}_{self.name_type}_{self.name_bs}'
        self.exp_name = f'{self.time_stamp}_{self.name}'
    
    def __repr__(self) -> str:
        return f'ts[{self.time_stamp}] type[{self.name_type}] name[{self.name}] exp_name[{self.exp_name}]'
