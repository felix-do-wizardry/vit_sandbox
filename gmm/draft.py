# %%
import pickle
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time, os, json


# %% NUMPY IMAGE MANIPULATION
class NumpyImage:
    @classmethod
    def add_grid(cls, a, col, row=None, grid_value=None, sep=1):
        if row is None:
            row = col
        assert len(a.shape) == 2
        assert len(a.shape) >= 2
        assert a.shape[-2] % row == 0
        assert a.shape[-1] % col == 0
        cell_h = int(a.shape[-2] / row)
        cell_w = int(a.shape[-1] / col)
        a_4d = a.reshape(row, cell_h, col, cell_w)
        a_4d_t = a_4d.transpose(0,2,1,3)
        b = np.pad(
            a_4d_t,
            pad_width=[[0,0]] * 2 + [[0,sep]] * 2,
            constant_values=grid_value,
        )
        c = b.transpose(0,2,1,3).reshape(row*(cell_h+sep), col*(cell_w+sep))
        d = np.pad(
            c,
            pad_width=[[sep,0]] * 2,
            constant_values=grid_value,
        )
        return d
    
    @classmethod
    def img_concat(cls, imgs, col=None, row=None, sep=0, border=0, bg=None):
        if isinstance(imgs, list):
            _count = len(imgs)
            assert len(imgs) > 0
            assert all([isinstance(v, np.ndarray) for v in imgs])
            assert all([len(v.shape) == len(list(imgs[0].shape)) for v in imgs[1:]])
            _shapes = np.array([v.shape for v in imgs]).astype(int)
            _shape = np.max(_shapes, axis=0)
            _pads = _shape[None] - _shapes
            _pads0 = np.floor(_pads / 2).astype(int)
            _pads1 = _pads - _pads0
            _pads_width = np.stack([_pads0, _pads1], axis=2)
            imgs_pad = [
                np.pad(
                    _img,
                    _pads_width[i],
                    constant_values=bg,
                )
                for i, _img in enumerate(imgs)
            ]
            imgs = np.stack(imgs_pad, 0)
        assert isinstance(imgs, np.ndarray)
        assert len(imgs.shape) >= 3
        assert imgs.shape[0] >= 1
        _imgs = imgs
        if len(imgs.shape) > 3:
            _imgs = imgs.reshape(-1, *imgs.shape[-2:])
        _count = _imgs.shape[0]
        if col is None:
            if row is None:
                col = _count
                row = 1
            else:
                col = int(np.ceil(_count / row))
        else:
            if row is None:
                row = int(np.ceil(_count / col))
        
        if border is False:
            border = sep
        _shape = np.array(_imgs.shape[1:])
        assert len(_shape) == 2
        _img = np.zeros(
            shape=(_shape + sep) * np.array([row, col]) + sep * (border if border else -1),
            dtype=float,
        )
        _img[:, :] = bg
        for gy in range(row):
            for gx in range(col):
                gi = gy * col + gx
                if gi >= _count:
                    break
                gp = np.array([gy, gx])
                pp = gp * (_shape + sep) + sep * (border if border else 0)
                # print(gp, pp, _shape)
                _img[
                    pp[0] : pp[0] + _shape[0],
                    pp[1] : pp[1] + _shape[1],
                ] = _imgs[gi]
        return _img


# %%
class ImageReadme:
    def __init__(self, image_width=240, begin_lines=[], dp='.'):
        self.image_width = image_width
        self.begin_lines = begin_lines
        assert isinstance(dp, str)
        self.dp = dp
        if not os.path.isdir(dp):
            os.makedirs(dp)
        
        self.sections = {}
        self.section_index = {}
    
    # def add_readme(self, name, section, ):
    #     self.add_section(section)
    #     section_index = self.section_index[section]
    #     _images = self.sections[section_index]
    #     _images
    
    def __repr__(self) -> str:
        _group_count = sum([len(d) for d in self.sections.values()])
        _fig_count = sum([len(d1) for d in self.sections.values() for d1 in d.values()])
        return ' | '.join([
            'ImageReadme',
            f'{len(self.sections)} sections',
            f'{_group_count} groups',
            f'{_fig_count} figs',
        ]).join('<>')
    
    def add_section_group(self, section, group):
        assert isinstance(section, str)
        assert isinstance(group, str)
        if section not in self.sections:
            self.sections[section] = {}
            # self.sections[section] = {
            #     'name': section,
            #     'groups': {},
            #     # 'readme': [],
            # }
        if group not in self.sections[section]:
            self.sections[section][group] = []
        # return self.section_index[section]
    
    def write_fig(self, fig, name, sub_dir=None):
        assert isinstance(name, str)
        assert len(name) > 0
        fp_rel = f'{name}.png'
        if sub_dir is not None:
            assert isinstance(sub_dir, str)
            assert len(sub_dir) > 0
            fp_rel = os.path.join(sub_dir, fp_rel)
        
        fp = os.path.join(self.dp, fp_rel)
        _dp = os.path.split(fp)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        fig.write_image(fp)
        return fp_rel
    
    def save_figs(self, section='', group='', sub_dir=None, save_readme=False, **kwargs):
        _ = self.add_section_group(section=section, group=group)
        for name, fig in kwargs.items():
            _fp_rel = self.write_fig(fig=fig, name=name, sub_dir=sub_dir)
            self.sections[section][group].append(_fp_rel)
        
        if save_readme:
            self.save_readme()
    
    def save_readme(self, sections=None, section_level=2):
        assert sections is None, 'for now'
        assert isinstance(section_level, int)
        assert section_level >= 1
        readme_lines = []
        for _section, d in self.sections.items():
            readme_lines.append(f"{'#' * section_level} {_section}")
            for _group, fps_rel in d.items():
                readme_lines.extend([
                    f'> {_group}',
                    '<p float="left" align="left">',
                ])
                for i, fp_rel in enumerate(fps_rel):
                    readme_lines.append(
                        f'<img src="{fp_rel}" width="{self.image_width}" />',
                    )
        readme_txt = '\n\n'.join(readme_lines)
        fp_rm = os.path.join(self.dp, 'README.md')
        with open(fp_rm, 'w') as fo:
            fo.writelines(readme_txt)
        # print(f'[PLOT] saved with README.md in <{self.dp}>')
        return self.dp

