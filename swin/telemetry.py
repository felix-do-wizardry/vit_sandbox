# %%
import numpy as np
import time, os, json, math, string, re
import wandb, torch

import pandas as pd

# TODO: 1 - create and manage wandb logging
# TODO: log running time for epochs/steps
# TODO: add project/run automatic analysis

# %%
# updated 220425_1200
class WnB_Run:
    time_units = {
        'ms': 0.001,
        's': 1.0,
        'm': 60.,
        'h': 3600.,
        'd': 86400.,
    }
    def __init__(self,
                entity='felix-do-wizardry',
                project='temp',
                name='temp',
                config=None,
                args=None,
                max_metrics=None,
                min_metrics=None,
                step_metric=None,
                with_timestamp=True,
                summary=None,
                enabled=True,
                # **kwargs,
                ):
            
        self.enabled = bool(enabled)
        if config is None:
            config = {}
        
        self.timestamp = time.strftime('%y%m%d_%H%M%S')
        self.time_start = time.time()
        self.time_current = time.time()
        self.time_elapsed = 0.
        
        assert isinstance(name, str)
        self.name = name
        assert isinstance(project, str)
        self.project = project
        assert isinstance(entity, str)
        self.entity = entity
        
        if self.enabled:
            wandb.init(
                project=project,
                entity=entity,
                config=None if config is None else config,
                name=name,
            )
            
            # set all other train/ metrics to use this step
            if isinstance(step_metric, str):
                assert len(step_metric) > 0
                self.step_metric = step_metric
                wandb.define_metric(step_metric)
                wandb.define_metric("*", step_metric=step_metric)
            else:
                self.step_metric = None
            
            if isinstance(max_metrics, list):
                for _m in max_metrics:
                    wandb.define_metric(_m, summary="max")
            if isinstance(min_metrics, list):
                for _m in min_metrics:
                    wandb.define_metric(_m, summary="min")
            
            if with_timestamp:
                wandb.run.summary['timestamp'] = self.timestamp
            
            if isinstance(summary, dict):
                for k, v in summary.items():
                    wandb.run.summary[k] = v
            
            if args is None:
                # wandb.run.summary.update()
                wandb.config.update({})
            else:
                wandb.config.update(args)
            # wandb.config.update(args)
            # print(f"Initiated WandB project[{_project}] name[{exp_name}]")
        
        self.step = 0
        self.timers = {}
        self.log_dict = {}
    
    def __repr__(self) -> str:
        if not self.enabled:
            return f'<WnB_Run[DISABLED] [{self.timestamp}] [{self.entity}/{self.project}/{self.name}]>'
        return f'<WnB_Run [{self.timestamp}] [{self.entity}/{self.project}/{self.name}]>'
    
    def update_time(self):
        self.time_current = time.time()
        self.time_elapsed = self.time_current - self.time_start
    
    def timer_start(self, name='train_time', unit='s'):
        assert isinstance(name, str)
        assert len(name) > 0
        if name not in self.timers:
            assert unit in self.time_units, f'unit[{unit}] must be one of {self.time_units}'
            self.timers[name] = {
                'name': name,
                'last_start': None,
                'elapsed': 0.,
                'unit': unit,
                'count': -1,
                'running': True,
                'value': 0.,
            }
        _timer = self.timers[name]
        _timer['last_start'] = time.time()
        _timer['count'] = _timer['count'] + 1
        _timer['running'] = True
    
    def timer_stop(self, name='train_time'):
        assert name in self.timers
        _timer = self.timers[name]
        # _timer['last_start'] = time.time()
        _time_current = time.time()
        _timer['elapsed'] = _time_current - _timer['last_start']
        _timer['value'] = _timer['elapsed'] / self.time_units[_timer['unit']]
        _timer['running'] = False
    
    def log(self, log=None, summary=None, mem=None, on_hold=False):
        self.update_time()
        if not self.enabled:
            return None
        if isinstance(summary, dict):
            for k, v in summary.items():
                assert isinstance(k, str)
                wandb.run.summary[k] = v
        # assert isinstance(log, dict)
        # if isinstance(log, dict):
        self.log_dict.update({
            _name: _timer['elapsed']
            for _name, _timer in self.timers.items()
        })
        if mem is not None:
            assert isinstance(mem, str)
            # max memory in GB
            _mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.log_dict.update({mem: float(_mem_gb)})
        
        if isinstance(log, dict):
            self.log_dict.update(log)
        
        if not on_hold:
            wandb.log(self.log_dict)
            self.step += 1
            self.log_dict = {}
        
        return self.step
    
    def finish(self):
        if not self.enabled:
            return None
        wandb.finish()
        return True
    
    @classmethod
    def calc_flops(cls, input_shape, model, device='cuda'):
        '''
        input_shape (list of int): shape for the input, including batch_size
        '''
        from fvcore.nn import FlopCountAnalysis
        
        _input = torch.tensor(
            # np.ones(shape=[1, 3, _img_size, _img_size]) / 2,
            np.ones(shape=input_shape) / 2,
            dtype=torch.float32,
            device=device,
        )
        model.train(True)
        flops = FlopCountAnalysis(model, _input)
        # print(f'fvcore GFLOPS: {flops.total() / 1e9}')
        
        return float(flops.total())
    
    @classmethod
    def calc_params(cls, model):
        '''
        '''
        param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])
        return int(param_count)

# %%
if __name__ == '__main__':
    WnB_Run
    
    args = None
    RUN = WnB_Run(
        entity='felix-do-wizardry',
        project='temp_proj',
        name='temp',
        # config=None,
        args=args,
        # max_metrics=['acc', 'acc5', 'train_mem_gb',],
        # min_metrics=None,
        step_metric='epoch',
        # with_timestamp=True,
        summary=dict(
            epoch=-1,
            acc=0.0,
            acc5=0.0,
            # type=EXP.name_type,
            # bs=EXP.bs,
            # bs_eff=EXP.bs_eff,
            # gpu=EXP.gpu,
            image_size=224,
        ),
        # enabled=utils.get_rank() == 0,
    )
    
    
    RUN.timer_start('epoch_time')
    RUN.timer_start('train_time')
    # train step
    RUN.timer_stop('train_time')
    
    RUN.timer_start('test_time')
    # test step
    RUN.timer_stop('test_time')
    RUN.timer_stop('epoch_time')
    
    RUN.log(
        summary=dict(
            # epoch=epoch,
            # acc=max_accuracy,
            # acc5=max_accuracy5,
        ),
        log=dict(
            # epoch=epoch,
            # train_loss=train_stats['loss'],
            # lr=train_stats['lr'],
            # test_acc=test_stats["acc1"],
            # test_acc5=test_stats["acc5"],
            # test_loss=test_stats["loss"],
        ),
        mem='train_mem_gb',
    )
    
    RUN.finish()
