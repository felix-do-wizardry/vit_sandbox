from distutils.command.config import config
import random
import warnings
from torch import nn
from torch.nn.utils import prune
import numpy as np

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmcv_custom.runner import EpochBasedRunnerAmp
try:
    import apex
except:
    print('apex is not installed')


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)
# Hai Do
# import numpy as np
# img_size = np.array([800, 1333])
# token_grid = np.ceil(img_size / 28).astype(int)
# token_grids = [token_grid]
# x = token_grid
# for i in range(3):
#     x = np.ceil(x / 2).astype(int)
#     token_grids.append(x)
# token_grids = np.array(token_grids)
# token_grids
    def calculate_token_grids(img_size):
        img_size = np.array(img_size)
        token_grid = np.ceil(img_size / 28).astype(int)
        token_grids = [token_grid]
        x = token_grid
        for i in range(3):
            x = np.ceil(x / 2).astype(int)
            token_grids.append(x)
        token_grids = np.array(token_grids)
        return token_grids

    def add_pi(max_nW = None):
        num_blocks = [2,2,6,2]
        num_heads = [3,6,12,24]
        # token_grids = calculate_token_grids([800,1333])
        for i in range(4):
            for j in range(num_blocks[i]):
                model.backbone.layers[i].blocks[j].attn.pi = nn.Parameter(torch.ones(1, num_heads[i], 49, 49)/49., requires_grad = True)
        # model.to(config.device)

    def add_pi_mask(max_nW = None):
        num_blocks = [2,2,6,2]
        num_heads = [3,6,12,24]
        # token_grids = calculate_token_grids([800,1333])
        for i in range(4):
            for j in range(num_blocks[i]):
                model.module.backbone.layers[i].blocks[j].attn.register_buffer('pi_mask', torch.ones(1, num_heads[i], 49 ,49))
    
    def pruning(runner, local = False):
        num_blocks = [2,2,6,2]
        if not local:
            for i in range(4):
                parameters_to_prune = []
                # for i in range(4):
                for j in range(num_blocks[i]):
                    parameters_to_prune.append((runner.model.module.backbone.layers[i].blocks[j].attn, 'pi'))
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount = 0.7)
        else:
            for i in range(4):
                for j in range(num_blocks[i]):
                ###pruning locally
                    prune.l1_unstructured(runner.model.module.backbone.layers[i].blocks[j].attn, name = 'pi', amount = 0.7)
        
            
        for i in range(4):
            for j in range(num_blocks[i]):
                pi_mask = runner.model.module.backbone.layers[i].blocks[j].attn.pi_mask.data
                prune.remove(runner.model.module.backbone.layers[i].blocks[j].attn, 'pi')
                runner.model.module.backbone.layers[i].blocks[j].attn.pi.requires_grad = False
                runner.model.module.backbone.layers[i].blocks[j].attn.register_buffer('pi_mask', pi_mask)
                del runner.model.module.backbone.layers[i].blocks[j].attn.pi
                # del runner.model.module.backbone.layers[i].blocks[j].attn.relative_position_bias_table
        
        ### CHECK PRUNING MASK
        for i in range(4):
            for j in range(num_blocks[i]):
                pi_mask = runner.model.module.backbone.layers[i].blocks[j].attn.pi_mask.data
                print((abs(pi_mask) != 0).sum())

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            model, optimizer = apex.amp.initialize(
                model.cuda(), optimizer, opt_level="O1")
            for m in model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True
    
    # add_pi_mask()
    # add_pi()
    # checkpoint = torch.load(f'/tam/data/coco/swin_chks/swim_gmm1x_prune/latest.pth')
    add_pi()
    # checkpoint = torch.load(f'/tam/data/coco/swin_chks/swim_gmm1x/latest.pth')

    # model.load_state_dict(checkpoint['state_dict'])
    # import pdb;pdb.set_trace()
    # pruning(model, local=False)
    

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    # build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # import pdb;pdb.set_trace()

    
    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    runner.load_checkpoint('/tam/data/coco/swin_chks/swim_gmm1x/latest.pth')
    pruning(runner,local=False)
    # import pdb;pdb.set_trace()
    # print(runner.model)
    # for name, param in runner.model.module.named_parameters():
    #     if param.grad is None:
    #         print(name)


        
    runner.run(data_loaders, cfg.workflow)
