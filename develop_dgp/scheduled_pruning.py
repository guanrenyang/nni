# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from examples.compression.models import (
    build_resnet18,
    prepare_dataloader,
    prepare_optimizer,
    train,
    training_step,
    evaluate,
    device
)

from nni.contrib.compression import TorchEvaluator
from nni.contrib.compression.pruning import TaylorPruner, LinearPruner, AGPPruner, LevelPruner
from nni.contrib.compression.utils import auto_set_denpendency_group_ids
from nni.compression.pytorch.speedup.v2 import ModelSpeedup

from typing import Dict
import matplotlib.pyplot as plt
import os
schedule_type = 'agp'


def pruning_info(model: torch.nn.Module, masks: Dict[str, Dict[str, torch.Tensor]], masks_dir: str = './masks'):
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
        
    for module_name, module in model.named_modules():
        if module_name in masks and 'weight' in masks[module_name]:
            mask = masks[module_name]['weight'].cpu()
            
            fig = plt.figure()
            plt.matshow(mask.reshape((mask.shape[0], -1)) if len(mask.shape)==4 else mask)
            plt.colorbar()
            plt.savefig(os.path.join(masks_dir, module_name+'.png'))

            plt.close(fig)
if __name__ == '__main__':
    # finetuning resnet18 on Cifar10
    model = build_resnet18()
    optimizer = prepare_optimizer(model)
    # No need to train on image net
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=1)
    _, test_loader = prepare_dataloader()
    print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Original model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

    config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5
    }]
    dummy_input = torch.rand(8, 3, 224, 224).to(device)
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
    optimizer = prepare_optimizer(model)
    evaluator = TorchEvaluator(train, optimizer, training_step)

    # sub_pruner = TaylorPruner(model, config_list, evaluator, training_steps=10)
    sub_pruner = LevelPruner(model, config_list, evaluator)

    if schedule_type == 'agp':
        scheduled_pruner = AGPPruner(sub_pruner, interval_steps=10, total_times=10)
    elif schedule_type == 'linear':
        scheduled_pruner = LinearPruner(sub_pruner, interval_steps=10, total_times=10)

    _, masks = scheduled_pruner.compress(max_steps=10 * 10, max_epochs=None)
    scheduled_pruner.unwrap_model()

    pruning_info(model, masks)
    from IPython import embed
    embed()

    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Pruned model without finetuning acc: ', evaluate(model, test_loader), '%')

    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=10)
    _, test_loader = prepare_dataloader()
    print('Pruned model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

