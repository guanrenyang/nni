# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from examples.compression.models import (
    # build_resnet18,
    prepare_dataloader,
    prepare_optimizer,
    train,
    training_step,
    evaluate,
    device
)

from nni.contrib.compression.pruning import (
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    DynamicGranularityPruner
)
from nni.contrib.compression.utils import auto_set_denpendency_group_ids
from nni.compression.pytorch.speedup.v2 import ModelSpeedup

from torchvision.models.resnet import resnet18

from transformers import AutoFeatureExtractor, ResNetForImageClassification

def build_resnet18():
    model = resnet18(pretrained=True)
    return model.to(device)

prune_type = 'l1'

import time
if __name__ == '__main__':
    # finetuning resnet18 on Cifar10
    t = time.time()
    # model = build_resnet18()
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=10)
    _, test_loader = prepare_dataloader()
    print('Original model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Original model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')
    print("Train time: {} s".format(time.time()-t))
    input()
    config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.5
    }]
    dummy_input = torch.rand(8, 3, 224, 224).to(device)
    config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
    optimizer = prepare_optimizer(model)

    # if prune_type == 'l1':
    #     pruner = L1NormPruner(model, config_list)
    # elif prune_type == 'l2':
    #     pruner = L2NormPruner(model, config_list)
    # else:
    #     pruner = FPGMPruner(model, config_list)
    pruner = DynamicGranularityPruner(model, config_list)

    _, masks = pruner.compress()
    pruner.unwrap_model()

    model = ModelSpeedup(model, dummy_input, masks).speedup_model()
    print('Pruned model paramater number: ', sum([param.numel() for param in model.parameters()]))
    print('Pruned model without finetuning acc: ', evaluate(model, test_loader), '%')

    optimizer = prepare_optimizer(model)
    train(model, optimizer, training_step, lr_scheduler=None, max_steps=None, max_epochs=10)
    _, test_loader = prepare_dataloader()
    print('Pruned model after 10 epochs finetuning acc: ', evaluate(model, test_loader), '%')

# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.

# import torch

# from examples.compression.models import (
#     build_resnet18,
#     prepare_dataloader,
#     prepare_optimizer,
#     train,
#     training_step,
#     evaluate,
#     device
# )

# from nni.contrib.compression.pruning import (
#     L1NormPruner,
#     L2NormPruner,
#     FPGMPruner,
#     DynamicGranularityPruner
# )
# from nni.contrib.compression.utils import auto_set_denpendency_group_ids
# from nni.compression.pytorch.speedup.v2 import ModelSpeedup

# prune_type = 'l1'

#     config_list = [{'op_names': ['layer3.1.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer1.0.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer4.1.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer1.1.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer2.0.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer3.0.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer4.0.downsample.0'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer1.0.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['fc'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer2.1.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer3.0.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer4.0.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer4.1.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer2.0.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer2.1.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer3.1.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer1.1.conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['conv1'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer3.0.downsample.0'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer4.0.conv2'],
#                     'sparse_ratio': 0.5},
#                     {'op_names': ['layer2.0.downsample.0'],
#                     'sparse_ratio': 0.5}]