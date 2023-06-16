# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from functools import reduce
import heapq
from typing import Callable, Dict, List, Tuple

import numpy
import torch

from .calculate_metrics import _METRICS, _DYNAMIC_GRANULARITY_METRICS
from ...base.compressor import _PRUNING_TARGET_SPACES
from ...base.target_space import PruningTargetSpace, TargetType


_MASKS = Dict[str, Dict[str, torch.Tensor]]

def granularity_aware_generate_sparsity(metrics: _DYNAMIC_GRANULARITY_METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    def img2col_back_ward(mask, raw_shape):
        return mask.reshape(raw_shape)
    # def compare_pruning_unit(a, b):
    #     # (granularity, layer, row_idx, block_idx, gran_threshold[elem[1], elem[2]])
    #     gran_a, gran_b = a[0], b[0]
    #     if (gran_a >= gran_b and a[4][gran_b] < b[4][gran_b]) or (gran_a < gran_b and a[4][gran_a] < b[4][gran_a]):
    #         return -1
    #     elif gran_a==gran_b and a[4][gran_a]==b[4][gran_b]:
    #         return 0
    #     else :
    #         return 1
    granularities: Dict[str, Dict[str, int]] = {'layer4.0.conv2': {'weight': 16}, 'layer4.1.conv1': {'weight': 16},\
                                                 'layer3.0.conv2': {'weight': 16}, 'layer1.0.conv1': {'weight': 16}, \
                                                    'layer2.0.conv2': {'weight': 16}, 'layer2.1.conv1': {'weight': 16}, \
                                                        'layer4.0.downsample.0': {'weight': 16}, 'layer3.1.conv1': {'weight': 16}, \
                                                            'layer2.0.downsample.0': {'weight': 16}, 'layer4.0.conv1': {'weight': 16},
                                                              'layer4.1.conv2': {'weight': 16}, 'layer1.1.conv2': {'weight': 16}, 
                                                              'layer1.1.conv1': {'weight': 16}, 'layer1.0.conv2': {'weight': 16}, 
                                                              'layer2.0.conv1': {'weight': 16}, 'layer3.1.conv2': {'weight': 16}, 'conv1': {'weight': 16}, 
                                                              'layer3.0.conv1': {'weight': 16}, 'layer3.0.downsample.0': {'weight': 16}, 'layer2.1.conv2': {'weight': 16}}

    granularity_unaware_metric = defaultdict(dict)
    for module_name, module_target_spaces in target_spaces.items():
        for target_name, target_space in module_target_spaces.items():
            if target_name == 'bias':
                continue
            granularity = granularities.get(module_name, {}).get(target_name, 16)
            granularity_unaware_metric[module_name][target_name] = metrics[module_name][target_name][str(granularity)]

            

    group: List[Tuple[str, str, PruningTargetSpace]] = []
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if target_name == 'weight':
                group.append((module_name, target_name, target_space))  # type: ignore

    masks = defaultdict(dict)
    
    group_sparse_ratio = None
    for _, _, target_space in group:
        if target_space.sparse_ratio is not None:
            if group_sparse_ratio is None:
                group_sparse_ratio = target_space.sparse_ratio
            else:
                assert group_sparse_ratio == target_space.sparse_ratio
    assert group_sparse_ratio is not None

    # at least how many elements to mask
    sparse_number_low = 0
    # at most how many elements to mask
    sparse_number_high = 0
    # how many elements in this group
    total_element_number = 0
    for _, _, target_space in group:
        element_number = target_space.target.numel()  # type: ignore
        total_element_number += element_number
        sparse_number_low += int(element_number * target_space.min_sparse_ratio) if target_space.min_sparse_ratio else 0
        sparse_number_high += int(element_number * target_space.max_sparse_ratio) if target_space.max_sparse_ratio else element_number
    # how many elements should be masked, controlled by sparse_ratio
    sparse_number = int(total_element_number * group_sparse_ratio)

    # if sparse_number <= sparse_number_low:
    #     # directly generate masks with target_space.min_sparse_ratio
    #     for module_name, target_name, target_space in group:
    #         sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
    #         masks[module_name][target_name] = _ratio_mask(granularity_unaware_metric[module_name][target_name], sparse_ratio)
    #         continue

    # if sparse_number >= sparse_number_high:
    #     # directly generate masks with target_space.max_sparse_ratio
    #     for module_name, target_name, target_space in group:
    #         sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 0.0
    #         masks[module_name][target_name] = _ratio_mask(granularity_unaware_metric[module_name][target_name], sparse_ratio)
    #         continue

    sparse_threshold = _global_threshold_generate(granularity_unaware_metric, group, sparse_number)
    for module_name, target_name, target_space in group:
        granularity = granularities[module_name][target_name]
        mask = _threshold_mask(granularity_unaware_metric[module_name][target_name], sparse_threshold)

        shape = [1 for _ in mask.shape] + [granularity]
        mask = mask.unsqueeze(-1).repeat(shape)
        mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))
        
        masks[module_name][target_name] = img2col_back_ward(mask, target_space.shape)

        print(masks[module_name][target_name].shape, target_space.target.shape)
        continue

    return masks
            
def generate_sparsity(metrics: _METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    """
    There are many ways to generate masks, in this function, most of the common generation rules are implemented,
    and these rules execute in a certain order.

    The following rules are included in this function:

        * Threshold. If sparse_threshold is set in target space, the mask will be generated by metric >= threshold is 1,
        and metric < threshold is 0.ß
        * Dependency. If dependency_group_id is set in target space, the metrics of the targets in the same group will be
        meaned as the group metric, then if target_space.internal_metric_block is set, all internal_metric_block of the
        targets will be put in one set to compute the lcm as the group internal block number.
        Split the group metric to group internal block number parts, compute mask for each part and merge as a group mask.
        All targets in this group share the group mask value.
        * Global. If global_group_id is set in target space, the metrics of the targets in the same group will be global ranked
        and generate the mask by taking smaller metric values as 0, others as 1 by sparse_ratio.
        * Ratio. The most common rule, directly generate the mask by taking the smaller metric values as 0, others as 1 by sparse_ratio.
        * Align. If align is set in target space, the mask will be generated by another existed mask.
    """

    def condition_dependency(target_space: PruningTargetSpace) -> bool:
        return target_space.dependency_group_id is not None

    def condition_global(target_space: PruningTargetSpace) -> bool:
        return target_space.global_group_id is not None

    def condition_ratio(target_space: PruningTargetSpace) -> bool:
        return target_space.sparse_ratio is not None

    def condition_threshold(target_space: PruningTargetSpace) -> bool:
        return target_space.sparse_threshold is not None

    def condition_align(target_space: PruningTargetSpace) -> bool:
        return target_space.align is not None

    masks = defaultdict(dict)

    threshold_target_spaces, remained_target_spaces = target_spaces_filter(target_spaces, condition_threshold)
    update_masks = _generate_threshold_sparsity(metrics, threshold_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, threshold_target_spaces))

    dependency_target_spaces, remained_target_spaces = target_spaces_filter(target_spaces, condition_dependency)
    update_masks = _generate_dependency_sparsity(metrics, dependency_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, dependency_target_spaces))

    global_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_global)
    update_masks = _generate_global_sparsity(metrics, global_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, global_target_spaces))

    ratio_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_ratio)
    update_masks = _generate_ratio_sparsity(metrics, ratio_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, ratio_target_spaces))

    align_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_align)
    update_masks = _generate_align_sparsity(masks, align_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, align_target_spaces))

    return masks


def target_spaces_filter(target_spaces: _PRUNING_TARGET_SPACES,
                         condition: Callable[[PruningTargetSpace], bool]) -> Tuple[_PRUNING_TARGET_SPACES, _PRUNING_TARGET_SPACES]:
    filtered_target_spaces = defaultdict(dict)
    remained_target_spaces = defaultdict(dict)

    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if (target_space.type is TargetType.PARAMETER and target_space.target is None) or not condition(target_space):
                remained_target_spaces[module_name][target_name] = target_space
            else:
                filtered_target_spaces[module_name][target_name] = target_space

    return filtered_target_spaces, remained_target_spaces


def _generate_ratio_sparsity(metrics: _METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    # NOTE: smaller metric value means more un-important
    masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            metric = metrics[module_name][target_name]
            min_sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
            max_sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 1.0
            sparse_ratio = min(max_sparse_ratio, max(min_sparse_ratio, target_space.sparse_ratio))  # type: ignore
            masks[module_name][target_name] = _ratio_mask(metric, sparse_ratio)
    return masks


def _generate_threshold_sparsity(metrics: _METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    # NOTE: smaller metric value means more un-important
    masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            metric = metrics[module_name][target_name]
            # metric < threshold will be 0, metric >= threshold will be 1
            mask = _threshold_mask(metric, target_space.sparse_threshold)  # type: ignore

            # if sparse_ratio does not meet `min_sparse_ratio`, `max_sparse_ratio`, re-generate mask
            sparse_ratio = 1.0 - mask.sum() / mask.numel()
            min_sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
            max_sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 1.0
            if sparse_ratio < min_sparse_ratio:
                mask = _ratio_mask(metric, min_sparse_ratio)
            if sparse_ratio > max_sparse_ratio:
                mask = _ratio_mask(metric, max_sparse_ratio)

            masks[module_name][target_name] = mask
    return masks


def _generate_align_sparsity(masks: _MASKS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    align_masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            align_module_name = target_space.align['module_name']  # type: ignore
            align_module_name = module_name if align_module_name is None else align_module_name
            assert align_module_name in masks
            src_mask = masks[align_module_name][target_space.align['target_name']]  # type: ignore
            align_dims: List[int] = target_space.align['dims']  # type: ignore
            reduce_dims = [d for d in range(len(src_mask.shape)) if d not in align_dims and d - len(src_mask.shape) not in align_dims]
            align_mask = src_mask.sum(reduce_dims).bool().float()
            if target_space._scaler is not None:
                assert target_space.shape is not None, f'The shape of {module_name}.{target_name} is not tracked'
                align_mask = \
                    target_space._scaler.shrink(target_space._scaler.expand(align_mask, target_space.shape), keepdim=True).bool().float()
            align_masks[module_name][target_name] = align_mask
    return align_masks


def _generate_global_sparsity(metrics: _METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    groups: Dict[str, List[Tuple[str, str, PruningTargetSpace]]] = defaultdict(list)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            groups[target_space.global_group_id].append((module_name, target_name, target_space))  # type: ignore

    masks = defaultdict(dict)
    for _, group in groups.items():
        group_sparse_ratio = None
        for _, _, target_space in group:
            if target_space.sparse_ratio is not None:
                if group_sparse_ratio is None:
                    group_sparse_ratio = target_space.sparse_ratio
                else:
                    assert group_sparse_ratio == target_space.sparse_ratio
        assert group_sparse_ratio is not None

        # at least how many elements to mask
        sparse_number_low = 0
        # at most how many elements to mask
        sparse_number_high = 0
        # how many elements in this group
        total_element_number = 0
        for _, _, target_space in group:
            element_number = target_space.target.numel()  # type: ignore
            total_element_number += element_number
            sparse_number_low += int(element_number * target_space.min_sparse_ratio) if target_space.min_sparse_ratio else 0
            sparse_number_high += int(element_number * target_space.max_sparse_ratio) if target_space.max_sparse_ratio else element_number
        # how many elements should be masked, controlled by sparse_ratio
        sparse_number = int(total_element_number * group_sparse_ratio)

        if sparse_number <= sparse_number_low:
            # directly generate masks with target_space.min_sparse_ratio
            for module_name, target_name, target_space in group:
                sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
                masks[module_name][target_name] = _ratio_mask(metrics[module_name][target_name], sparse_ratio)
            continue

        if sparse_number >= sparse_number_high:
            # directly generate masks with target_space.max_sparse_ratio
            for module_name, target_name, target_space in group:
                sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 0.0
                masks[module_name][target_name] = _ratio_mask(metrics[module_name][target_name], sparse_ratio)
            continue

        sparse_threshold = _global_threshold_generate(metrics, group, sparse_number)
        for module_name, target_name, target_space in group:
            masks[module_name][target_name] = _threshold_mask(metrics[module_name][target_name], sparse_threshold)
        continue
    return masks


def _generate_dependency_sparsity(metrics: _METRICS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    groups: Dict[str, List[Tuple[str, str, PruningTargetSpace]]] = defaultdict(list)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            groups[target_space.dependency_group_id].append((module_name, target_name, target_space))  # type: ignore

    masks = defaultdict(dict)
    for _, group in groups.items():
        block_numbers = [1]
        group_sparsity_ratio = None
        filtered_metrics = defaultdict(dict)

        for module_name, target_name, target_space in group:
            assert target_space.internal_metric_block is None or isinstance(target_space.internal_metric_block, int)
            block_numbers.append(target_space.internal_metric_block if target_space.internal_metric_block else 1)
            if target_space.sparse_ratio is not None:
                if group_sparsity_ratio is None:
                    group_sparsity_ratio = target_space.sparse_ratio
                else:
                    assert group_sparsity_ratio == target_space.sparse_ratio
            filtered_metrics[module_name][target_name] = metrics[module_name][target_name]
        block_number = reduce(numpy.lcm, block_numbers)
        assert group_sparsity_ratio is not None
        group_metric = _metric_fuse(filtered_metrics)
        group_mask = _ratio_mask(group_metric, group_sparsity_ratio, view_size=[block_number, -1])

        for module_name, target_name, _ in group:
            masks[module_name][target_name] = group_mask.clone()

    return masks


# the following are helper functions

def _ratio_mask(metric: torch.Tensor, sparse_ratio: float, view_size: int | List[int] = -1):
    if sparse_ratio == 0.0:
        return torch.ones_like(metric)

    if sparse_ratio == 1.0:
        return torch.zeros_like(metric)

    assert 0.0 < sparse_ratio < 1.0
    if isinstance(view_size, int) or len(view_size[:-1]) == 0:
        block_number = 1
    else:
        block_number = numpy.prod(view_size[:-1])
    sparse_number_per_block = int(metric.numel() // block_number * sparse_ratio)
    viewed_metric = metric.view(view_size)
    _, indices = viewed_metric.topk(sparse_number_per_block, largest=False)
    return torch.ones_like(viewed_metric).scatter(-1, indices, 0.0).reshape_as(metric)


def _threshold_mask(metric: torch.Tensor, sparse_threshold: float):
    return (metric >= sparse_threshold).float().to(metric.device)


def _global_threshold_generate(metrics: _METRICS,
                               group: List[Tuple[str, str, PruningTargetSpace]],
                               sparse_number: int) -> float:
    buffer = []
    buffer_elem = 0
    for module_name, target_name, target_space in group:
        metric = metrics[module_name][target_name]
        grain_size = target_space.target.numel() // metric.numel()  # type: ignore
        for m in metric.cpu().detach().view(-1):
            if buffer_elem <= sparse_number:
                heapq.heappush(buffer, (-m.item(), grain_size))
                buffer_elem += grain_size
            else:
                _, previous_grain_size = heapq.heappushpop(buffer, (-m.item(), grain_size))
                buffer_elem += grain_size - previous_grain_size
    return -heapq.heappop(buffer)[0]


def _nested_multiply_update_masks(default_dict: _MASKS, update_dict: _MASKS):
    # if a target already has a mask, the old one will multiply the new one as the target mask,
    # that means the mask in default dict will more and more sparse.
    for module_name, target_tensors in update_dict.items():
        for target_name, target_tensor in target_tensors.items():
            if target_name in default_dict[module_name] and isinstance(default_dict[module_name][target_name], torch.Tensor):
                default_dict[module_name][target_name] = (default_dict[module_name][target_name] * target_tensor).bool().float()
            else:
                default_dict[module_name][target_name] = target_tensor


def _metric_fuse(metrics: _METRICS) -> torch.Tensor:
    # mean all metric value
    fused_metric = None
    count = 0
    for _, module_metrics in metrics.items():
        for _, target_metric in module_metrics.items():
            if fused_metric is not None:
                fused_metric += target_metric
            else:
                fused_metric = target_metric.clone()
            count += 1
    assert fused_metric is not None
    return fused_metric / count


def _expand_masks(masks: _MASKS, target_spaces: _PRUNING_TARGET_SPACES) -> _MASKS:
    # expand the mask shape from metric shape to target shape
    new_masks = defaultdict(dict)
    for module_name, module_masks in masks.items():
        for target_name, target_mask in module_masks.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler:
                new_masks[module_name][target_name] = \
                    target_space._scaler.expand(target_mask, target_space.shape, keepdim=True, full_expand=False)  # type: ignore
            else:
                new_masks[module_name][target_name] = target_mask
    return new_masks
