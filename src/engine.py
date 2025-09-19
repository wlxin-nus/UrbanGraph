# -*- coding: utf-8 -*-
"""
训练与评估引擎 (Training & Evaluation Engine)

本模块包含执行模型训练和评估的单个周期的核心函数。
- mse_loss_masked: 计算MSE损失，但会忽略建筑物内部节点和NaN目标值。
- calculate_hourly_metrics: 在原始（未归一化）尺度上计算每个小时的详细回归指标。
- train_epoch: 执行一个完整的训练周期（epoch）。
- evaluate_epoch: 执行一个完整的评估/测试周期（epoch）。
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import r2_score


def mse_loss_masked(predictions_scaled, targets_scaled, mask):
    """
    计算被掩码的均方误差损失。
    只在非建筑区域且目标值有效的节点上计算损失。
    """
    expanded_mask = mask.unsqueeze(1).expand_as(targets_scaled)
    valid_targets_mask = ~torch.isnan(targets_scaled)
    final_mask = expanded_mask & valid_targets_mask

    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions_scaled.device, requires_grad=True)

    loss = F.mse_loss(predictions_scaled[final_mask], targets_scaled[final_mask])
    return loss


def calculate_hourly_metrics(predictions_scaled, targets_scaled, node_masks, target_mean, target_std):
    """在原始尺度上计算每个预测时间步的回归指标。"""
    # 反归一化
    preds_unscaled = predictions_scaled.cpu() * (target_std + 1e-8) + target_mean
    targets_unscaled = targets_scaled.cpu() * (target_std + 1e-8) + target_mean

    T_horizon = preds_unscaled.shape[1]
    hourly_metrics = {}

    preds_np = preds_unscaled.numpy()
    targets_np = targets_unscaled.numpy()
    mask_np = node_masks.cpu().numpy()

    for t in range(T_horizon):
        preds_t = preds_np[:, t][mask_np]
        targets_t = targets_np[:, t][mask_np]

        valid_mask = ~np.isnan(targets_t)
        preds_t_valid, targets_t_valid = preds_t[valid_mask], targets_t[valid_mask]

        if preds_t_valid.shape[0] < 2:
            hourly_metrics[t] = {'mse': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'count': 0}
            continue

        mse = np.mean((preds_t_valid - targets_t_valid) ** 2)
        r2 = r2_score(targets_t_valid, preds_t_valid) if len(np.unique(targets_t_valid)) > 1 else np.nan

        hourly_metrics[t] = {
            'mse': mse,
            'mae': np.mean(np.abs(preds_t_valid - targets_t_valid)),
            'rmse': np.sqrt(mse),
            'r2': r2,
            'count': preds_t_valid.shape[0]
        }
    return hourly_metrics


def train_epoch(model, loader, optimizer, device, timeline_features, target_mean, target_std):
    """执行一个训练周期。"""
    model.train()
    total_loss = 0
    num_seq_processed = 0
    epoch_start_time = time.time()

    for list_of_batches in loader:
        optimizer.zero_grad()

        # 1. 模型前向传播
        preds_scaled = model(list_of_batches, timeline_features, device)

        # 2. 准备目标张量并计算损失
        targets_list = [step.y.squeeze() for step in list_of_batches[1:]]
        targets_original = torch.stack(targets_list, dim=1).to(device)
        targets_scaled = (targets_original - target_mean) / (target_std + 1e-8)

        mask = ~list_of_batches[1].to(device).building_mask

        loss = mse_loss_masked(preds_scaled, targets_scaled, mask)

        # 3. 反向传播和优化
        if loss.item() > 0:
            loss.backward()
            optimizer.step()

        num_graphs = list_of_batches[0].num_graphs
        total_loss += loss.item() * num_graphs
        num_seq_processed += num_graphs

    epoch_duration = time.time() - epoch_start_time
    avg_loss = total_loss / num_seq_processed if num_seq_processed > 0 else 0.0
    return avg_loss, epoch_duration


def evaluate_epoch(model, loader, device, timeline_features, target_mean, target_std):
    """执行一个评估/测试周期。"""
    model.eval()
    all_preds_scaled, all_targets_scaled, all_masks = [], [], []
    total_loss = 0
    num_seq_processed = 0
    eval_start_time = time.time()

    with torch.no_grad():
        for list_of_batches in loader:
            preds_scaled = model(list_of_batches, timeline_features, device)

            targets_list = [step.y.squeeze() for step in list_of_batches[1:]]
            targets_original = torch.stack(targets_list, dim=1).to(device)
            targets_scaled = (targets_original - target_mean) / (target_std + 1e-8)

            mask = ~list_of_batches[1].to(device).building_mask
            loss = mse_loss_masked(preds_scaled, targets_scaled, mask)

            num_graphs = list_of_batches[0].num_graphs
            total_loss += loss.item() * num_graphs
            num_seq_processed += num_graphs

            all_preds_scaled.append(preds_scaled.cpu())
            all_targets_scaled.append(targets_scaled.cpu())
            all_masks.append(mask.cpu())

    eval_duration = time.time() - eval_start_time
    avg_loss = total_loss / num_seq_processed if num_seq_processed > 0 else 0.0

    if not all_preds_scaled:
        return avg_loss, {}, eval_duration

    final_preds = torch.cat(all_preds_scaled, dim=0)
    final_targets = torch.cat(all_targets_scaled, dim=0)
    final_masks = torch.cat(all_masks, dim=0)

    hourly_metrics = calculate_hourly_metrics(final_preds, final_targets, final_masks, target_mean.cpu(),
                                              target_std.cpu())
    return avg_loss, hourly_metrics, eval_duration
