# -*- coding: utf-8 -*-
"""
统一主训练脚本

本脚本是所有模型训练和评估的统一入口。它负责：
1.  使用 argparse 解析所有超参数和配置。
2.  通过 `--model_name` 参数选择要训练的模型 (TDRGCN, RGCNGRU, GCNLSTM, etc.)。
3.  加载数据、划分数据集、计算归一化参数。
4.  根据选择的模型，初始化对应的模型架构(们)，如生成器和判别器。
5.  从 `src.engine` 模块调用正确的训练循环 (标准或对抗性)。
6.  在测试集上评估性能最佳的模型。
7.  保存特定于该模型的最佳权重和训练报告 (JSON格式)。
"""
import argparse
import json
import pickle
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# 从 src 模块导入所有需要的组件
from src.models import (
    TDRGCN, RGCNGRUModelWithHourlyHeads, RGCNTransformerModelWithHourlyHeads,
    GCNLSTMModelWithHourlyHeads, GINELSTMModelWithHourlyHeads,
    GAELSTMModelWithHourlyHeads, GGANLSTMModelWithHourlyHeads, PredictionDiscriminatorRGCN,
    CGANLSTMModel, PatchGANDiscriminator
)
from src.engine import (
    train_epoch, evaluate_epoch,
    train_epoch_adversarial_gggan, evaluate_epoch_gggan,
    train_epoch_adversarial_cgan, evaluate_epoch_cgan,
    train_epoch_gaelstm
)
from src.utils import (
    generate_time_features_for_sequence, print_hourly_metrics_summary,
    calculate_aggregated_metrics_report
)


def main(args):
    """主训练函数。"""
    train_start_time = time.time()

    # --- 1. 设置和初始化 ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting training for model: {args.model_name} on device: {device} ---")

    # --- 2. 加载和预处理数据 ---
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "rb") as f:
        all_sequences = pickle.load(f)

    # --- 3. 数据集划分和归一化 ---
    # (此部分逻辑对于所有模型通用)
    num_obs_steps = 1  # 所有模型都使用1个观测步
    expected_len = args.T_pred_horizon + num_obs_steps
    valid_sequences = [seq for seq in all_sequences if
                       isinstance(seq, list) and len(seq) == expected_len and all(isinstance(g, Data) for g in seq)]
    print(f"Found {len(valid_sequences)} / {len(all_sequences)} valid sequences of length {expected_len}.")

    indices = np.random.permutation(len(valid_sequences))
    train_size, val_size = int(0.7 * len(valid_sequences)), int(0.2 * len(valid_sequences))
    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size + val_size], indices[
                                                                                                                train_size + val_size:]
    train_data, val_data, test_data = [valid_sequences[i] for i in train_indices], [valid_sequences[i] for i in
                                                                                    val_indices], [valid_sequences[i]
                                                                                                   for i in
                                                                                                   test_indices]

    print("Calculating normalization parameters from training set...")
    # Node/Static Feature Scaler
    all_x = torch.cat([seq[0].x for seq in train_data], dim=0)
    node_feat_mean, node_feat_std = torch.mean(all_x, dim=0), torch.std(all_x, dim=0)
    node_feat_std[node_feat_std < 1e-8] = 1.0
    # Target Value Scaler
    all_y = torch.cat([g.y.squeeze() for seq in train_data for g in seq[num_obs_steps:] if ~g.building_mask.all()],
                      dim=0)
    target_mean, target_std = torch.mean(all_y), torch.std(all_y)

    # --- 4. 创建 Dataloaders ---
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # --- 5. 初始化模型、优化器和损失函数 ---
    sample_graph = valid_sequences[0][0]
    model_params = {'static_node_in_dim': sample_graph.x.shape[1],
                    'global_env_in_dim': sample_graph.graph_global_env_features.shape[0], 'time_in_dim': 4,
                    **vars(args)}

    # 兼容GRU/LSTM/Transformer的参数名
    model_params['lstm_hidden_dim'] = model_params.get('gru_hidden_dim', 128)
    model_params['num_lstm_layers'] = model_params.get('num_gru_layers', 1)
    model_params['dropout_rate_lstm'] = model_params.get('dropout_rate_gru', 0.2)
    model_params['transformer_d_model'] = model_params.get('gru_hidden_dim', 128)
    model_params['transformer_dropout_rate'] = model_params.get('dropout_rate_gru', 0.2)
    model_params['dropout_rate_gcn'] = model_params.get('dropout_rate_rgcn', 0.3)
    model_params['dropout_rate_gine'] = model_params.get('dropout_rate_rgcn', 0.3)

    model_G, model_D = None, None
    optimizer_G, optimizer_D = None, None
    is_gan = False

    model_map = {
        'TDRGCN': TDRGCN,
        'RGCNGRU': RGCNGRUModelWithHourlyHeads,
        'RGCNTransformer': RGCNTransformerModelWithHourlyHeads,
        'GCNLSTM': GCNLSTMModelWithHourlyHeads,
        'GINELSTM': GINELSTMModelWithHourlyHeads,
        'GAELSTM': GAELSTMModelWithHourlyHeads,
        'GGANLSTM': GGANLSTMModelWithHourlyHeads,
        'CGANLSTM': CGANLSTMModel,
    }

    if args.model_name in model_map:
        model_G = model_map[args.model_name](**model_params).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    print(
        f"Generator/Main Model ({args.model_name}) Parameters: {sum(p.numel() for p in model_G.parameters() if p.requires_grad):,}")

    # Setup for GAN models
    if args.model_name == 'GGANLSTM':
        is_gan = True
        model_D = PredictionDiscriminatorRGCN(
            original_node_feature_dim=model_params['static_node_in_dim'], prediction_dim=1,
            hidden_dim=args.disc_hidden_dim, num_layers=args.disc_num_layers,
            num_relations=model_params['num_relations'], dropout_rate=args.disc_dropout_rate
        ).to(device)
        print(f"Discriminator (RGCN) Parameters: {sum(p.numel() for p in model_D.parameters() if p.requires_grad):,}")
    elif args.model_name == 'CGANLSTM':
        is_gan = True
        disc_input_channels = model_params['unet_input_channels_after_fusion'] + 1
        model_D = PatchGANDiscriminator(input_channels=disc_input_channels).to(device)
        print(
            f"Discriminator (PatchGAN) Parameters: {sum(p.numel() for p in model_D.parameters() if p.requires_grad):,}")

    # Optimizers
    if is_gan:
        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    else:
        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=args.scheduler_patience)

    # --- 6. 训练循环 ---
    save_dir = Path(args.save_dir) / args.model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / f"best_model_seed{args.seed}.pth"
    base_dt = datetime(args.year, args.month, args.day, 8)  # Prediction starts at 8 AM
    time_features = generate_time_features_for_sequence(base_dt, args.T_pred_horizon).to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Select the correct training and evaluation loop
        if args.model_name == 'GGANLSTM':
            # ... Adversarial training for GGAN
            pass  # Placeholder for GGAN training loop
        elif args.model_name == 'CGANLSTM':
            # ... Adversarial training for CGAN
            pass  # Placeholder for CGAN training loop
        elif args.model_name == 'GAELSTM':
            train_loss, train_time = train_epoch_gaelstm(model_G, train_loader, optimizer_G, device, time_features,
                                                         node_feat_mean, node_feat_std, target_mean, target_std,
                                                         lambda_kld=args.lambda_kld)
        else:  # Standard models
            train_loss, train_time = train_epoch(model_G, train_loader, optimizer_G, device, time_features,
                                                 node_feat_mean, node_feat_std, target_mean, target_std)

        val_loss, val_metrics, _ = evaluate_epoch(model_G, val_loader, device, time_features, node_feat_mean,
                                                  node_feat_std, target_mean, target_std)
        scheduler_G.step(val_loss)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer_G.param_groups[0]['lr']:.6f} | Time: {train_time:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss;
            patience_counter = 0
            torch.save(model_G.state_dict(), model_save_path)
            print("    -> Best model saved!")
        else:
            patience_counter += 1
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # --- 7. 最终评估 ---
    print(f"\n--- Final Evaluation on Test Set ({args.model_name}) ---")
    model_G.load_state_dict(torch.load(model_save_path))
    test_loss, test_metrics, test_time = evaluate_epoch(model_G, test_loader, device, time_features, node_feat_mean,
                                                        node_feat_std, target_mean, target_std)
    print(f"Test Loss: {test_loss:.4f}, Inference Time: {test_time:.2f}s")
    print_hourly_metrics_summary("Test", test_metrics, args.T_pred_horizon)

    # --- 8. 保存报告 ---
    report = {'config': vars(args), 'test_loss': test_loss,
              'test_metrics_aggregated': calculate_aggregated_metrics_report(test_metrics, args.T_pred_horizon)}
    report_path = save_dir / f"training_report_seed{args.seed}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    print(f"\nTraining report saved to: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="统一训练脚本")

    # Core Args
    parser.add_argument('--data_path', type=str, required=True, help='图序列数据路径 (.pkl)')
    parser.add_argument('--save_dir', type=str, default='results', help='保存结果的根目录')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['TDRGCN', 'RGCNGRU', 'RGCNTransformer', 'GCNLSTM', 'GINELSTM', 'GAELSTM', 'GGANLSTM',
                                 'CGANLSTM'])
    parser.add_argument('--seed', type=int, default=42)

    # Training Args
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=45)

    # Model Structure Args
    parser.add_argument('--T_pred_horizon', type=int, default=12)
    parser.add_argument('--num_relations', type=int, default=5, help='RGCN/GGAN/GAE中关系的数量')
    parser.add_argument('--global_env_emb_dim', type=int, default=16)
    parser.add_argument('--time_emb_dim', type=int, default=8)
    parser.add_argument('--rgcn_hidden_dim', type=int, default=128)
    parser.add_argument('--rgcn_output_dim', type=int, default=128)
    parser.add_argument('--gcn_hidden_dim', type=int, default=128)
    parser.add_argument('--gcn_output_dim', type=int, default=128)
    parser.add_argument('--gine_hidden_dim', type=int, default=128)
    parser.add_argument('--gine_output_dim', type=int, default=128)
    parser.add_argument('--gine_edge_dim', type=int, default=6)
    parser.add_argument('--gru_hidden_dim', type=int, default=128, help='用于 GRU/LSTM/Transformer 的隐藏层维度')
    parser.add_argument('--num_gru_layers', type=int, default=1, help='用于 GRU/LSTM 的层数')
    parser.add_argument('--fusion_mlp_output_dim', type=int, default=128)
    parser.add_argument('--mlp_prediction_hidden_dim', type=int, default=64)
    parser.add_argument('--dropout_rate_rgcn', type=float, default=0.3)
    parser.add_argument('--dropout_rate_gru', type=float, default=0.2)
    parser.add_argument('--dropout_rate_pred_head', type=float, default=0.2)

    # Transformer Specific
    parser.add_argument('--transformer_nhead', type=int, default=4)
    parser.add_argument('--transformer_num_encoder_layers', type=int, default=2)
    parser.add_argument('--transformer_num_decoder_layers', type=int, default=2)
    parser.add_argument('--transformer_dim_feedforward', type=int, default=512)

    # GAE (CGVAE) Specific
    parser.add_argument('--cgvae_latent_dim', type=int, default=64)
    parser.add_argument('--cgvae_num_encoder_layers', type=int, default=2)
    parser.add_argument('--cgvae_num_decoder_layers', type=int, default=2)
    parser.add_argument('--lambda_kld', type=float, default=0.1, help='KLD loss weight for GAELSTM')

    # GGAN / CGAN Specific
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for Generator')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for Discriminator')
    parser.add_argument('--lambda_adv', type=float, default=0.001, help='Adversarial loss weight for GGANLSTM')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='L1 loss weight for CGANLSTM')
    parser.add_argument('--disc_hidden_dim', type=int, default=64)
    parser.add_argument('--disc_num_layers', type=int, default=2)
    parser.add_argument('--disc_dropout_rate', type=float, default=0.3)

    # CGAN (UNet) Specific
    parser.add_argument('--static_feat_dim', type=int, default=16)
    parser.add_argument('--unet_input_channels_after_fusion', type=int, default=32)
    parser.add_argument('--unet_feature_output_channels', type=int, default=128)
    parser.add_argument('--h0_unet_output_channels', type=int, default=128)

    # Date Args
    parser.add_argument('--year', type=int, default=2023);
    parser.add_argument('--month', type=int, default=5);
    parser.add_argument('--day', type=int, default=3)

    args = parser.parse_args()
    main(args)

