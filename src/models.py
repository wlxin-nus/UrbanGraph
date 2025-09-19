# -*- coding: utf-8 -*-
"""
模型架构定义模块 (Model Architectures)

本模块包含项目中使用的所有神经网络模型。
- MLPEncoder: 一个标准的多层感知机编码器，用于对非图特征进行编码。
- RGCNModule: 一个包含三层RGCNConv的关系图卷积网络模块，用于捕捉空间依赖。
- RGCNLSTMModelWithSingleHead: 项目的核心时空预测模型架构，整合了特征编码器、
  空间图编码器(RGCN)、时空演化模块(LSTM)和预测头。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


class MLPEncoder(nn.Module):
    """一个通用的多层感知机 (MLP) 编码器。"""

    def __init__(self, in_dim, out_dim, hid_dim=None, dropout_rate=0.1):
        super().__init__()
        if hid_dim is None:
            hid_dim = max(min(in_dim, out_dim), (in_dim + out_dim) // 2)
            if hid_dim == 0: hid_dim = max(in_dim, out_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.LayerNorm(hid_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class RGCNModule(nn.Module):
    """一个三层的关系图卷积网络 (RGCN) 模块。"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout_rate=0.5):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)
        self.conv3 = RGCNConv(hidden_dim, out_dim, num_relations)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.prelu3 = nn.PReLU(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None or edge_attr.shape[1] < 5:
            raise ValueError("RGCNModule: edge_attr 缺少或列数不足以提取 edge_type。")

        # 边类型在 edge_attr 的第5列 (索引4)
        edge_type = edge_attr[:, 4].long()

        x = self.conv1(x, edge_index, edge_type=edge_type)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type=edge_type)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_type=edge_type)
        x = self.bn3(x)
        x = self.prelu3(x)
        return x


class RGCNLSTMModelWithSingleHead(nn.Module):
    """
    T-DRGCN 模型的核心实现。

    该模型集成了RGCN来处理空间依赖，并使用LSTM来建模时间动态。
    它采用单一预测头，一次性输出未来所有时间步的预测值。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')
        self.num_relations = kwargs.get('num_relations', 5)
        self.lstm_hidden_dim = kwargs.get('lstm_hidden_dim')

        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        rgcn_hidden_dim = kwargs.get('rgcn_hidden_dim', 128)
        rgcn_output_dim = kwargs.get('rgcn_output_dim', 128)
        num_lstm_layers = kwargs.get('num_lstm_layers', 1)

        # Encoders
        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim,
                                             dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim,
                                       dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.h0_c0_from_rgcn_encoder = MLPEncoder(rgcn_output_dim, self.lstm_hidden_dim,
                                                  dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))

        # RGCN modules
        self.rgcn_module_for_h0 = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, rgcn_output_dim,
                                             self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))
        self.rgcn_module_for_sequence = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, rgcn_output_dim,
                                                   self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))

        # Fusion MLP
        fusion_input_dim = rgcn_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(
            in_dim=fusion_input_dim,
            out_dim=fusion_output_dim,
            hid_dim=kwargs.get('fusion_mlp_hidden_dim'),
            dropout_rate=kwargs.get('dropout_rate_fusion_mlp', 0.2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=fusion_output_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0
        )

        # Prediction Head
        self.single_prediction_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
            nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), self.T_pred_horizon)
        )

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        # 1. Compute initial hidden state (h0, c0) from the first timestep
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        rgcn_out_t0 = self.rgcn_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr)
        h0 = self.h0_c0_from_rgcn_encoder(rgcn_out_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0 = h0.repeat(self.lstm.num_layers, 1, 1)
            c0 = c0.repeat(self.lstm.num_layers, 1, 1)

        # 2. Prepare the input sequence for the LSTM
        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)

            rgcn_out_t = self.rgcn_module_for_sequence(norm_x_t, pyg_batch_t.edge_index, pyg_batch_t.edge_attr)

            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)
            env_emb_t_expanded = env_emb_t[pyg_batch_t.batch]

            time_feat_t = timeline_time_features[t_idx, :].to(device)
            time_emb_t = self.time_encoder(time_feat_t)
            time_emb_t_expanded = time_emb_t.unsqueeze(0).expand(pyg_batch_t.num_nodes, -1)

            concat_features = torch.cat([rgcn_out_t, env_emb_t_expanded, time_emb_t_expanded], dim=-1)
            fused_features = self.fusion_mlp(concat_features)
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)

        # 3. Run the LSTM
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        # 4. Make predictions using the last hidden state
        last_lstm_output = lstm_out[:, -1, :]
        predictions = self.single_prediction_head(last_lstm_output)

        return predictions


class RGCNGRUModelWithHourlyHeads(nn.Module):
    """
    基准模型: RGCN-GRU。

    该模型使用GRU替代LSTM，并为每个预测时间步使用一个独立的预测头。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')
        self.num_relations = kwargs.get('num_relations', 5)

        gru_hidden_dim = kwargs.get('gru_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        rgcn_hidden_dim = kwargs.get('rgcn_hidden_dim', 128)
        rgcn_output_dim = kwargs.get('rgcn_output_dim', 128)
        num_gru_layers = kwargs.get('num_gru_layers', 1)

        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim,
                                             dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim,
                                       dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.h0_from_rgcn_encoder = MLPEncoder(rgcn_output_dim, gru_hidden_dim,
                                               dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))

        self.rgcn_module_for_h0 = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, rgcn_output_dim,
                                             self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))
        self.rgcn_module_for_sequence = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, rgcn_output_dim,
                                                   self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))

        fusion_input_dim = rgcn_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(in_dim=fusion_input_dim, out_dim=fusion_output_dim,
                                     hid_dim=kwargs.get('fusion_mlp_hidden_dim'),
                                     dropout_rate=kwargs.get('dropout_rate_fusion_mlp', 0.2))

        self.gru = nn.GRU(input_size=fusion_output_dim, hidden_size=gru_hidden_dim, num_layers=num_gru_layers,
                          batch_first=True, dropout=kwargs.get('dropout_rate_gru', 0.2) if num_gru_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        rgcn_out_t0 = self.rgcn_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr)
        h0 = self.h0_from_rgcn_encoder(rgcn_out_t0).unsqueeze(0)
        if self.gru.num_layers > 1:
            h0 = h0.repeat(self.gru.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            rgcn_out_t = self.rgcn_module_for_sequence(norm_x_t, pyg_batch_t.edge_index, pyg_batch_t.edge_attr)
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)
            env_emb_t_expanded = env_emb_t[pyg_batch_t.batch]
            time_feat_t = timeline_time_features[t_idx, :].to(device)
            time_emb_t = self.time_encoder(time_feat_t)
            time_emb_t_expanded = time_emb_t.unsqueeze(0).expand(pyg_batch_t.num_nodes, -1)
            concat_features = torch.cat([rgcn_out_t, env_emb_t_expanded, time_emb_t_expanded], dim=-1)
            fused_features = self.fusion_mlp(concat_features)
            lstm_input_sequence.append(fused_features)

        stacked_gru_input = torch.stack(lstm_input_sequence, dim=1)
        gru_out, _ = self.gru(stacked_gru_input, h0)

        hourly_preds = [self.hourly_prediction_heads[t](gru_out[:, t, :]).squeeze(-1) for t in
                        range(self.T_pred_horizon)]
        return torch.stack(hourly_preds, dim=1)


class PositionalEncoding(nn.Module):
    """为Transformer注入序列位置信息。"""

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [sequence_length, batch_size, feature_dim]
        """
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class RGCNTransformerModelWithHourlyHeads(nn.Module):
    """
    基准模型: RGCN-Transformer。

    该模型使用标准的Transformer Encoder-Decoder架构替代RNN来建模时间动态。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')
        self.num_relations = kwargs.get('num_relations', 5)
        self.rgcn_output_dim = kwargs.get('rgcn_output_dim', 128)
        self.num_encoder_obs_steps = kwargs.get('num_encoder_obs_steps', 1)

        # Transformer parameters
        d_model = kwargs.get('transformer_d_model', 128)
        n_head = kwargs.get('transformer_nhead', 4)
        num_enc_layers = kwargs.get('transformer_num_encoder_layers', 2)
        num_dec_layers = kwargs.get('transformer_num_decoder_layers', 2)
        dim_ff = kwargs.get('transformer_dim_feedforward', d_model * 4)
        transformer_dropout = kwargs.get('transformer_dropout_rate', 0.2)

        self.d_model = d_model

        # Encoders
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim,
                                             dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim,
                                       dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))

        # RGCN modules
        rgcn_hidden_dim = kwargs.get('rgcn_hidden_dim', 128)
        self.rgcn_module_for_encoder_inputs = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, self.rgcn_output_dim,
                                                         self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))
        self.rgcn_module_for_decoder_inputs = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, self.rgcn_output_dim,
                                                         self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))

        # Fusion MLP to match d_model
        fusion_input_dim = self.rgcn_output_dim + global_env_emb_dim + time_emb_dim
        self.fusion_mlp = MLPEncoder(
            in_dim=fusion_input_dim, out_dim=self.d_model,
            hid_dim=kwargs.get('fusion_mlp_hidden_dim'),
            dropout_rate=kwargs.get('dropout_rate_fusion_mlp', 0.1)
        )

        self.pos_encoder = PositionalEncoding(self.d_model, transformer_dropout)

        # Transformer Encoder and Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_head, dim_feedforward=dim_ff,
                                                   dropout=transformer_dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=n_head, dim_feedforward=dim_ff,
                                                   dropout=transformer_dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        # Prediction Heads
        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _process_one_timestep_input(self, pyg_batch, rgcn_module, time_feature, device):
        norm_x = (pyg_batch.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        rgcn_out = rgcn_module(norm_x, pyg_batch.edge_index, pyg_batch.edge_attr)
        env_emb = self.global_env_encoder(pyg_batch.graph_global_env_features)[pyg_batch.batch]
        time_emb = self.time_encoder(time_feature.to(device)).unsqueeze(0).expand(pyg_batch.num_nodes, -1)
        fused = self.fusion_mlp(torch.cat([rgcn_out, env_emb, time_emb], dim=-1))
        return fused

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        # Encoder processing
        encoder_inputs = [self._process_one_timestep_input(list_of_batched_timesteps[i].to(device),
                                                           self.rgcn_module_for_encoder_inputs,
                                                           torch.zeros(self.time_in_dim), device) for i in
                          range(self.num_encoder_obs_steps)]
        src = torch.stack(encoder_inputs, dim=0)  # Shape: (L_enc, N_total, d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        # Decoder processing
        decoder_inputs = [
            self._process_one_timestep_input(list_of_batched_timesteps[self.num_encoder_obs_steps + i].to(device),
                                             self.rgcn_module_for_decoder_inputs, timeline_time_features[i], device) for
            i in range(self.T_pred_horizon)]
        tgt = torch.stack(decoder_inputs, dim=0)  # Shape: (L_dec, N_total, d_model)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self._generate_square_subsequent_mask(self.T_pred_horizon, device)
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # Shape: (L_dec, N_total, d_model)

        # Prediction
        preds = [self.hourly_prediction_heads[t](decoder_output[t, :, :]).squeeze(-1) for t in
                 range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GCN-LSTM
# ==============================================================================
class GCNModule(nn.Module):
    """一个三层的图卷积网络 (GCN) 模块。"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.prelu3 = nn.PReLU(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight=None):
        if x.size(0) == 0: return x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn1(x)
        x = self.prelu1(x);
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn2(x)
        x = self.prelu2(x);
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn3(x)
        x = self.prelu3(x)
        return x


class GCNLSTMModelWithHourlyHeads(nn.Module):
    """
    基准模型: GCN-LSTM。

    该模型使用同构的GCN替代RGCN来处理空间信息。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        gcn_hidden_dim = kwargs.get('gcn_hidden_dim', 128)
        gcn_output_dim = kwargs.get('gcn_output_dim', 128)

        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        self.h0_c0_from_gcn_encoder = MLPEncoder(gcn_output_dim, lstm_hidden_dim)
        self.gcn_module_for_h0 = GCNModule(self.static_node_in_dim, gcn_hidden_dim, gcn_output_dim,
                                           kwargs.get('dropout_rate_gcn', 0.3))
        self.gcn_module_for_sequence = GCNModule(self.static_node_in_dim, gcn_hidden_dim, gcn_output_dim,
                                                 kwargs.get('dropout_rate_gcn', 0.3))

        fusion_input_dim = gcn_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))

        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        gcn_out_t0 = self.gcn_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index,
                                            getattr(pyg_batch_t0, 'edge_weight', None))
        h0 = self.h0_c0_from_gcn_encoder(gcn_out_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            gcn_out_t = self.gcn_module_for_sequence(norm_x_t, pyg_batch_t.edge_index,
                                                     getattr(pyg_batch_t, 'edge_weight', None))
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([gcn_out_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GINE-LSTM
# ==============================================================================
class GINEModule(nn.Module):
    """一个三层的图同构网络 (GINE) 模块。"""

    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.prelu3 = nn.PReLU(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        if x.size(0) == 0: return x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn1(x)
        x = self.prelu1(x);
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn2(x)
        x = self.prelu2(x);
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn3(x)
        x = self.prelu3(x)
        x = self.lin_out(x)
        return x


class GINELSTMModelWithHourlyHeads(nn.Module):
    """
    基准模型: GINE-LSTM。

    该模型使用GINE替代RGCN，使其能够处理边特征。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        gine_hidden_dim = kwargs.get('gine_hidden_dim', 128)
        gine_output_dim = kwargs.get('gine_output_dim', 128)
        gine_edge_dim = kwargs.get('gine_edge_dim', 6)

        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        self.h0_c0_from_gine_encoder = MLPEncoder(gine_output_dim, lstm_hidden_dim)
        self.gine_module_for_h0 = GINEModule(self.static_node_in_dim, gine_hidden_dim, gine_output_dim, gine_edge_dim,
                                             kwargs.get('dropout_rate_gine', 0.3))
        self.gine_module_for_sequence = GINEModule(self.static_node_in_dim, gine_hidden_dim, gine_output_dim,
                                                   gine_edge_dim, kwargs.get('dropout_rate_gine', 0.3))

        fusion_input_dim = gine_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))

        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        gine_out_t0 = self.gine_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr.float())
        h0 = self.h0_c0_from_gine_encoder(gine_out_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            gine_out_t = self.gine_module_for_sequence(norm_x_t, pyg_batch_t.edge_index, pyg_batch_t.edge_attr.float())
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([gine_out_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


class PositionalEncoding(nn.Module):
    """为Transformer注入序列位置信息。"""

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class RGCNTransformerModelWithHourlyHeads(nn.Module):
    """基准模型: RGCN-Transformer。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')
        self.num_relations = kwargs.get('num_relations', 5)
        self.rgcn_output_dim = kwargs.get('rgcn_output_dim', 128)
        self.num_encoder_obs_steps = kwargs.get('num_encoder_obs_steps', 1)

        d_model = kwargs.get('transformer_d_model', 128)
        n_head = kwargs.get('transformer_nhead', 4)
        num_enc_layers = kwargs.get('transformer_num_encoder_layers', 2)
        num_dec_layers = kwargs.get('transformer_num_decoder_layers', 2)
        dim_ff = kwargs.get('transformer_dim_feedforward', d_model * 4)
        transformer_dropout = kwargs.get('transformer_dropout_rate', 0.2)
        self.d_model = d_model

        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim,
                                             dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim,
                                       dropout_rate=kwargs.get('dropout_rate_encoders', 0.1))

        rgcn_hidden_dim = kwargs.get('rgcn_hidden_dim', 128)
        self.rgcn_module_for_encoder_inputs = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, self.rgcn_output_dim,
                                                         self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))
        self.rgcn_module_for_decoder_inputs = RGCNModule(self.static_node_in_dim, rgcn_hidden_dim, self.rgcn_output_dim,
                                                         self.num_relations, kwargs.get('dropout_rate_rgcn', 0.3))

        fusion_input_dim = self.rgcn_output_dim + global_env_emb_dim + time_emb_dim
        self.fusion_mlp = MLPEncoder(
            in_dim=fusion_input_dim, out_dim=self.d_model,
            hid_dim=kwargs.get('fusion_mlp_hidden_dim'),
            dropout_rate=kwargs.get('dropout_rate_fusion_mlp', 0.1)
        )

        self.pos_encoder = PositionalEncoding(self.d_model, transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_head, dim_feedforward=dim_ff,
                                                   dropout=transformer_dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=n_head, dim_feedforward=dim_ff,
                                                   dropout=transformer_dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _process_one_timestep_input(self, pyg_batch, rgcn_module, time_feature, device):
        norm_x = (pyg_batch.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        rgcn_out = rgcn_module(norm_x, pyg_batch.edge_index, pyg_batch.edge_attr)
        env_emb = self.global_env_encoder(pyg_batch.graph_global_env_features)[pyg_batch.batch]
        time_emb = self.time_encoder(time_feature.to(device)).unsqueeze(0).expand(pyg_batch.num_nodes, -1)
        fused = self.fusion_mlp(torch.cat([rgcn_out, env_emb, time_emb], dim=-1))
        return fused

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        encoder_inputs = [self._process_one_timestep_input(list_of_batched_timesteps[i].to(device),
                                                           self.rgcn_module_for_encoder_inputs,
                                                           torch.zeros(self.time_in_dim, device=device), device) for i
                          in range(self.num_encoder_obs_steps)]
        src = torch.stack(encoder_inputs, dim=0)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        decoder_inputs = [
            self._process_one_timestep_input(list_of_batched_timesteps[self.num_encoder_obs_steps + i].to(device),
                                             self.rgcn_module_for_decoder_inputs, timeline_time_features[i], device) for
            i in range(self.T_pred_horizon)]
        tgt = torch.stack(decoder_inputs, dim=0)
        tgt = self.pos_encoder(tgt)
        tgt_mask = self._generate_square_subsequent_mask(self.T_pred_horizon, device)
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        preds = [self.hourly_prediction_heads[t](decoder_output[t, :, :]).squeeze(-1) for t in
                 range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GCN-LSTM
# ==============================================================================
class GCNModule(nn.Module):
    """一个三层的图卷积网络 (GCN) 模块。"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.prelu3 = nn.PReLU(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight=None):
        if x.size(0) == 0: return x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn1(x)
        x = self.prelu1(x);
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn2(x)
        x = self.prelu2(x);
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1: x = self.bn3(x)
        x = self.prelu3(x)
        return x


class GCNLSTMModelWithHourlyHeads(nn.Module):
    """基准模型: GCN-LSTM。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        gcn_hidden_dim = kwargs.get('gcn_hidden_dim', 128)
        gcn_output_dim = kwargs.get('gcn_output_dim', 128)

        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        self.h0_c0_from_gcn_encoder = MLPEncoder(gcn_output_dim, lstm_hidden_dim)
        self.gcn_module_for_h0 = GCNModule(self.static_node_in_dim, gcn_hidden_dim, gcn_output_dim,
                                           kwargs.get('dropout_rate_gcn', 0.3))
        self.gcn_module_for_sequence = GCNModule(self.static_node_in_dim, gcn_hidden_dim, gcn_output_dim,
                                                 kwargs.get('dropout_rate_gcn', 0.3))

        fusion_input_dim = gcn_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))

        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        gcn_out_t0 = self.gcn_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index,
                                            getattr(pyg_batch_t0, 'edge_weight', None))
        h0 = self.h0_c0_from_gcn_encoder(gcn_out_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            gcn_out_t = self.gcn_module_for_sequence(norm_x_t, pyg_batch_t.edge_index,
                                                     getattr(pyg_batch_t, 'edge_weight', None))
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([gcn_out_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GINE-LSTM
# ==============================================================================
class GINEModule(nn.Module):
    """一个三层的图同构网络 (GINE) 模块。"""

    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.prelu3 = nn.PReLU(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        if x.size(0) == 0: return x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn1(x)
        x = self.prelu1(x);
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn2(x)
        x = self.prelu2(x);
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1: x = self.bn3(x)
        x = self.prelu3(x)
        x = self.lin_out(x)
        return x


class GINELSTMModelWithHourlyHeads(nn.Module):
    """基准模型: GINE-LSTM。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)
        gine_hidden_dim = kwargs.get('gine_hidden_dim', 128)
        gine_output_dim = kwargs.get('gine_output_dim', 128)
        gine_edge_dim = kwargs.get('gine_edge_dim', 6)

        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        self.h0_c0_from_gine_encoder = MLPEncoder(gine_output_dim, lstm_hidden_dim)
        self.gine_module_for_h0 = GINEModule(self.static_node_in_dim, gine_hidden_dim, gine_output_dim, gine_edge_dim,
                                             kwargs.get('dropout_rate_gine', 0.3))
        self.gine_module_for_sequence = GINEModule(self.static_node_in_dim, gine_hidden_dim, gine_output_dim,
                                                   gine_edge_dim, kwargs.get('dropout_rate_gine', 0.3))

        fusion_input_dim = gine_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))

        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        gine_out_t0 = self.gine_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr.float())
        h0 = self.h0_c0_from_gine_encoder(gine_out_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            gine_out_t = self.gine_module_for_sequence(norm_x_t, pyg_batch_t.edge_index, pyg_batch_t.edge_attr.float())
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([gine_out_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GAE-LSTM (使用 CGVAE 架构)
# ==============================================================================
class RGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, dropout_rate, use_residual=True):
        super().__init__()
        self.conv = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_projection = nn.Linear(in_channels,
                                                 out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x_input, edge_index, edge_attr):
        if x_input.size(0) == 0: return x_input
        edge_type = edge_attr[:, 4].long() if edge_attr.shape[1] >= 5 else edge_attr[:, 0].long()
        h = self.conv(x_input, edge_index, edge_type=edge_type)
        if h.shape[0] > 1: h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.use_residual:
            h = h + self.residual_projection(x_input)
        return h


class Encoder_CGVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, num_relations, dropout_rate):
        super().__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(RGCNBlock(current_dim, hidden_dim, num_relations, dropout_rate,
                                    use_residual=(i > 0 and current_dim == hidden_dim)))
            current_dim = hidden_dim
        self.rgcn_layers = nn.ModuleList(layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.rgcn_layers:
            x = layer(x, edge_index, edge_attr)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder_CGVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, original_node_feature_dim, num_layers, num_relations,
                 dropout_rate):
        super().__init__()
        layers = []
        current_dim = latent_dim + original_node_feature_dim
        for i in range(num_layers):
            layers.append(RGCNBlock(current_dim, hidden_dim, num_relations, dropout_rate,
                                    use_residual=(i > 0 and current_dim == hidden_dim)))
            current_dim = hidden_dim
        self.rgcn_layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, x_original, edge_index, edge_attr):
        x = torch.cat([z, x_original], dim=1)
        for layer in self.rgcn_layers:
            x = layer(x, edge_index, edge_attr)
        return self.fc_out(x)


class RGCN_CGVAE_FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_feature_dim, num_encoder_layers, num_decoder_layers,
                 num_relations, dropout_rate):
        super().__init__()
        self.encoder = Encoder_CGVAE(input_dim, hidden_dim, latent_dim, num_encoder_layers, num_relations, dropout_rate)
        self.decoder = Decoder_CGVAE(latent_dim, hidden_dim, output_feature_dim, input_dim, num_decoder_layers,
                                     num_relations, dropout_rate)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index, edge_attr):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = self.reparameterize(mu, logvar)
        output_features = self.decoder(z, x, edge_index, edge_attr)
        return output_features, mu, logvar


class GAELSTMModelWithHourlyHeads(nn.Module):
    """基准模型: GAE-LSTM (基于CGVAE实现)。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)

        # GAE/CGVAE params
        cgvae_hidden_dim = kwargs.get('cgvae_hidden_dim', 128)
        cgvae_latent_dim = kwargs.get('cgvae_latent_dim', 64)
        cgvae_output_dim = kwargs.get('cgvae_output_dim', 128)
        num_relations = kwargs.get('num_relations', 5)

        self.h0_c0_from_gnn_encoder = MLPEncoder(cgvae_output_dim, lstm_hidden_dim)
        self.cgvae_module_for_h0 = RGCN_CGVAE_FeatureExtractor(
            input_dim=self.static_node_in_dim, hidden_dim=cgvae_hidden_dim, latent_dim=cgvae_latent_dim,
            output_feature_dim=cgvae_output_dim,
            num_encoder_layers=kwargs.get('cgvae_num_encoder_layers', 2),
            num_decoder_layers=kwargs.get('cgvae_num_decoder_layers', 2),
            num_relations=num_relations, dropout_rate=kwargs.get('cgvae_dropout_rate', 0.3)
        )
        self.cgvae_module_for_sequence = RGCN_CGVAE_FeatureExtractor(
            input_dim=self.static_node_in_dim, hidden_dim=cgvae_hidden_dim, latent_dim=cgvae_latent_dim,
            output_feature_dim=cgvae_output_dim,
            num_encoder_layers=kwargs.get('cgvae_num_encoder_layers', 2),
            num_decoder_layers=kwargs.get('cgvae_num_decoder_layers', 2),
            num_relations=num_relations, dropout_rate=kwargs.get('cgvae_dropout_rate', 0.3)
        )
        # ... (rest of the __init__ is identical to GINELSTMModelWithHourlyHeads)
        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        fusion_input_dim = cgvae_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))
        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)
        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])
        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        # This forward pass is for inference/downstream task, so KLD loss is handled in the training loop
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        # We only need the generated features, not mu/logvar for the main path
        features_t0, _, _ = self.cgvae_module_for_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr.float())
        h0 = self.h0_c0_from_gnn_encoder(features_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            features_t, _, _ = self.cgvae_module_for_sequence(norm_x_t, pyg_batch_t.edge_index,
                                                              pyg_batch_t.edge_attr.float())
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([features_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: GGAN-LSTM (使用 CGGAN 架构)
# ==============================================================================
class NodeFeatureGeneratorRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_relations, dropout_rate):
        super().__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(RGCNBlock(current_dim, hidden_dim, num_relations, dropout_rate,
                                    use_residual=(i > 0 and current_dim == hidden_dim)))
            current_dim = hidden_dim
        self.rgcn_layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.rgcn_layers:
            x = layer(x, edge_index, edge_attr)
        return self.fc_out(x)


class PredictionDiscriminatorRGCN(nn.Module):
    def __init__(self, original_node_feature_dim, prediction_dim, hidden_dim, num_layers, num_relations, dropout_rate):
        super().__init__()
        layers = []
        current_dim = original_node_feature_dim + prediction_dim
        for i in range(num_layers):
            layers.append(RGCNBlock(current_dim, hidden_dim, num_relations, dropout_rate,
                                    activation_fn=nn.LeakyReLU(0.2, inplace=True),
                                    use_residual=(i > 0 and current_dim == hidden_dim)))
            current_dim = hidden_dim
        self.rgcn_layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_original, y_candidate, edge_index, edge_attr):
        discriminator_input = torch.cat([x_original, y_candidate], dim=1)
        h = discriminator_input
        for layer in self.rgcn_layers:
            h = layer(h, edge_index, edge_attr)
        return self.fc_out(h)


class GGANLSTMModelWithHourlyHeads(nn.Module):  # Generator part
    """基准模型: GGAN-LSTM (生成器部分)。"""

    def __init__(self, **kwargs):
        super().__init__()
        # ... (This __init__ is identical to GAELSTMModelWithHourlyHeads, just replace cgvae_* keys with gen_*)
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_node_in_dim = kwargs.get('static_node_in_dim')
        self.global_env_in_dim = kwargs.get('global_env_in_dim')
        self.time_in_dim = kwargs.get('time_in_dim')

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 128)
        global_env_emb_dim = kwargs.get('global_env_emb_dim', 16)
        time_emb_dim = kwargs.get('time_emb_dim', 8)

        # Generator GNN params
        gen_hidden_dim = kwargs.get('gen_hidden_dim', 128)
        gen_output_dim = kwargs.get('gen_output_dim', 128)
        num_relations = kwargs.get('num_relations', 5)

        self.h0_c0_from_gnn_encoder = MLPEncoder(gen_output_dim, lstm_hidden_dim)
        self.node_feature_generator_h0 = NodeFeatureGeneratorRGCN(
            input_dim=self.static_node_in_dim, hidden_dim=gen_hidden_dim, output_dim=gen_output_dim,
            num_layers=kwargs.get('gen_num_layers', 2), num_relations=num_relations,
            dropout_rate=kwargs.get('gen_dropout_rate', 0.3)
        )
        self.node_feature_generator_sequence = NodeFeatureGeneratorRGCN(
            input_dim=self.static_node_in_dim, hidden_dim=gen_hidden_dim, output_dim=gen_output_dim,
            num_layers=kwargs.get('gen_num_layers', 2), num_relations=num_relations,
            dropout_rate=kwargs.get('gen_dropout_rate', 0.3)
        )

        # ... (rest is identical to GAELSTM)
        self.global_env_encoder = MLPEncoder(self.global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(self.time_in_dim, time_emb_dim)
        fusion_input_dim = gen_output_dim + global_env_emb_dim + time_emb_dim
        fusion_output_dim = kwargs.get('fusion_mlp_output_dim', 128)
        self.fusion_mlp = MLPEncoder(fusion_input_dim, fusion_output_dim, hid_dim=kwargs.get('fusion_mlp_hidden_dim'))
        num_lstm_layers = kwargs.get('num_lstm_layers', 1)
        self.lstm = nn.LSTM(input_size=fusion_output_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)
        self.hourly_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, kwargs.get('mlp_prediction_hidden_dim', 64)), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout_rate_pred_head', 0.2)),
                nn.Linear(kwargs.get('mlp_prediction_hidden_dim', 64), 1)
            ) for _ in range(self.T_pred_horizon)
        ])
        self.register_buffer('node_feat_mean', torch.zeros(self.static_node_in_dim))
        self.register_buffer('node_feat_std', torch.ones(self.static_node_in_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        norm_x_t0 = (pyg_batch_t0.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
        features_t0 = self.node_feature_generator_h0(norm_x_t0, pyg_batch_t0.edge_index, pyg_batch_t0.edge_attr.float())
        h0 = self.h0_c0_from_gnn_encoder(features_t0).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        if self.lstm.num_layers > 1:
            h0, c0 = h0.repeat(self.lstm.num_layers, 1, 1), c0.repeat(self.lstm.num_layers, 1, 1)

        lstm_input_sequence = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            norm_x_t = (pyg_batch_t.x - self.node_feat_mean) / (self.node_feat_std + 1e-8)
            features_t = self.node_feature_generator_sequence(norm_x_t, pyg_batch_t.edge_index,
                                                              pyg_batch_t.edge_attr.float())
            env_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features)[pyg_batch_t.batch]
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).expand(
                pyg_batch_t.num_nodes, -1)
            fused_features = self.fusion_mlp(torch.cat([features_t, env_emb_t, time_emb_t], dim=-1))
            lstm_input_sequence.append(fused_features)

        stacked_lstm_input = torch.stack(lstm_input_sequence, dim=1)
        lstm_out, _ = self.lstm(stacked_lstm_input, (h0, c0))

        preds = [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)]
        return torch.stack(preds, dim=1)


# ==============================================================================
#  新增基准模型: CGAN-LSTM (U-Net based)
# ==============================================================================
class UNetFeatureExtractorModule(nn.Module):
    def __init__(self, input_channels, output_feature_channels=32, encoder_channels=(64, 128, 256), middle_channels=256,
                 decoder_channels=(128, 64)):
        super().__init__()
        self.encoder1 = self._encoder_block(input_channels, encoder_channels[0], norm=False)
        self.encoder2 = self._encoder_block(encoder_channels[0], encoder_channels[1])
        self.encoder3 = self._encoder_block(encoder_channels[1], encoder_channels[2])
        self.middle = nn.Sequential(nn.Conv2d(encoder_channels[2], middle_channels, 3, 1, 1, bias=False),
                                    nn.InstanceNorm2d(middle_channels), nn.ReLU(inplace=True))
        self.decoder3 = self._decoder_block(middle_channels + encoder_channels[2], decoder_channels[0])
        self.decoder2 = self._decoder_block(decoder_channels[0] + encoder_channels[1], decoder_channels[1])
        self.final_feature_conv = nn.Conv2d(decoder_channels[1] + encoder_channels[0], output_feature_channels, 1)

    def _encoder_block(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True));
        return nn.Sequential(*layers)

    def _decoder_block(self, in_c, out_c, norm=True):
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True));
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.encoder1(x);
        e2 = self.encoder2(e1);
        e3 = self.encoder3(e2)
        m = self.middle(e3)
        d3 = self.decoder3(torch.cat([m, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        feature_map = self.final_feature_conv(torch.cat([d2, e1], dim=1))
        return F.interpolate(feature_map, size=x.size()[2:], mode='bilinear', align_corners=False)


class H0PixelFeatureExtractor(nn.Module):
    def __init__(self, static_feat_dim, unet_input_ch_for_h0_unet, unet_feature_output_channels_for_h0,
                 unet_enc_ch_list, unet_mid_ch, unet_dec_ch_list, lstm_hidden_dim, num_lstm_layers, **kwargs):
        super().__init__()
        self.num_lstm_layers = num_lstm_layers
        self.h0_unet_feature_extractor = UNetFeatureExtractorModule(
            input_channels=unet_input_ch_for_h0_unet, output_feature_channels=unet_feature_output_channels_for_h0,
            encoder_channels=unet_enc_ch_list, middle_channels=unet_mid_ch, decoder_channels=unet_dec_ch_list
        )
        self.h0_pixel_projector = MLPEncoder(in_dim=unet_feature_output_channels_for_h0, out_dim=lstm_hidden_dim,
                                             add_layer_norm=False)

    def forward(self, static_7am_image_normalized):
        unet_features_7am = self.h0_unet_feature_extractor(static_7am_image_normalized)
        B, C, H, W = unet_features_7am.shape
        pixel_features_7am = unet_features_7am.permute(0, 2, 3, 1).reshape(B * H * W, C)
        projected_pixel_features = self.h0_pixel_projector(pixel_features_7am)
        h0 = projected_pixel_features.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        return h0


class CGANLSTMModel(nn.Module):  # Generator part of CGAN-LSTM
    """基准模型: CGAN-LSTM (生成器部分)。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.T_pred_horizon = kwargs.get('T_pred_horizon', 12)
        self.static_feat_dim = kwargs.get('static_feat_dim')
        self.H_img, self.W_img = kwargs.get('H_img', 50), kwargs.get('W_img', 50)

        global_env_in_dim = kwargs.get('global_env_in_dim')
        time_in_dim = kwargs.get('time_in_dim')
        global_env_emb_dim = kwargs.get('global_env_emb_dim')
        time_emb_dim = kwargs.get('time_emb_dim')

        self.global_env_encoder = MLPEncoder(global_env_in_dim, global_env_emb_dim)
        self.time_encoder = MLPEncoder(time_in_dim, time_emb_dim)

        unet_input_channels = kwargs.get('unet_input_channels_after_fusion')
        self.pre_unet_fusion = nn.Sequential(
            nn.Conv2d(self.static_feat_dim + global_env_emb_dim + time_emb_dim, unet_input_channels, 1, 1, 0),
            nn.InstanceNorm2d(unet_input_channels), nn.ReLU(True)
        )
        self.unet_feature_extractor = UNetFeatureExtractorModule(
            input_channels=unet_input_channels,
            output_feature_channels=kwargs.get('unet_feature_output_channels'),
            encoder_channels=kwargs.get('unet_encoder_channels_list'),
            middle_channels=kwargs.get('unet_middle_channels_val'),
            decoder_channels=kwargs.get('unet_decoder_channels_list')
        )

        self.h0_pixel_feature_extractor = H0PixelFeatureExtractor(**kwargs)

        lstm_hidden_dim = kwargs.get('lstm_hidden_dim')
        num_lstm_layers = kwargs.get('num_lstm_layers')
        self.lstm = nn.LSTM(kwargs.get('unet_feature_output_channels'), lstm_hidden_dim, num_lstm_layers,
                            batch_first=True,
                            dropout=kwargs.get('dropout_rate_lstm', 0.2) if num_lstm_layers > 1 else 0.0)

        self.hourly_prediction_heads = nn.ModuleList([
            MLPEncoder(lstm_hidden_dim, 1, kwargs.get('mlp_prediction_hidden_dim'), activation_fn=None,
                       add_layer_norm=False)
            for _ in range(self.T_pred_horizon)
        ])

        self.register_buffer('static_image_feat_mean', torch.zeros(self.static_feat_dim))
        self.register_buffer('static_image_feat_std', torch.ones(self.static_feat_dim))

    def forward(self, list_of_batched_timesteps: list, timeline_time_features: torch.Tensor, device: torch.device):
        pyg_batch_t0 = list_of_batched_timesteps[0].to(device)
        B, nodes_per_graph = pyg_batch_t0.num_graphs, self.H_img * self.W_img

        static_x_flat = pyg_batch_t0.x
        norm_static_x_flat = (static_x_flat - self.static_image_feat_mean.to(device)) / (
                    self.static_image_feat_std.to(device) + 1e-8)
        norm_static_x_img = norm_static_x_flat.view(B, nodes_per_graph, self.static_feat_dim).permute(0, 2,
                                                                                                      1).contiguous().view(
            B, self.static_feat_dim, self.H_img, self.W_img)

        h0 = self.h0_pixel_feature_extractor(norm_static_x_img);
        c0 = torch.zeros_like(h0, device=device)

        all_unet_features = []
        for t_idx in range(self.T_pred_horizon):
            pyg_batch_t = list_of_batched_timesteps[t_idx + 1].to(device)
            glob_emb_t = self.global_env_encoder(pyg_batch_t.graph_global_env_features).unsqueeze(-1).unsqueeze(
                -1).expand(-1, -1, self.H_img, self.W_img)
            time_emb_t = self.time_encoder(timeline_time_features[t_idx].to(device)).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1).expand(B, -1, self.H_img, self.W_img)
            fused_img_t = self.pre_unet_fusion(torch.cat([norm_static_x_img, glob_emb_t, time_emb_t], dim=1))
            unet_feat_map_t = self.unet_feature_extractor(fused_img_t)
            pixel_feats_t = unet_feat_map_t.permute(0, 2, 3, 1).reshape(B * nodes_per_graph, -1)
            all_unet_features.append(pixel_feats_t)

        lstm_input = torch.stack(all_unet_features, dim=1)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        preds_flat = torch.stack(
            [self.hourly_prediction_heads[t](lstm_out[:, t, :]).squeeze(-1) for t in range(self.T_pred_horizon)], dim=1)
        return preds_flat


class PatchGANDiscriminator(nn.Module):
    """判别器模型 (PatchGAN)。"""

    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(
            self._disc_block(input_channels, 64, stride=2, normalize=False),
            self._disc_block(64, 128, stride=2), self._disc_block(128, 256, stride=2),
            self._disc_block(256, 512, stride=1, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def _disc_block(self, in_c, out_c, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, stride, padding, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True));
        return nn.Sequential(*layers)

    def forward(self, x): return self.model(x)
