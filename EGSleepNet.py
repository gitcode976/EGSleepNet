import math
import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch


class MultiSpectralAttentionLayer1D(nn.Module):
    def __init__(self, channel, length, reduction=16, freq_sel_method='top16'):
        """Implement the IDCT function."""

class FcaBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, freq_sel_method_name='bot29'):
        """Implement the Channel Attention function."""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        """Implement the PE function."""

class EGSleepNet(nn.Module): ## 时间作为顶点，空间和频域作为特征
    
    def __init__(self, config):
        super(EGSleepNet, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

        self.dct_layer = FcaBasicBlock(29, 29) # 29是时间维度
        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

        self.hg1 = CosineHypergraphLayer2(128*3, 128*3, 30, 5)

        self.gat1 = GATConv(config.dim_model * 3, config.dim_model * 3 // config.gat_heads, heads=config.gat_heads, concat=True, dropout=config.dropout)
        self.gat2 = GATConv(config.dim_model * 3, config.dim_model * 3, heads=1, concat=False, dropout=config.dropout)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
    # 每个单通道Transformer编码后的特征，用GNN学习
    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        combined_x1 = torch.cat([x2, x3], dim=-1)
        combined_x1 = self.fusion_layer(combined_x1)
        x1 = x1 + combined_x1

        combined_x2 = torch.cat([x1, x3], dim=-1)
        combined_x2 = self.fusion_layer(combined_x2)
        x2 = x2 + combined_x2

        combined_x3 = torch.cat([x1, x2], dim=-1)
        combined_x3 = self.fusion_layer(combined_x3)
        x3 = x3 + combined_x3

        x1 = self.transformer_encoder_1(x1)     # (batch_size, 29, 128)
        x2 = self.transformer_encoder_2(x2)
        x3 = self.transformer_encoder_3(x3)


        x = torch.cat([x1, x2, x3], dim=2) # (batch_size, 29, 384)
        x = self.dct_layer(x)

        x = self.drop(x)
        x = self.layer_norm(x)
        
        mean_row = x.mean(dim=1, keepdim=True) 
        x = torch.cat([x, mean_row], dim=1)
        residual = x
        
        x, H1 = self.hg1(x)
        x = F.relu(x)
        residual_1 = x
        
        A_global, edge_index = hypergraph_to_adjacency(H1)

        edge_index = edge_index.to(x.device)
        data_list = []
        for i in range(x.shape[0]):
            node_feat = x[i]  
            data = Data(x=node_feat, edge_index=edge_index)
            data_list.append(data)
        batch_graph = Batch.from_data_list(data_list)
        out = F.elu(self.gat1(batch_graph.x, batch_graph.edge_index)) # 145, 128
        out = F.elu(self.gat2(out, batch_graph.edge_index))

        residual_2 = out
        out = residual.view(-1, 384) + residual_1.view(-1,384) + residual_2
        out = global_mean_pool(out, batch_graph.batch)
        ###
        # x = self.position_multi(x)
        # x = self.transformer_encoder_multi(x)
        # x = self.layer_norm(x + residual)       # residual connection
        ###

        # x = x.view(x.size(0), -1)
        x = self.fc1(out)
        x = self.fc2(x)
        return x