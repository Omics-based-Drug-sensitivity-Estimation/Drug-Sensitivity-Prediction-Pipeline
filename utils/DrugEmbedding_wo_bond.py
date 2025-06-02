import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Dict

from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn

# Multi-Head Attention with adjacency matrix
class AdjMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(AdjMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size, _, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (batch, num_heads, seq_len, seq_len)
        adj_matrix = adj_matrix.view(batch_size, 1, seq_len, seq_len)

        scores = scores * adj_matrix
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(context)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = AdjMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, seq_len: int) -> torch.Tensor:
        attn_out = self.attention(x, adj_matrix, seq_len)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# Drug Embedding Model
class DrugEmbeddingModel(nn.Module):
    def __init__(self, input_feature_dim: int, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, seq_len: int):
        super(DrugEmbeddingModel, self).__init__()
        self.linear = nn.Linear(input_feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, bond_info: List[Tuple[int, int, str]]) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = self.linear(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        seq_len += 1
        new_adj_matrix = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        new_adj_matrix[:, 1:, 1:] = adj_matrix.clone()
        new_adj_matrix[:, 0, :] = 1
        new_adj_matrix[:, :, 0] = 1
        adj_matrix = new_adj_matrix.clone()

        # Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x, adj_matrix, seq_len)

        # Return full sequence (excluding CLS token)
        return x
    

if __name__ == "__main__":
    
    drug_sensitivity_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/10_fold_data/mixed/MixedSet_test_Fold0.csv'  # 예시 폴드 파일
    smiles_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/GEP_Wilcoxon_test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=drug_sensitivity_filepath,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,
        cnv_standardize=True,
        mut_standardize=True,
        drug_sensitivity_min_max=True,
        column_names=('drug', 'cell_line', 'IC50')
    )

    # DataLoader 설정
    batch_size = 4  # 배치 크기 설정
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DrugEmbeddingModel(78, embed_dim, num_heads, ff_dim, num_layers, 256).to(device)
    model.eval()

    # 첫 번째 배치만 테스트
    for batch in trainloader:
        drug_data, gep_data, cnv_data, mut_data, ic50 = batch
        x, adj_matrix, bond_info = drug_data

        # x와 adj_matrix는 이미 배치 차원을 포함하고 있음
        x = x.to(torch.float).to(device)
        adj_matrix = adj_matrix.to(device)

        with torch.no_grad():
            output = model(x, adj_matrix, bond_info)

        # 출력 형태 확인
        print("=== Output Check ===")
        print(f"Input tokens shape: {x.shape}")
        print(f"Input adj_matrix shape: {adj_matrix.shape}")
        print(f"Output shape: {output.shape}")
        break  # 첫 번째 배치만 처리하고 종료