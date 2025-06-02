# model.py
"""
PASO_GEP_CNV_MUT: Multi-omics + ChemBERTa Drug Response Prediction Model
- v1: Attention-based fusion with dense layers
- v2: CrossAttention module for fusion
- v3: CrossAttention blocks + CLS token fusion

Each model integrates:
- Drug (SMILES) embedding from ChemBERTa
- Gene Expression Profile (GEP)
- Copy Number Variation (CNV)
- Mutation (MUT)
For IC50 prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from data.TripleOmics_ChemBERT_Dataset import ChemEncoder
from utils.utils import get_device, get_log_molar
from utils.layers import ContextAttentionLayer, dense_layer
from utils.CrossAttention import CrossAttentionModule
from utils.hyperparams import LOSS_FN_FACTORY


##############################################
# PASO_GEP_CNV_MUT_v1: Attention-based fusion
##############################################
class PASO_GEP_CNV_MUT_v1(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        # Drug encoder (ChemBERTa)
        self.chem_encoder = ChemEncoder(
            params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'),
            freeze=not params.get('train_backbone', False)
        ).to(self.device)
        H = self.chem_encoder.hidden_size
        n_pw = params.get("number_of_pathways", 619)

        # Attention layers for each omics
        self.smiles_attention = nn.ModuleList([
            ContextAttentionLayer(H, params['smiles_padding_length'], 1, n_pw, params['smiles_attention_size'], temperature=params['molecule_temperature'])
            for _ in range(3)
        ])
        self.omics_dense = nn.ModuleList([
            nn.Sequential(nn.Linear(n_pw, params['omics_dense_size']), nn.ReLU(), nn.Dropout(params['dropout']))
            for _ in range(3)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(H * 3 + params['omics_dense_size'] * 3, 512),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(512, 1)
        )

    def forward(self, drug_data, gep, cnv, mut):
        ids, mask = drug_data['input_ids'].to(self.device), drug_data['attention_mask'].to(self.device)
        smiles_emb = self.chem_encoder(ids, mask)  # [B, L, H]
        gep, cnv, mut = [x.to(self.device).unsqueeze(-1) for x in (gep, cnv, mut)]

        attn_outs = []
        for i, omics in enumerate([gep, cnv, mut]):
            context, _ = self.smiles_attention[i](smiles_emb, omics)
            attn_outs.append(context)
        omics_dense = [dense(omics.squeeze(-1)) for dense, omics in zip(self.omics_dense, [gep, cnv, mut])]
        x = torch.cat(attn_outs + omics_dense, dim=1)
        return self.mlp(x).squeeze(-1), {}

    def loss(self, pred, target): return self.loss_fn(pred, target)


######################################################
# PASO_GEP_CNV_MUT_v2: CrossAttention module version
######################################################
class PASO_GEP_CNV_MUT_v2(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        self.chem_encoder = ChemEncoder(
            params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'),
            freeze=not params.get('train_backbone', False)
        ).to(self.device)
        H = self.chem_encoder.hidden_size

        self.cross_gep = CrossAttentionModule(H, params['num_layers'], params['n_heads'], params['dropout'])
        self.cross_cnv = CrossAttentionModule(H, params['num_layers'], params['n_heads'], params['dropout'])
        self.cross_mut = CrossAttentionModule(H, params['num_layers'], params['n_heads'], params['dropout'])

        self.mlp = nn.Sequential(
            nn.Linear(H * 3, 512), nn.ReLU(), nn.Dropout(params['dropout']), nn.Linear(512, 1)
        )

    def forward(self, drug_data, gep, cnv, mut):
        ids, mask = drug_data['input_ids'].to(self.device), drug_data['attention_mask'].to(self.device)
        smiles_emb = self.chem_encoder(ids, mask).mean(1)  # [B, H]
        gep, cnv, mut = [x.to(self.device) for x in (gep, cnv, mut)]

        gep_out, _ = self.cross_gep(smiles_emb.unsqueeze(1), gep)
        cnv_out, _ = self.cross_cnv(smiles_emb.unsqueeze(1), cnv)
        mut_out, _ = self.cross_mut(smiles_emb.unsqueeze(1), mut)

        x = torch.cat([gep_out, cnv_out, mut_out], dim=1)
        return self.mlp(x).squeeze(-1), {}

    def loss(self, pred, target): return self.loss_fn(pred, target)


##############################################################
# PASO_GEP_CNV_MUT_v3: CrossAttention + CLS token fusion
##############################################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, omics_dim, embed_dim, n_heads, dropout):
        super().__init__()
        self.q_proj = nn.Linear(omics_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, omics_vec, token_emb, key_padding_mask):
        q = self.q_proj(omics_vec).unsqueeze(1)
        ctx, w = self.attn(q, token_emb, token_emb, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        return ctx.squeeze(1), w.squeeze(1)

class PASO_GEP_CNV_MUT_v3(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        self.chem_encoder = ChemEncoder(
            params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'),
            freeze=not params.get('train_backbone', False)
        ).to(self.device)
        H = self.chem_encoder.hidden_size
        n_pw = params.get("number_of_pathways", 619)
        dense = params.get("omics_dense_size", 256)

        self.gep_lin = nn.Sequential(nn.Linear(n_pw, dense), nn.ReLU(), nn.Dropout(params['dropout']))
        self.cnv_lin = nn.Sequential(nn.Linear(n_pw, dense), nn.ReLU(), nn.Dropout(params['dropout']))
        self.mut_lin = nn.Sequential(nn.Linear(n_pw, dense), nn.ReLU(), nn.Dropout(params['dropout']))

        heads = params.get("n_heads", 8)
        self.attn_gep = CrossAttentionBlock(dense, H, heads, params['dropout'])
        self.attn_cnv = CrossAttentionBlock(dense, H, heads, params['dropout'])
        self.attn_mut = CrossAttentionBlock(dense, H, heads, params['dropout'])

        self.mlp = nn.Sequential(
            nn.Linear(H * 4, 512), nn.ReLU(), nn.Dropout(params['dropout']), nn.Linear(512, 1)
        )

    def forward(self, drug_tokens, gep, cnv, mut):
        ids, mask = drug_tokens["input_ids"].to(self.device), drug_tokens["attention_mask"].to(self.device)
        kpm = (mask == 0)
        tok_emb = self.chem_encoder(ids, mask)
        cls = tok_emb[:, 0]

        gep_ctx, _ = self.attn_gep(self.gep_lin(gep.to(self.device)), tok_emb, kpm)
        cnv_ctx, _ = self.attn_cnv(self.cnv_lin(cnv.to(self.device)), tok_emb, kpm)
        mut_ctx, _ = self.attn_mut(self.mut_lin(mut.to(self.device)), tok_emb, kpm)

        x = torch.cat([cls, gep_ctx, cnv_ctx, mut_ctx], dim=1)
        return self.mlp(x).squeeze(-1), {}

    def loss(self, pred, target): return self.loss_fn(pred, target)


##############################################
# Model Factory for train.py
##############################################
MODEL_FACTORY = {
    "v1": PASO_GEP_CNV_MUT_v1,
    "v2": PASO_GEP_CNV_MUT_v2,
    "v3": PASO_GEP_CNV_MUT_v3,
}

