
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from utils.layers import ContextAttentionLayer, dense_layer
from utils.utils import get_device, get_log_molar
from utils.DrugEmbedding import DrugEmbeddingModel

from yj.encoder import ChemEncoder
from yj.encoder import ChemBERTaOmicsDataset, chemberta_collate

import torch
import torch.nn as nn
from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from utils.CrossAttention import CrossAttentionModule
from utils.utils import get_device, get_log_molar
from yj.encoder import ChemEncoder

class PASO_GEP_CNV_MUT(nn.Module):
    def __init__(self, params):
        super(PASO_GEP_CNV_MUT, self).__init__()
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        # IC50 scaling
        self.min_max_scaling = bool(params.get('drug_sensitivity_processing_parameters', {}))
        if self.min_max_scaling:
            self.IC50_max = params['drug_sensitivity_processing_parameters']['parameters']['max']
            self.IC50_min = params['drug_sensitivity_processing_parameters']['parameters']['min']

        # ChemBERTa Encoder
        self.chemberta_encoder = ChemEncoder(
            params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'),
            freeze=not params.get('train_backbone', False)
        ).to(self.device)

        # Model Hyperparameters
        self.hidden_dim = params.get('smiles_embedding_size', 384)
        self.n_heads = params.get('n_heads', 4)
        self.num_layers = params.get('num_layers', 4)
        self.dropout = params.get('dropout', 0.5)

        # Cross-Attention Layers
        self.cross_attention_gep = CrossAttentionModule(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.n_heads,
            dropout=self.dropout
        )
        self.cross_attention_cnv = CrossAttentionModule(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.n_heads,
            dropout=self.dropout
        )
        self.cross_attention_mut = CrossAttentionModule(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.n_heads,
            dropout=self.dropout
        )

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, drug_data, gep, cnv, mut):
        # Drug data from ChemBERTaOmicsDataset (input_ids, attention_mask)
        ids = drug_data['input_ids'].to(self.device)
        mask = drug_data['attention_mask'].to(self.device)

        # ChemBERTa embeddings: [B, L, H]
        embedded_smiles = self.chemberta_encoder(ids, mask)

        # Reduce SMILES token dimension via mean pooling (or CLS token)
        smiles_repr = embedded_smiles.mean(dim=1)  # [B, H]

        # Cross-Attention
        gep = gep.to(self.device)
        cnv = cnv.to(self.device)
        mut = mut.to(self.device)

        drug_gep, _ = self.cross_attention_gep(smiles_repr.unsqueeze(1), gep)
        drug_cnv, _ = self.cross_attention_cnv(smiles_repr.unsqueeze(1), cnv)
        drug_mut, _ = self.cross_attention_mut(smiles_repr.unsqueeze(1), mut)

        fused = torch.cat([drug_gep, drug_cnv, drug_mut], dim=1)
        predictions = self.output_mlp(fused)

        pred_dict = {}
        if not self.training:
            pred_dict.update({
                'IC50': predictions,
                'log_micromolar_IC50': get_log_molar(predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min)
                if self.min_max_scaling else predictions
            })

        return predictions.squeeze(-1), pred_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat.view_as(y), y)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))