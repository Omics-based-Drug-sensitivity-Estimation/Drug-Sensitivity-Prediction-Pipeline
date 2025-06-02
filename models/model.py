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


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Main Model
class PASO_GEP_CNV_MUT(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(PASO_GEP_CNV_MUT, self).__init__(*args, **kwargs)

        # Model Parameters
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get('drug_sensitivity_processing_parameters', {}) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params['drug_sensitivity_processing_parameters']['parameters']['max']
            self.IC50_min = params['drug_sensitivity_processing_parameters']['parameters']['min']
        
        # Drug Embedding Model
        self.chemberta_encoder = ChemEncoder(params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'), freeze=not params.get('train_backbone', False)).to(self.device)

        # Model Inputs(smiles_embedding_size 수정함)
        self.params['smiles_embedding_size'] = self.chemberta_encoder.hidden_size
        self.smiles_padding_length = params.get('smiles_padding_length', 256)
        self.number_of_pathways = params.get('number_of_pathways', 619)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model Architecture (Hyperparameters)
        self.molecule_gep_heads = params.get('molecule_gep_heads', [2])
        self.molecule_cnv_heads = params.get('molecule_cnv_heads', [2])
        self.molecule_mut_heads = params.get('molecule_mut_heads', [2])
        self.gene_heads = params.get('gene_heads', [1])
        self.cnv_heads = params.get('cnv_heads', [1])
        self.mut_heads = params.get('mut_heads', [1])
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.hidden_sizes = (
            [
                # Only use DrugEmbeddingModel output
                self.molecule_gep_heads[0] * params['smiles_embedding_size'] + # 6x256
                self.molecule_cnv_heads[0] * params['smiles_embedding_size'] +
                self.molecule_mut_heads[0] * params['smiles_embedding_size'] +
                sum(self.gene_heads) * self.omics_dense_size +
                sum(self.cnv_heads) * self.omics_dense_size +
                sum(self.mut_heads) * self.omics_dense_size
            ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get('activation_fn', 'relu')]

        # Attention Layers (single layer from embedding output)
        smiles_hidden_sizes = [params['smiles_embedding_size']]

        self.molecule_attention_layers_gep = nn.Sequential(OrderedDict([
            (
                f'molecule_gep_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_gep_heads[0])
        ]))

        self.molecule_attention_layers_cnv = nn.Sequential(OrderedDict([
            (
                f'molecule_cnv_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_cnv_heads[0])
        ]))

        self.molecule_attention_layers_mut = nn.Sequential(OrderedDict([
            (
                f'molecule_mut_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[0],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.molecule_temperature
                )
            ) for head in range(self.molecule_mut_heads[0])
        ]))

        self.gene_attention_layers = nn.Sequential(OrderedDict([
            (
                f'gene_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.gene_heads[0])
        ]))

        self.cnv_attention_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.cnv_heads[0])
        ]))

        self.mut_attention_layers = nn.Sequential(OrderedDict([
            (
                f'mut_attention_0_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[0],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get('context_nonlinearity', nn.Sequential()),
                    temperature=self.gene_temperature
                )
            ) for head in range(self.mut_heads[0])
        ]))

        self.gep_dense_layers = nn.Sequential(OrderedDict([
            (
                f'gep_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.gene_heads[0])
        ]))

        self.cnv_dense_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.cnv_heads[0])
        ]))

        self.mut_dense_layers = nn.Sequential(OrderedDict([
            (
                f'mut_dense_0_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for head in range(self.mut_heads[0])
        ]))

        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, drug_data, gep, cnv, mut):
        """
        Args:
            drug_data: Dict[str, Tensor])
            gep (torch.Tensor): Gene expression data, shape [bs, number_of_genes]
            cnv (torch.Tensor): Copy number variation data, shape [bs, number_of_genes]
            mut (torch.Tensor): Mutation data, shape [bs, number_of_genes]

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict
            predictions is IC50 drug sensitivity prediction of shape [bs, 1].
            prediction_dict includes the prediction and attention weights.
        """
        ids, mask = drug_data["input_ids"].to(self.device), drug_data["attention_mask"].to(self.device)
        embedded_smiles = self.chemberta_encoder(ids, mask)   # [B, L, H]

        gep = torch.unsqueeze(gep, dim=-1)  # [bs, number_of_genes, 1]
        cnv = torch.unsqueeze(cnv, dim=-1)  # [bs, number_of_genes, 1]
        mut = torch.unsqueeze(mut, dim=-1)  # [bs, number_of_genes, 1]
        gep = gep.to(device=self.device)
        cnv = cnv.to(device=self.device)
        mut = mut.to(device=self.device)
        

        # Validate output shape
        if embedded_smiles.shape[1] != self.smiles_padding_length or \
           embedded_smiles.shape[2] != self.params['smiles_embedding_size']:
            raise ValueError(
                f"Drug embedding output shape {embedded_smiles.shape} does not match "
                f"expected ([bs, {self.smiles_padding_length}, {self.params['smiles_embedding_size']}])"
            )

        # Use only the embedding output
        encoded_smiles = [embedded_smiles]

        # Molecule context attention
        encodings, smiles_alphas_gep, smiles_alphas_cnv, smiles_alphas_mut = [], [], [], []
        gene_alphas, cnv_alphas, mut_alphas = [], [], []
        for head in range(self.molecule_gep_heads[0]):
            e, a = self.molecule_attention_layers_gep[head](encoded_smiles[0], gep)
            encodings.append(e)
            smiles_alphas_gep.append(a)

        for head in range(self.molecule_cnv_heads[0]):
            e, a = self.molecule_attention_layers_cnv[head](encoded_smiles[0], cnv)
            encodings.append(e)
            smiles_alphas_cnv.append(a)

        for head in range(self.molecule_mut_heads[0]):
            e, a = self.molecule_attention_layers_mut[head](encoded_smiles[0], mut)
            encodings.append(e)
            smiles_alphas_mut.append(a)

        # Gene context attention
        for head in range(self.gene_heads[0]):
            e, a = self.gene_attention_layers[head](gep, encoded_smiles[0], average_seq=False)
            e = self.gep_dense_layers[head](e)
            encodings.append(e)
            gene_alphas.append(a)

        for head in range(self.cnv_heads[0]):
            e, a = self.cnv_attention_layers[head](cnv, encoded_smiles[0], average_seq=False)
            e = self.cnv_dense_layers[head](e)
            encodings.append(e)
            cnv_alphas.append(a)

        for head in range(self.mut_heads[0]):
            e, a = self.mut_attention_layers[head](mut, encoded_smiles[0], average_seq=False)
            e = self.mut_dense_layers[head](e)
            encodings.append(e)
            mut_alphas.append(a)

        encodings = torch.cat(encodings, dim=1) # (bs, 6 x 256, omics_dense x 3)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get('batch_norm', False) else encodings
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            smiles_attention_gep = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_gep], dim=-1)
            smiles_attention_cnv = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_cnv], dim=-1)
            smiles_attention_mut = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_mut], dim=-1)
            gene_attention = torch.cat([torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1)
            cnv_attention = torch.cat([torch.unsqueeze(p, -1) for p in cnv_alphas], dim=-1)
            mut_attention = torch.cat([torch.unsqueeze(p, -1) for p in mut_alphas], dim=-1)
            prediction_dict.update({
                'gene_attention': gene_attention,
                'cnv_attention': cnv_attention,
                'mut_attention': mut_attention,
                'smiles_attention_gep': smiles_attention_gep,
                'smiles_attention_cnv': smiles_attention_cnv,
                'smiles_attention_mut': smiles_attention_mut,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min)
                    if self.min_max_scaling else predictions
            })

        return predictions, prediction_dict

    def loss(self, yhat, y):
        if yhat.ndim == 2 and yhat.shape[1] == 1:
            yhat = yhat.squeeze(1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = yß.squeeze(1)
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        if not isinstance(smiles_language, pytoda.smiles.smiles_language.SMILESLanguage):
            raise TypeError(
                f'Please insert a smiles language (object of type '
                f'pytoda.smiles.smiles_language.SMILESLanguage). Given was {type(smiles_language)}'
            )
        self.smiles_language = smiles_language

    def load(self, path, *args, **kwargs):
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        torch.save(self.state_dict(), path, *args, **kwargs)






# Example Usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # File paths
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_train_Fold0.csv'
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    # Dataset
    dataset = ChemBERTaOmicsDataset(
            drug_sensitivity_csv=drug_sensitivity_filepath,
            smiles_csv=smiles_filepath,
            gep_csv=gep_filepath,
            cnv_csv=cnv_filepath,
            mut_csv=mut_filepath,
            tokenizer_name="DeepChem/ChemBERTa-77M-MLM",  # Optional
            max_len=128
        )
    # DataLoader
    batch_size = 4
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=chemberta_collate)

    # Model parameters
    params = {
        'smiles_padding_length': 128,
        'smiles_embedding_size': 384,
        'number_of_pathways': 619,
        'smiles_attention_size': 64,
        'gene_attention_size': 1,
        'molecule_temperature': 1.0,
        'gene_temperature': 1.0,
        'molecule_gep_heads': [2],
        'molecule_cnv_heads': [2],
        'molecule_mut_heads': [2],
        'gene_heads': [1],
        'cnv_heads': [1],
        'mut_heads': [1],
        'n_heads': 2,
        'num_layers': 4,
        'omics_dense_size': 256,
        'stacked_dense_hidden_sizes': [1024, 512],
        'dropout': 0.5,
        'temperature': 1.0,
        'activation_fn': 'relu',
        'batch_norm': True,
        'drug_sensitivity_processing_parameters': {
            'parameters': {'max': 100, 'min': 0}
        },
        'loss_fn': 'mse'
    }



######################################################################################################


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


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Enhanced Multi-Head Cross-Attention Module
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k or (d_model // n_heads)
        self.d_v = d_v or (d_model // n_heads)
        
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_v)
        self.W_O = nn.Linear(n_heads * self.d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)  # query의 시퀀스 길이
        seq_len_k = key.size(1)     # key의 시퀀스 길이
        
        # Linear transformations in batch from d_model => h x d_k
        Q = self.W_Q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, seq_len_k, self.n_heads, self.d_v).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores shape: [batch_size, n_heads, seq_len_q, seq_len_k]
        
        if mask is not None:
            # mask shape: [batch_size, seq_len_k]
            # scores shape: [batch_size, n_heads, seq_len_q, seq_len_k]
            # mask를 적절한 차원으로 확장
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len_k]
            mask = mask.expand(-1, self.n_heads, seq_len_q, -1)  # [batch_size, n_heads, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.n_heads * self.d_v
        )
        
        # Final linear transformation
        output = self.W_O(context)
        
        # Add & Norm - query와 output의 크기가 다를 수 있으므로 조건 확인
        if output.size() == query.size():
            output = self.layer_norm(output + query)
        else:
            output = self.layer_norm(output)
        
        return output, attention_weights


# Gated Fusion Module
class GatedFusionModule(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusionModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU()
        )
        
    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=-1)
        gate = self.gate(concat)
        transformed = self.transform(concat)
        return gate * x1 + (1 - gate) * transformed


# Cross-Modal Attention Block
class CrossModalAttentionBlock(nn.Module):
    def __init__(self, drug_dim, omics_dim, hidden_dim,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.drug_dim = drug_dim

        # omics_dim==drug_dim 이면 nn.Identity(), 아니면 nn.Linear()
        self.omics_projection = (
            nn.Linear(omics_dim, drug_dim)
            if omics_dim != drug_dim else nn.Identity()
        )


        # Bidirectional cross-attention
        self.drug_to_omics_attn = MultiHeadCrossAttention(drug_dim, n_heads, dropout=dropout)
        self.omics_to_drug_attn = MultiHeadCrossAttention(drug_dim, n_heads, dropout=dropout)
        
        # Gated fusion
        self.drug_fusion = GatedFusionModule(drug_dim)
        self.omics_fusion = GatedFusionModule(drug_dim)
        
        # Feed-forward networks
        self.drug_ffn = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, drug_dim)
        )
        self.omics_ffn = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, drug_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(drug_dim)
        self.layer_norm2 = nn.LayerNorm(drug_dim)
        
    def forward(self, drug_features, omics_features, drug_mask=None):
        # omics_features shape: [batch_size, num_genes, omics_dim]
        # drug_features shape: [batch_size, seq_len, drug_dim]
        
        # Ensure omics_features has correct shape
        if omics_features.dim() == 2:
            omics_features = omics_features.unsqueeze(-1)  # [batch_size, num_genes, 1]
        
        # Project omics features
        if omics_features.size(-1) == self.drug_dim:
            omics_proj = omics_features
        else:
            omics_proj = self.omics_projection(omics_features)
        # Cross-attention
        drug_attended, drug_attn_weights = self.drug_to_omics_attn(
            drug_features, omics_proj, omics_proj, mask=None
        )
        omics_attended, omics_attn_weights = self.omics_to_drug_attn(
            omics_proj, drug_features, drug_features, mask=drug_mask
        )

            
            
        # Gated fusion
        drug_fused = self.drug_fusion(drug_features, drug_attended)
        omics_fused = self.omics_fusion(omics_proj, omics_attended)
        
        # Feed-forward with residual connection
        drug_output = self.layer_norm1(drug_fused + self.drug_ffn(drug_fused))
        omics_output = self.layer_norm2(omics_fused + self.omics_ffn(omics_fused))
        
        return drug_output, omics_output, drug_attn_weights, omics_attn_weights


# Enhanced Main Model
class PASO_GEP_CNV_MUT_2(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super(PASO_GEP_CNV_MUT_2, self).__init__(*args, **kwargs)

        # Model Parameters
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get('drug_sensitivity_processing_parameters', {}) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params['drug_sensitivity_processing_parameters']['parameters']['max']
            self.IC50_min = params['drug_sensitivity_processing_parameters']['parameters']['min']
        
        # Drug Embedding Model
        self.chemberta_encoder = ChemEncoder(
            params.get('tokenizer_name', 'DeepChem/ChemBERTa-77M-MLM'), 
            freeze=not params.get('train_backbone', False)
        ).to(self.device)

        # Model Inputs
        self.params['smiles_embedding_size'] = self.chemberta_encoder.hidden_size
        self.smiles_padding_length = params.get('smiles_padding_length', 256)
        self.number_of_pathways = params.get('number_of_pathways', 619)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model Architecture (Hyperparameters)
        self.molecule_gep_heads = params.get('molecule_gep_heads', [2])
        self.molecule_cnv_heads = params.get('molecule_cnv_heads', [2])
        self.molecule_mut_heads = params.get('molecule_mut_heads', [2])
        self.gene_heads = params.get('gene_heads', [1])
        self.cnv_heads = params.get('cnv_heads', [1])
        self.mut_heads = params.get('mut_heads', [1])
        self.n_heads = params.get('n_heads', 4)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.cross_modal_hidden_dim = params.get('cross_modal_hidden_dim', 512)
        
        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get('activation_fn', 'relu')]

        # Cross-Modal Attention Blocks with omics_dim=1
        self.gep_cross_attention = nn.ModuleList([
            CrossModalAttentionBlock(
                drug_dim=self.params['smiles_embedding_size'],
                omics_dim=1,  # GEP features are 1-dimensional per gene
                hidden_dim=self.cross_modal_hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        self.cnv_cross_attention = nn.ModuleList([
            CrossModalAttentionBlock(
                drug_dim=self.params['smiles_embedding_size'],
                omics_dim=1,  # CNV features are 1-dimensional per gene
                hidden_dim=self.cross_modal_hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        self.mut_cross_attention = nn.ModuleList([
            CrossModalAttentionBlock(
                drug_dim=self.params['smiles_embedding_size'],
                omics_dim=1,  # MUT features are 1-dimensional per gene
                hidden_dim=self.cross_modal_hidden_dim,
                n_heads=self.n_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        # Feature aggregation layers
        self.drug_aggregation = nn.Sequential(
            nn.Linear(self.params['smiles_embedding_size'] * 3, self.params['smiles_embedding_size']),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.omics_aggregation = nn.Sequential(
            nn.Linear(self.params['smiles_embedding_size'] * 3, self.omics_dense_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Global attention pooling
        self.drug_global_attention = nn.Sequential(
            nn.Linear(self.params['smiles_embedding_size'], 1),
            nn.Softmax(dim=1)
        )
        
        self.omics_global_attention = nn.Sequential(
            nn.Linear(self.omics_dense_size, 1),
            nn.Softmax(dim=1)
        )

        # Final prediction layers
        fusion_dim = self.params['smiles_embedding_size'] + self.omics_dense_size
        self.hidden_sizes = (
            [fusion_dim] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )
        
        # Batch norm only when we have batches
        self.use_batch_norm = params.get('batch_norm', True)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )
        self.to(self.device)
    def forward(self, drug_data, gep, cnv, mut):
        ids, mask = drug_data["input_ids"].to(self.device), drug_data["attention_mask"].to(self.device)
        embedded_smiles = self.chemberta_encoder(ids, mask)   # [B, L, H]

        # Move omics data to device first
        gep = gep.to(device=self.device)
        cnv = cnv.to(device=self.device)
        mut = mut.to(device=self.device)
        
        # Ensure omics data has correct shape [batch_size, num_genes, 1]
        if gep.dim() == 2:
            gep = gep.unsqueeze(-1)  # [bs, number_of_genes, 1]
        if cnv.dim() == 2:
            cnv = cnv.unsqueeze(-1)  # [bs, number_of_genes, 1]
        if mut.dim() == 2:
            mut = mut.unsqueeze(-1)  # [bs, number_of_genes, 1]

        # Store attention weights
        gep_drug_attns, gep_omics_attns = [], []
        cnv_drug_attns, cnv_omics_attns = [], []
        mut_drug_attns, mut_omics_attns = [], []

        # Cross-modal attention layers
        drug_gep, drug_cnv, drug_mut = embedded_smiles, embedded_smiles, embedded_smiles
        omics_gep, omics_cnv, omics_mut = gep, cnv, mut
        
        for i in range(self.num_layers):
            drug_gep, omics_gep, d_attn, o_attn = self.gep_cross_attention[i](
                drug_gep, omics_gep, drug_mask=mask
            )
            gep_drug_attns.append(d_attn)
            gep_omics_attns.append(o_attn)
            
            drug_cnv, omics_cnv, d_attn, o_attn = self.cnv_cross_attention[i](
                drug_cnv, omics_cnv, drug_mask=mask
            )
            cnv_drug_attns.append(d_attn)
            cnv_omics_attns.append(o_attn)
            
            drug_mut, omics_mut, d_attn, o_attn = self.mut_cross_attention[i](
                drug_mut, omics_mut, drug_mask=mask
            )
            mut_drug_attns.append(d_attn)
            mut_omics_attns.append(o_attn)

        # Aggregate multi-modal drug features
        drug_features = torch.cat([drug_gep, drug_cnv, drug_mut], dim=-1)
        drug_features = self.drug_aggregation(drug_features)
        
        # Global attention pooling for drug features
        drug_attn_weights = self.drug_global_attention(drug_features)
        drug_pooled = torch.sum(drug_features * drug_attn_weights, dim=1)
        
        # Aggregate multi-modal omics features
        omics_features = torch.cat([omics_gep, omics_cnv, omics_mut], dim=-1)
        omics_features = self.omics_aggregation(omics_features)
        
        # Global attention pooling for omics features
        omics_attn_weights = self.omics_global_attention(omics_features)
        omics_pooled = torch.sum(omics_features * omics_attn_weights, dim=1)
        
        # Concatenate drug and omics features
        combined_features = torch.cat([drug_pooled, omics_pooled], dim=-1)
        
        # Apply batch normalization if specified and batch size > 1
        if self.use_batch_norm and combined_features.size(0) > 1:
            inputs = self.batch_norm(combined_features)
        else:
            inputs = combined_features
        
        # Dense layers
        for dl in self.dense_layers:
            inputs = dl(inputs)

        # Final prediction
        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            # Average attention weights across layers
            avg_gep_drug_attn = torch.stack(gep_drug_attns).mean(0)
            avg_cnv_drug_attn = torch.stack(cnv_drug_attns).mean(0)
            avg_mut_drug_attn = torch.stack(mut_drug_attns).mean(0)
            avg_gep_omics_attn = torch.stack(gep_omics_attns).mean(0)
            avg_cnv_omics_attn = torch.stack(cnv_omics_attns).mean(0)
            avg_mut_omics_attn = torch.stack(mut_omics_attns).mean(0)
            
            prediction_dict.update({
                'gene_attention': avg_gep_omics_attn,
                'cnv_attention': avg_cnv_omics_attn,
                'mut_attention': avg_mut_omics_attn,
                'smiles_attention_gep': avg_gep_drug_attn,
                'smiles_attention_cnv': avg_cnv_drug_attn,
                'smiles_attention_mut': avg_mut_drug_attn,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min)
                    if self.min_max_scaling else predictions
            })

        return predictions, prediction_dict

    def loss(self, yhat, y):
        # Ensure shapes match
        if yhat.dim() == 2 and yhat.shape[1] == 1:
            yhat = yhat.squeeze(1)
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        
        # Ensure same device
        if yhat.device != y.device:
            y = y.to(yhat.device)
            
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        if not isinstance(smiles_language, pytoda.smiles.smiles_language.SMILESLanguage):
            raise TypeError(
                f'Please insert a smiles language (object of type '
                f'pytoda.smiles.smiles_language.SMILESLanguage). Given was {type(smiles_language)}'
            )
        self.smiles_language = smiles_language

    def load(self, path, *args, **kwargs):
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        torch.save(self.state_dict(), path, *args, **kwargs)