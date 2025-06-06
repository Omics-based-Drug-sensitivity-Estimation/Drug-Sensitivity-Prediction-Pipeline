# PASO: Predictive Attention-based SMILES-Omics Model

A modular pipeline for drug sensitivity (IC50) prediction using multi-omics data and advanced drug representations. This project improves upon the baseline model from the 2025 paper "Anticancer drug response prediction integrating multi-omics pathway-based difference features and multiple deep learning techniques."

## 📋 Overview

### Drug Sensitivity Estimation
- IC50 (half maximal inhibitory concentration) represents the concentration of a drug required to inhibit 50% of cell viability or activity
- Critical for personalized medicine and drug discovery
- Combines multi-omics data with drug structural information for accurate prediction

### Data Sources
1. **Omics Data** (from CCLE database)
   - Gene Expression (GEP): TPM, log2(TPM+1)
   - Mutation (MUT): Binary (0/1/2)
   - Copy Number Variation (CNV): log2-transformed, discrete categorization
   - 688 cell lines
   - 619 canonical pathways from MSigDB (c2_kegg_medicus)

2. **Drug Data** (from GDSC2)
   - SMILES representations for 233 drugs
   - IC50 values for 688 cell lines

## 🧠 Model Architecture

### Drug Encoders
1. **ChemBERTa-based** (`v1`, `v2`, `v3`)
   - Pre-trained transformer model for SMILES sequences
   - Output: [128, 384] embedding per drug

2. **Graph-based (BGD)** (`v4`, `v5`)
   - Graph Neural Network processing
   - Incorporates bond orders, connectivity, and DeepChem features

### Model Versions

| Version | Drug Encoder      | Fusion Method              | Key Features                    |
|---------|-------------------|----------------------------|--------------------------------|
| `v1`    | ChemBERTa         | Context Attention + Dense  | Baseline attention fusion      |
| `v2`    | ChemBERTa         | Cross-Attention MLP        | Enhanced interaction modeling  |
| `v3`    | ChemBERTa         | CLS + Multihead Attention  | Token-level attention          |
| `v4`    | Graph Transformer | Context Attention + Dense  | Structural drug representation |
| `v5`    | Graph Transformer | Cross-Attention Transformer| Deepest fusion architecture    |

## 🔧 Project Structure

```
PASO/
├── train.py                        # Main training script
├── model.py                        # Model factory and definitions
├── DrugEmbedding.py               # Graph-based drug encoder
├── TripleOmics_Drug_Dataset.py    # BGD dataset handler
├── data/
│   ├── TripleOmics_ChemBERT_Dataset.py   # ChemBERTa dataset
│   └── ...
├── models/
│   ├── ChemBERT_models.py         # v1-v3 implementations
│   ├── BGD_models.py              # v4 implementation
│   └── BGD_model_v2.py           # v5 implementation
├── utils/
│   ├── utils.py                   # Device and scaling utilities
│   ├── loss_functions.py          # Pearson, RMSE, R2 metrics
│   ├── hyperparams.py             # Optimizer and loss registries
│   └── layers.py                  # Attention and dense layers
```

## 🚀 Quick Start

```bash
python train.py \
  --model_version v2 \
  --config_path configs/paso_v2_config.json
```

## ⚙️ Configuration

Configuration is managed via JSON files:

```json
{
  "fold": 5,
  "optimizer": "adam",
  "smiles_padding_length": 128,
  "smiles_embedding_size": 384,
  "number_of_pathways": 619,
  "loss_fn": "mse",
  "drug_sensitivity_processing_parameters": {
    "parameters": {"min": -8.65, "max": 13.1}
  },
  "train_dataset_args": {
    "drug_sensitivity_filepath": "data/...",
    "smiles_filepath": "data/...",
    "gep_filepath": "data/...",
    "cnv_filepath": "data/...",
    "mut_filepath": "data/..."
  }
}
```

## 📊 Output

- Best models per fold (saved in `result/model/.../weights`)
- Performance metrics (MSE, Pearson correlation, R²)
- Training logs and visualizations

## 🛠️ Dependencies

- Deep Learning: `torch`, `transformers`
- Chemistry: `rdkit`, `deepchem`
- Data Processing: `pandas`, `scikit-learn`
- Utilities: `tqdm`

## 📬 Contact

Maintainer: [Yoonjin Cho](mailto:yoonjincho@kaist.ac.kr)  
Part of multi-omics predictive modeling research at KAIST.
