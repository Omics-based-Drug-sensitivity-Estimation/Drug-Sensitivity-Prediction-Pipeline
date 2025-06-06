# 🔬 Drug Sensitivity Prediction Pipeline

A modular pipeline for drug sensitivity (IC50) prediction using multi-omics data and advanced drug representations. This project builds upon the baseline model introduced in the 2025 paper, [*"Anticancer drug response prediction integrating multi-omics pathway-based difference features and multiple deep learning techniques."*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012905)

## 📋 Overview

### Drug Sensitivity Estimation
- IC50 (half maximal inhibitory concentration) represents the concentration of a drug required to inhibit 50% of cell viability or activity
- Critical for personalized medicine and drug discovery
- Combines multi-omics data with drug structural information for accurate prediction

### Data Sources
1. **Omics Data** (from CCLE database)
   - 688 cell lines
     - Gene Expression (GEP)
     - Mutation (MUT)
     - Copy Number Variation (CNV)
   - 619 canonical pathways from MSigDB (c2_kegg_medicus)
   * The three omics datasets—**GEP**, **MUT**, and **CNV**—were post-processed to quantify statistically significant differences between *pathway-in* and *pathway-out* genes, as described in the referenced study ([PLOS Computational Biology, 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012905)).

<div align="center">

| Omics type | Final matrix shape |
| ---------- | ------------------ |
| GEP        | 1 × 619            |
| MUT        | 1 × 619            |
| CNV        | 1 × 619            |
</div>


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
<div align="center">
| Version | Drug Encoder      | Fusion Method              | Key Features                    |
|---------|-------------------|----------------------------|--------------------------------|
| `v1`    | ChemBERTa         | Context Attention + Dense  | Baseline attention fusion      |
| `v2`    | ChemBERTa         | Cross-Attention + MLP        | Enhanced interaction modeling  |
| `v3`    | ChemBERTa         | CLS +  Attention  | Applying CLS at the latest layer          |
| `v4`    | Graph Transformer | Context Attention + Dense  | Structural drug representation |
| `v5`    | Graph Transformer | Cross-Attention + MLP| Enhanced interaction modeling    |
</div>
   
## 🔧 Project Structure

```
PASO/
├── train/
│   └── train.py                    # Main training script
├── models/
│   ├── ChemBERT_Models.py         # v1-v3 implementations
│   ├── BGD_Models.py              # v4-v5 implementations
│   └── model.py                   # Model factory
├── data/
│   ├── TripleOmics_ChemBERT_Dataset.py   # ChemBERTa dataset
│   ├── TripleOmics_BGD_Dataset.py        # BGD dataset
│   ├── CCLE-GDSC-SMILES.csv              # Drug SMILES data
│   ├── MUDICUS_Omic_619_pathways.pkl     # Pathway data
│   ├── GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv
│   ├── MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv
│   └── CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv
├── utils/
│   ├── cross_attention.py         # Cross attention implementation
│   ├── drug_embedding.py          # Drug embedding modules
│   ├── layers.py                  # Network layers
│   ├── loss_functions.py          # Pearson, RMSE, R2 metrics
│   ├── hyperparams.py             # Optimizer and loss registries
│   └── utils.py                   # Device and scaling utilities
├── analysis.py                    # Data analysis and visualization
└── debug_utils.py                 # Debugging utilities
```

## 🚀 Quick Start

```bash
python train/train.py \
  --model_version v2 \
  --config_path configs/paso_v2_config.json
```

## ⚙️ Configuration

Configuration is managed via JSON files in the `configs/` directory. Here's a detailed example with comments:

```json
{
  // Data preprocessing settings
  "drug_sensitivity_min_max": true,        // Normalize IC50 values
  "augment_smiles": true,                  // Enable SMILES augmentation
  "smiles_start_stop_token": true,         // Add start/stop tokens to SMILES
  "number_of_genes": 619,                  // Number of pathways
  "smiles_padding_length": 128,            // Maximum SMILES sequence length

  // Model architecture parameters
  "smiles_embedding_size": 16,             // Drug embedding dimension
  "stacked_dense_hidden_sizes": [1536, 512, 128],  // MLP layer sizes
  "omics_dense_size": 64,                  // Omics embedding dimension
  "activation_fn": "relu",                 // Activation function
  "dropout": 0.3,                          // Dropout rate
  "batch_norm": true,                      // Enable batch normalization

  // CNN parameters for SMILES processing
  "filters": [64, 64, 64],                 // CNN filter sizes
  "kernel_sizes": [[3, 16], [5, 16], [11, 16]],  // CNN kernel sizes
  "smiles_attention_size": 64,             // Drug attention dimension
  "gene_attention_size": 1,                // Gene attention dimension

  // Multi-head attention parameters
  "molecule_gep_heads": [2, 2, 2, 2, 2],   // Heads for drug-GEP attention
  "molecule_cnv_heads": [2, 2, 2, 2, 2],   // Heads for drug-CNV attention
  "molecule_mut_heads": [2, 2, 2, 2, 2],   // Heads for drug-MUT attention
  "gene_heads": [1, 1, 1, 1, 1],          // Heads for gene attention
  "cnv_heads": [1, 1, 1, 1, 1],           // Heads for CNV attention
  "mut_heads": [1, 1, 1, 1, 1],           // Heads for MUT attention

  // Transformer parameters
  "n_heads": 2,                            // Number of attention heads
  "num_layers": 4,                         // Number of transformer layers

  // Training parameters
  "batch_size": 512,                       // Batch size
  "epochs": 200,                           // Maximum epochs
  "lr": 0.001,                             // Learning rate
  "optimizer": "adam",                     // Optimizer choice
  "loss_fn": "mse",                        // Loss function

  // Learning rate scheduling
  "scheduler": "plateau",                  // LR scheduler type
  "scheduler_kw": {                        // Scheduler parameters
    "patience": 3,                         // Epochs to wait before reducing LR
    "factor": 0.3                          // LR reduction factor
  },

  // Model saving and training control
  "train_backbone": false,                 // Whether to train backbone(for chemberta)
  "save_model": 25,                        // Save model every N epochs

  // Hardware settings
  "num_workers": 4,                        // Number of data loading workers
  "dataset_device": "cpu",                 // Device for dataset processing

  // Cross-validation settings
  "folds": 10,                             // Number of cross-validation folds
  "seed": 42,                              // Random seed

  // Additional settings
  "embed_scale_grad": false,               // Scale gradients for embeddings
  "smiles_vocabulary_size": 56,            // Size of SMILES vocabulary

  // IC50 processing parameters
  "drug_sensitivity_processing_parameters": {
    "processing": "min_max",               // Normalization method
    "parameters": {                        // Normalization range
      "min": -8.658382,
      "max": 13.107465
    }
  },

  // Dataset paths
  "train_dataset_args": {
    "drug_sens_csv": "data/10_fold_data/mixed/MixedSet_train_Fold0.csv",
    "smiles_csv": "data/CCLE-GDSC-SMILES.csv",
    "gep_csv": "data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv",
    "cnv_csv": "data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv",
    "mut_csv": "data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv",
    "standardise_omics": true,             // Standardize omics data
    "minmax_ic50": true                    // Normalize IC50 values
  },

  "test_dataset_args": {
    // Similar structure to train_dataset_args
    // Paths point to test set data
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

Maintainer: [Yoonjin Cho](https://github.com/darejinn), [GyungDeok Bae](https://github.com/bgduck33)
Part of multi-omics predictive modeling research at YAI(Yonsei Artificial Intelligence)
