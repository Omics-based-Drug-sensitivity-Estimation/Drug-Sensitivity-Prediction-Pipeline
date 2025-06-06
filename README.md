
# 🔬 Drug Sensitivity Prediction Pipeline
*Modular, end-to-end IC₅₀ prediction using multi-omics pathway features and advanced drug representations.*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **This project extends** the paper  
> *“Anticancer drug response prediction integrating multi-omics pathway-based difference features and multiple deep-learning techniques.”* (PLOS Comput Biol, 2024)
> 
---

## 📋 Table of Contents
1. [Background & Motivation](#background--motivation)  
2. [Data](#data)  
3. [Model Architecture](#model-architecture)  
4. [Repository Structure](#repository-structure)  
5. [Quick Start](#quick-start)  
6. [Configuration](#configuration)  
7. [Outputs & Logging](#outputs--logging)  
8. [Installation](#installation)  
9. [Results Snapshot](#results-snapshot)  
10. [Contributing & Contact](#contributing--contact)

---

## Background & Motivation

| Why IC₅₀ prediction matters | Our contribution |
|-----------------------------|------------------|
| • Essential for **precision oncology** & drug repositioning<br>• Requires coupling **drug chemistry** with **cell-specific biology** | • **Two interchangeable drug encoders** (ChemBERTa & graph transformer)<br>• **Bi-directional cross-attention** between drug & each omics layer (⇒ 6 interaction maps) |

---

## Data
### 1️⃣ Omics (CCLE, 688 cell lines)

| Omics type | Pre-processing | Final tensor |
|------------|---------------|--------------|
| GEP | Mann-Whitney U | **1 × 619** |
| MUT | χ²-G test | **1 × 619** |
| CNV | χ²-G test | **1 × 619** |

*Each entry stores the –log₁₀ P-value measuring pathway-in vs. pathway-out difference.*

### 2️⃣ Drugs (GDSC2, 233 compounds)
* SMILES strings  
* Matched IC₅₀ values for each *(drug, cell-line)* pair

---

## Model Architecture
### Drug Encoders
| Name | Versions | Output | Notes |
|------|----------|--------|-------|
| **ChemBERTa** | `v1–v3` | 1 × 384 | Pre-trained SMILES language model |
| **BGD** (graph) | `v4–v5` | 1 × 128 | Graph transformer + DeepChem atom/bond feats |

### Fusion Variants

| Version | Drug Encoder | Fusion | Highlight |
|---------|--------------|--------|-----------|
| `v1` | ChemBERTa | **Context-Attention** + MLP | Baseline |
| `v2` | ChemBERTa | **Cross-Attention** + MLP | Deeper interaction |
| `v3` | ChemBERTa | **CLS pooling** + Attention | Simpler, faster |
| `v4` | Graph | **Context-Attention** | Baseline |
| `v5` | Graph | **Cross-Attention** + MLP | Deeper interaction |

> **Cross-attention design**: drug ↔ omics (GEP, MUT, CNV) in both directions → 6 heads feeding a shared MLP for regression.

---

## Repository Structure
```text
PASO/
├── configs/
├── train/
│   └── train.py
├── models/
│   ├── ChemBERT_Models.py
│   ├── BGD_Models.py
│   └── model.py
├── data/
│   ├── TripleOmics_ChemBERT_Dataset.py
│   ├── TripleOmics_BGD_Dataset.py
│   └── … (omics & SMILES CSVs)
├── utils/
├── analysis.py
└── debug_utils.py
````

---

## Quick Start

```bash
# create & activate environment
conda env create -f environment.yml
conda activate paso_env

# train ChemBERTa cross-attention model
python train/train.py \
  --model_version v2 \
  --config_path configs/paso_v2_config.json
```

Replace `v2` with `v5` for the graph version.

---

## Configuration

Key fields (see full JSON in `configs/`):

```jsonc
{
  "smiles_padding_length": 128,
  "number_of_genes": 619,
  "dropout": 0.3,
  "epochs": 200,
  "batch_size": 512,
  "optimizer": "adam",
  "scheduler": "plateau",
  "folds": 10
}
```

---

## Outputs & Logging

```
results/
└── v2/
    ├── Fold1/
    │   ├── weights.pt
    │   └── metrics.json   # MSE, RMSE, Pearson, R²
    └── …
```

Run `analysis.py` for plots and attention heatmaps.

---

## Installation

```bash
git clone https://github.com/Omics-based-Drug-sensitivity-Estimation/Drug-Sensitivity-Prediction-Pipeline.git
cd Drug-Sensitivity-Prediction-Pipeline
conda env create -f environment.yml
conda activate paso_env
# or: pip install torch transformers rdkit-pypi deepchem dgl-lifesci pandas scikit-learn tqdm
```

---

## Results Snapshot


<p align="center">
  <img src="https://github.com/user-attachments/assets/62b4dbd0-f510-4b20-95e9-4789627cb7c5" width="425" alt="Figure 1">
  <br><em>Figure 1. Drug embedding comparison (Original vs. Modified attention)</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5f28a37e-4e18-4b83-8b3f-da5945e02404" width="425" alt="Figure 2">
  <br><em>Figure 2. Cross-attention variant performance</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a4b6bc42-3e90-4d4f-80c2-3668d2330e41" width="700" alt="Figure 3">
  <br><em>Figure 3. Pearson r on cell-blinded split (scatter)</em>
</p>

---

## Contributing & Contact

Pull requests and issues are welcome!

* **Yoonjin Cho** — [yoonjin.cho22@med.yuhs.ac](mailto:yoonjin.cho22@med.yuhs.ac) · [@darejinn](https://github.com/darejinn)
* **GyungDeok Bae** — [baegyungduck@gmail.com](mailto:baegyungduck@gmail.com) · [@bgduck33](https://github.com/bgduck33)

```

