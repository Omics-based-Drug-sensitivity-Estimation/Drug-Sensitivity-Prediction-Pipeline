# model.py
"""
PASO Model Factory
- v1, v2, v3: ChemBERTa-based models
- v4, v5: Graph-based BGD models
"""

import torch.nn as nn
from models.ChemBERT_Models import PASO_Chem_v1, PASO_Chem_v2, PASO_Chem_v3
from models.BGD_Models import PASO_BGD_v1
from models.BGD_Models import PASO_BGD_v2
MODEL_FACTORY = {
    "v1": PASO_Chem_v1,
    "v2": PASO_Chem_v2,
    "v3": PASO_Chem_v3,
    "v4": PASO_BGD_v1,       # Transformer + attention (graph-based)
    "v5": PASO_BGD_v2        # Transformer + cross-attention (graph-based)
}
