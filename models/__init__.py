from models.ChemBERT_Models import PASO_Chem_v1, PASO_Chem_v2, PASO_Chem_v3
from models.BGD_Models import PASO_BGD_v1, PASO_BGD_v2

MODEL_FACTORY = {
    "v1": PASO_Chem_v1,
    "v2": PASO_Chem_v2,
    "v3": PASO_Chem_v3,
    "v4": PASO_BGD_v1,
    "v5": PASO_BGD_v2,
}