import os
import json
import pickle
import torch
import pandas as pd
from yj.encoder3 import chemberta_collate
from yj.encoder3 import ChemBERTaOmicsDataset

from models.model3 import PASO_GEP_CNV_MUT
from utils.utils import get_device, get_log_molar

# 🔹 Fold index
fold = 0  # Fold 1 corresponds to index 0

# 🔹 경로 설정 (수정: 필요한 경우!)
training_name = 'bert_paso_crossattention_bgd_5e-7_notfreeze'
model_dir = os.path.join('result/model', training_name, f'Fold{fold+1}')
model_path = os.path.join(model_dir, 'weights', f'Fold_{fold+1}best_pearson_bgd-test.pt')
params_path = os.path.join(model_dir, 'TCGA_classifier_best_aucpr_GEP.json')

# 🔹 파라미터 로드
with open(params_path, 'r') as f:
    params = json.load(f)

# 🔹 Gene list (pathway list) 로드
with open('data/MUDICUS_Omic_619_pathways.pkl', 'rb') as f:
    pathway_list = pickle.load(f)

params.update({
    "number_of_genes": len(pathway_list),
})

# 🔹 Test Dataset 로드 (Fold 1)
test_dataset = ChemBERTaOmicsDataset(
    drug_sensitivity_csv=f'data/10_fold_data/mixed/MixedSet_test_Fold{fold}.csv',
    smiles_csv='data/CCLE-GDSC-SMILES.csv',
    gep_csv='data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv',
    cnv_csv='data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv',
    mut_csv='data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv',
    tokenizer_name="DeepChem/ChemBERTa-77M-MLM",
    max_len=128
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    collate_fn=chemberta_collate
)

# 🔹 모델 로드
device = get_device()
model = PASO_GEP_CNV_MUT(params).to(device)
model.load(model_path)
model.eval()

# 🔹 예측 수행 및 결과 저장
all_predictions = []
all_loglabels = []
all_cell_lines = []
all_drugs = []
all_cids = []

with torch.no_grad():
    for batch in test_loader:
        drug_data = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        gep = batch["gep"].to(device)
        cnv = batch["cnv"].to(device)
        mut = batch["mut"].to(device)

        # 예측 수행
        y_hat, pred_dict = model(drug_data, gep, cnv, mut)
        preds = pred_dict["log_micromolar_IC50"].cpu().numpy()

        # 실제값 (log IC50 스케일)
        log_labels = get_log_molar(batch["ic50"], ic50_max=params["drug_sensitivity_processing_parameters"]["parameters"]["max"], ic50_min=params["drug_sensitivity_processing_parameters"]["parameters"]["min"]).cpu().numpy()

        all_predictions.extend(preds)
        all_loglabels.extend(log_labels)
        all_cell_lines.extend(batch["cell_line"])
        all_drugs.extend(batch["drug"])
        all_cids.extend(batch["cid"])

# 🔹 DataFrame 생성
df = pd.DataFrame({
    "CellLine": all_cell_lines,
    "Drug": all_drugs,
    "CID": all_cids,
    "Prediction_logIC50": [float(x) for x in all_predictions],
    "Actual_logIC50": [float(x) for x in all_loglabels]
})

# 🔹 CSV 저장
output_csv_path = os.path.join(model_dir, 'results', f'fold{fold+1}_test_predictions.csv')
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, index=False)
print(f"✅ Prediction CSV saved to: {output_csv_path}")