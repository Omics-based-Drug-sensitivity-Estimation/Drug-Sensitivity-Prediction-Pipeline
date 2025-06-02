import sys
import os
import pandas as pd
import logging
import json
import pickle
from time import time
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yj.encoder import ChemBERTaOmicsDataset, chemberta_collate
from models.model import PASO_GEP_CNV_MUT
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score
from utils.utils import get_device, get_log_molar

def main(drug_sensitivity_filepath, gep_filepath, cnv_filepath, mut_filepath,
         smiles_filepath, gene_filepath, model_path, params, training_name):

    torch.backends.cudnn.benchmark = True
    params.update({"batch_size": 512, "epochs": 200, "num_workers": 4, "stacked_dense_hidden_sizes": [1024, 512]})

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Parameters: %s", params)

    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    n_folds = params.get("fold", 1)
    early_stopping_patience = params.get("early_stopping_patience", 20)
    lr_scheduler_patience = params.get("lr_scheduler_patience", 10)

    for fold in range(n_folds):
        logger.info("============== Fold [%d/%d] ==============", fold+1, n_folds)
        model_dir = os.path.join(model_path, training_name, f'Fold{fold+1}')
        os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)

        drug_sensitivity_train = f"{drug_sensitivity_filepath}train_Fold{fold}.csv"
        train_dataset = ChemBERTaOmicsDataset(drug_sensitivity_train, smiles_filepath, gep_filepath,
                                              cnv_filepath, mut_filepath, tokenizer_name="DeepChem/ChemBERTa-77M-MLM", max_len=128)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,
                                                   drop_last=True, num_workers=params["num_workers"], collate_fn=chemberta_collate)

        drug_sensitivity_test = f"{drug_sensitivity_filepath}test_Fold{fold}.csv"
        min_val = params["drug_sensitivity_processing_parameters"]["parameters"]["min"]
        max_val = params["drug_sensitivity_processing_parameters"]["parameters"]["max"]
        test_dataset = ChemBERTaOmicsDataset(drug_sensitivity_test, smiles_filepath, gep_filepath,
                                             cnv_filepath, mut_filepath, tokenizer_name="DeepChem/ChemBERTa-77M-MLM", max_len=128)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False,
                                                  drop_last=False, num_workers=params["num_workers"], collate_fn=chemberta_collate)

        logger.info("FOLD [%d] Training samples: %d, Testing samples: %d", fold+1, len(train_dataset), len(test_dataset))

        device = get_device()
        model = PASO_GEP_CNV_MUT({**params, "number_of_genes": len(pathway_list)}).to(device)
        model.train()

        optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](model.parameters(), lr=params.get("lr", 0.001))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=lr_scheduler_patience, factor=0.5, verbose=True)

        early_stop_counter = 0
        best_val_loss = float('inf')

        for epoch in range(params["epochs"]):
            logger.info("== Fold [%d/%d] Epoch [%d/%d] ==", fold+1, n_folds, epoch+1, params["epochs"])
            train(model, device, epoch, fold, train_loader, optimizer, params, max_val, min_val)

            val_loss, val_pearson, val_rmse, val_r2, _, _ = eval_model(model, device, test_loader, params, epoch, fold, max_val, min_val)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, "weights", "best_model.pt"))
                logger.info("New best model saved with val_loss: %.4f", val_loss)
            else:
                early_stop_counter += 1
                logger.info("EarlyStopping: %d/%d", early_stop_counter, early_stopping_patience)
                if early_stop_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

def train(model, device, epoch, fold, loader, optimizer, params, max_val, min_val):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Fold {fold+1} Epoch {epoch+1} Train"):
        optimizer.zero_grad()
        drug_data = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        y_hat, _ = model(drug_data, batch["gep"].to(device), batch["cnv"].to(device), batch["mut"].to(device))
        loss = model.loss(y_hat, batch["ic50"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        total_loss += loss.item()
    logging.info("Train Loss: %.5f", total_loss / len(loader))

def eval_model(model, device, loader, params, epoch, fold, max_val, min_val):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Fold {fold+1} Epoch {epoch+1} Eval"):
            drug_data = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
            y_hat, pred_dict = model(drug_data, batch["gep"].to(device), batch["cnv"].to(device), batch["mut"].to(device))
            loss = model.loss(y_hat, batch["ic50"].to(device))
            total_loss += loss.item()
            preds.extend(pred_dict["log_micromolar_IC50"].cpu())
            labels.extend(get_log_molar(batch["ic50"], ic50_max=max_val, ic50_min=min_val).cpu())

    preds, labels = torch.tensor(preds), torch.tensor(labels)
    pearson = pearsonr(preds, labels)
    rmse = torch.sqrt(torch.mean((preds - labels) ** 2))
    r2 = r2_score(preds, labels)
    logging.info("Eval Loss: %.5f, Pearson: %.4f, RMSE: %.4f, R2: %.4f", total_loss / len(loader), pearson, rmse, r2)
    return total_loss / len(loader), pearson, rmse, r2, preds, labels

if __name__ == "__main__":
    main(
        drug_sensitivity_filepath='data/10_fold_data/drug_blind/DrugBlind_',
        smiles_filepath='data/CCLE-GDSC-SMILES.csv',
        gep_filepath='data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv',
        cnv_filepath='data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv',
        mut_filepath='data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv',
        gene_filepath='data/MUDICUS_Omic_619_pathways.pkl',
        model_path='result/model',
        params={
            'fold': 1,
            'optimizer': "adam",
            'smiles_padding_length': 128,
            'smiles_embedding_size': 384,
            'number_of_pathways': 619,
            'dropout': 0.5,
            'batch_norm': True,
            'activation_fn': 'relu',
            'drug_sensitivity_processing_parameters': {'parameters': {"min": -8.658382, "max": 13.107465}},
            'loss_fn': 'mse',
            'early_stopping_patience': 15,
            'lr_scheduler_patience': 5
        },
        training_name='maxlen_128_drugblinded'
    )