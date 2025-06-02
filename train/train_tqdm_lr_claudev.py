import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# DeepChem 로거 가져오기
dc_logger = logging.getLogger("deepchem")
dc_logger.setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json
import pickle
from time import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from yj.encoder import ChemBERTaOmicsDataset, chemberta_collate
from models.model import PASO_GEP_CNV_MUT, PASO_GEP_CNV_MUT_2
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score
from utils.utils import get_device, get_log_molar


class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
            
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class AttentionAnalyzer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.attention_dir = os.path.join(model_dir, "attention_analysis")
        os.makedirs(self.attention_dir, exist_ok=True)
        
    def extract_attention_weights(self, model, batch, device):
        """Extract attention weights from model during forward pass"""
        model.eval()
        
        with torch.no_grad():
            drug_data = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            _, pred_dict = model(
                drug_data,
                batch["gep"].to(device),
                batch["cnv"].to(device),
                batch["mut"].to(device)
            )
        
        # Extract attention weights from prediction dictionary
        attention_weights = {
            'gene_attention': pred_dict.get('gene_attention', None),
            'cnv_attention': pred_dict.get('cnv_attention', None),
            'mut_attention': pred_dict.get('mut_attention', None),
            'smiles_attention_gep': pred_dict.get('smiles_attention_gep', None),
            'smiles_attention_cnv': pred_dict.get('smiles_attention_cnv', None),
            'smiles_attention_mut': pred_dict.get('smiles_attention_mut', None)
        }
        
        # Remove None values
        attention_weights = {k: v for k, v in attention_weights.items() if v is not None}
        
        return attention_weights, pred_dict
    
    def visualize_attention_heatmap(self, attention_weights, title, save_path):
        """Visualize attention weights as heatmap"""
        plt.figure(figsize=(12, 8))
        
        # Handle different tensor dimensions
        if attention_weights.dim() > 2:
            # Average over batch dimension if present
            if attention_weights.dim() == 3:
                attention_weights = attention_weights.mean(dim=0)
            elif attention_weights.dim() == 4:
                # Average over batch and head dimensions
                attention_weights = attention_weights.mean(dim=(0, -1))
        
        # Convert to numpy
        attention_matrix = attention_weights.numpy()
        
        # Create heatmap
        sns.heatmap(attention_matrix, cmap='Blues', cbar=True, square=True)
        plt.title(title)
        plt.xlabel('Sequence Position')
        plt.ylabel('Pathway/Gene Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_pathway_attention(self, model, test_loader, device, epoch):
        """Analyze attention patterns across pathways"""
        model.eval()
        pathway_attention_scores = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Analyzing pathway attention")):
                if batch_idx >= 5:  # Limit analysis to first 5 batches for efficiency
                    break
                    
                attention_weights, _ = self.extract_attention_weights(model, batch, device)
                
                # Accumulate attention scores
                for key, weights in attention_weights.items():
                    if weights is None:
                        continue
                    if key not in pathway_attention_scores:
                        pathway_attention_scores[key] = []
                    pathway_attention_scores[key].append(weights.cpu())
        
        # Visualize average attention scores
        for key, scores_list in pathway_attention_scores.items():
            if len(scores_list) > 0:
                avg_scores = torch.stack(scores_list).mean(dim=0)
                save_path = os.path.join(self.attention_dir, f'{key}_epoch{epoch}.png')
                self.visualize_attention_heatmap(avg_scores, f'{key} Attention (Epoch {epoch})', save_path)
        
        return pathway_attention_scores
    
    def plot_attention_distribution(self, attention_scores, epoch):
        """Plot distribution of attention scores"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (key, scores) in enumerate(attention_scores.items()):
            if idx >= 6:
                break
            
            if len(scores) == 0:
                continue
                
            # Flatten and concatenate all scores
            flat_scores = torch.cat([s.flatten() for s in scores]).numpy()
            
            # Plot histogram
            axes[idx].hist(flat_scores, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[idx].set_title(f'{key} Distribution')
            axes[idx].set_xlabel('Attention Score')
            axes[idx].set_ylabel('Density')
            axes[idx].grid(True, alpha=0.3)
            
            # Add statistics
            axes[idx].axvline(flat_scores.mean(), color='red', linestyle='--', 
                            label=f'Mean: {flat_scores.mean():.3f}')
            axes[idx].axvline(np.median(flat_scores), color='green', linestyle='--', 
                            label=f'Median: {np.median(flat_scores):.3f}')
            axes[idx].legend()
        
        # Hide empty subplots
        for idx in range(len(attention_scores), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.attention_dir, f'attention_distribution_epoch{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_summary_report(self, attention_scores, epoch):
        """Create summary statistics of attention scores"""
        summary = {}
        
        for key, scores in attention_scores.items():
            if len(scores) == 0:
                continue
                
            flat_scores = torch.cat([s.flatten() for s in scores])
            summary[key] = {
                'mean': float(flat_scores.mean()),
                'std': float(flat_scores.std()),
                'min': float(flat_scores.min()),
                'max': float(flat_scores.max()),
                'sparsity': float((flat_scores < 0.01).float().mean()),  # 낮은 attention의 비율
                'entropy': float(-torch.sum(flat_scores * torch.log(flat_scores + 1e-8))) / len(flat_scores)
            }
        
        report_path = os.path.join(self.attention_dir, f'attention_summary_epoch{epoch}.json')
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary


def plot_training_curves(model_dir, fold, metrics_history):
    """Plot comprehensive training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Pearson Correlation
    ax2.plot(epochs, metrics_history['val_pearson'], 'g-', label='Val Pearson', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('Validation Pearson Correlation', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # RMSE
    ax3.plot(epochs, metrics_history['val_rmse'], 'm-', label='Val RMSE', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.set_title('Validation RMSE', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # R2 Score
    ax4.plot(epochs, metrics_history['val_r2'], 'c-', label='Val R2', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('R2 Score', fontsize=12)
    ax4.set_title('Validation R2 Score', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.suptitle(f'Training Curves - Fold {fold}', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(model_dir, f'training_curves_fold{fold}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions_vs_actual(model_dir, fold, predictions, labels, epoch):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 10))
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Scatter plot
    plt.scatter(labels, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    pearson = np.corrcoef(predictions, labels)[0, 1]
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    
    plt.xlabel('Actual IC50 (log)', fontsize=12)
    plt.ylabel('Predicted IC50 (log)', fontsize=12)
    plt.title(f'Predictions vs Actual - Fold {fold}, Epoch {epoch}', fontsize=14)
    plt.text(0.05, 0.95, f'Pearson: {pearson:.3f}\nRMSE: {rmse:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(model_dir, f'predictions_vs_actual_fold{fold}_epoch{epoch}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(
    drug_sensitivity_filepath,
    gep_filepath,
    cnv_filepath,
    mut_filepath,
    smiles_filepath,
    gene_filepath,
    model_path,
    params,
    training_name
):
    # Process parameter file:
    torch.backends.cudnn.benchmark = True
    
    # Update parameters with training configuration
    params.update({
        "batch_size": 32,
        "epochs": 100,
        "num_workers": 4,
        "stacked_dense_hidden_sizes": [1024, 512, 256],
        "early_stopping_patience": 15,
        "lr_scheduler_patience": 5,
        "lr_scheduler_factor": 0.5,
        "min_lr": 1e-6
    })
    
    logger.info("Parameters: %s", params)

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    n_folds = params.get("fold", 11)
    logger.info("Starting %d-fold cross-validation", n_folds)
    
    # Dictionary to store all fold results
    all_fold_results = {}

    for fold in range(n_folds):
        logger.info("============== Fold [%d/%d] ==============", fold+1, n_folds)
        
        # Create model directory
        model_dir = os.path.join(model_path, training_name, f'Fold{fold+1}')
        os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
        
        # Save parameters
        with open(os.path.join(model_dir, "params.json"), "w") as fp:
            json.dump(params, fp, indent=4)

        # Load datasets
        drug_sensitivity_train = drug_sensitivity_filepath + f'train_Fold{fold}.csv'
        train_dataset = ChemBERTaOmicsDataset(
            drug_sensitivity_csv=drug_sensitivity_train,
            smiles_csv=smiles_filepath,
            gep_csv=gep_filepath,
            cnv_csv=cnv_filepath,
            mut_csv=mut_filepath,
            tokenizer_name="DeepChem/ChemBERTa-77M-MLM",
            max_len=params["smiles_padding_length"]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=params["num_workers"],
            collate_fn=chemberta_collate
        )

        drug_sensitivity_test = drug_sensitivity_filepath + f'test_Fold{fold}.csv'
        test_dataset = ChemBERTaOmicsDataset(
            drug_sensitivity_csv=drug_sensitivity_test,
            smiles_csv=smiles_filepath,
            gep_csv=gep_filepath,
            cnv_csv=cnv_filepath,
            mut_csv=mut_filepath,
            tokenizer_name="DeepChem/ChemBERTa-77M-MLM",
            max_len=params["smiles_padding_length"]
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=params["num_workers"],
            collate_fn=chemberta_collate
        )
        
        logger.info(
            "FOLD [%d/%d] Training dataset has %d samples, test set has %d.",
            fold+1, n_folds, len(train_dataset), len(test_dataset)
        )
        
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Initialize model
        params.update({"number_of_pathways": len(pathway_list)})
        model = PASO_GEP_CNV_MUT_2(params).to(device)
        
        # Initialize optimizer
        optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "adam")](
            model.parameters(), lr=params.get("lr", 0.001)
        )
        
        # Initialize schedulers
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=params["lr_scheduler_patience"], 
            factor=params["lr_scheduler_factor"],
            min_lr=params["min_lr"],
            verbose=True
        )
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=params["early_stopping_patience"],
            verbose=True,
            path=os.path.join(model_dir, "weights", f"best_model_fold{fold+1}.pt")
        )
        
        # Initialize attention analyzer
        attention_analyzer = AttentionAnalyzer(model_dir)
        
        # Metrics history
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_pearson': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        # Best metrics tracking
        best_metrics = {
            'min_loss': float('inf'),
            'max_pearson': -float('inf'),
            'max_r2': -float('inf'),
            'min_rmse': float('inf')
        }
        
        logger.info("Training started for Fold %d...", fold+1)
        start_time = time()
        
        for epoch in range(params["epochs"]):
            logger.info("== Fold [%d/%d] Epoch [%d/%d] ==", fold+1, n_folds, epoch+1, params['epochs'])
            
            # Training
            train_loss = training(
                model, device, epoch, fold, train_loader, optimizer, 
                params, start_time
            )
            metrics_history['train_loss'].append(train_loss)
            
            # Evaluation
            test_loss, test_pearson, test_rmse, test_r2, predictions, labels = evaluation(
                model, device, test_loader, params, epoch, fold
            )
            
            # Update metrics history
            metrics_history['val_loss'].append(test_loss)
            metrics_history['val_pearson'].append(float(test_pearson))
            metrics_history['val_rmse'].append(float(test_rmse))
            metrics_history['val_r2'].append(float(test_r2))
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
            
            # Early stopping check
            early_stopping(test_loss, model, epoch)
            
            # Update best metrics
            if test_loss < best_metrics['min_loss']:
                best_metrics['min_loss'] = test_loss
                best_metrics['min_loss_epoch'] = epoch
                
            if test_pearson > best_metrics['max_pearson']:
                best_metrics['max_pearson'] = float(test_pearson)
                best_metrics['max_pearson_epoch'] = epoch
                # Plot predictions vs actual for best pearson
                plot_predictions_vs_actual(model_dir, fold+1, predictions, labels, epoch+1)
                
            if test_r2 > best_metrics['max_r2']:
                best_metrics['max_r2'] = float(test_r2)
                best_metrics['max_r2_epoch'] = epoch
                
            if test_rmse < best_metrics['min_rmse']:
                best_metrics['min_rmse'] = float(test_rmse)
                best_metrics['min_rmse_epoch'] = epoch
            
            # Attention analysis (every 10 epochs or last epoch)
            if epoch % 10 == 0 or epoch == params["epochs"] - 1:
                logger.info("Performing attention analysis...")
                attention_scores = attention_analyzer.analyze_pathway_attention(
                    model, test_loader, device, epoch+1
                )
                attention_analyzer.plot_attention_distribution(attention_scores, epoch+1)
                attention_summary = attention_analyzer.create_attention_summary_report(
                    attention_scores, epoch+1
                )
                logger.info(f"Attention summary saved")
            
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": test_loss,
                "val/pearson": test_pearson,
                "val/rmse": test_rmse,
                "val/r2": test_r2,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
                # 예측·라벨 scatter
            if epoch % 5 == 0:
                wandb.log({
                    "val/pred_vs_true": wandb.plot.scatter(
                        xs=labels.numpy(), ys=predictions.numpy(),
                        title=f"epoch{epoch}", xname="True", yname="Pred"
                    )
                })
        # Plot final training curves
        plot_training_curves(model_dir, fold+1, metrics_history)
        
        # Save fold results
        fold_results = {
            'best_metrics': best_metrics,
            'final_epoch': epoch + 1,
            'metrics_history': metrics_history,
            'early_stopped': early_stopping.early_stop,
            'best_epoch': early_stopping.best_epoch
        }
        
        all_fold_results[f'fold_{fold+1}'] = fold_results
        
        # Save fold results
        with open(os.path.join(model_dir, 'fold_results.json'), 'w') as f:
            json.dump(fold_results, f, indent=4)
        
        logger.info(
            f"Fold {fold+1} completed. Best metrics:\n"
            f"\tMin Loss: {best_metrics['min_loss']:.4f} (epoch {best_metrics.get('min_loss_epoch', 0)+1})\n"
            f"\tMax Pearson: {best_metrics['max_pearson']:.4f} (epoch {best_metrics.get('max_pearson_epoch', 0)+1})\n"
            f"\tMax R2: {best_metrics['max_r2']:.4f} (epoch {best_metrics.get('max_r2_epoch', 0)+1})\n"
            f"\tMin RMSE: {best_metrics['min_rmse']:.4f} (epoch {best_metrics.get('min_rmse_epoch', 0)+1})"
        )
    
    # Create summary plots for all folds
    create_cross_validation_summary(model_path, training_name, all_fold_results)
    
    # Save all fold results
    final_results_path = os.path.join(model_path, training_name, 'all_fold_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(all_fold_results, f, indent=4)
    
    logger.info("All folds completed. Results saved.")


def create_cross_validation_summary(model_path, training_name, all_fold_results):
    """Create summary plots for cross-validation results"""
    summary_dir = os.path.join(model_path, training_name, 'cv_summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Extract metrics from all folds
    metrics = {'pearson': [], 'r2': [], 'rmse': [], 'loss': []}
    
    for fold_name, fold_data in all_fold_results.items():
        best_metrics = fold_data['best_metrics']
        metrics['pearson'].append(best_metrics['max_pearson'])
        metrics['r2'].append(best_metrics['max_r2'])
        metrics['rmse'].append(best_metrics['min_rmse'])
        metrics['loss'].append(best_metrics['min_loss'])
    
    # Create box plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    metric_names = ['Pearson Correlation', 'R² Score', 'RMSE', 'Loss']
    
    for idx, (key, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.boxplot(values, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.scatter([1] * len(values), values, alpha=0.5, s=50)
        ax.set_ylabel(metric_names[idx], fontsize=12)
        ax.set_xticklabels(['All Folds'])
        ax.grid(True, alpha=0.3)
        
        # Add mean and std
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.text(0.5, 0.95, f'Mean: {mean_val:.3f} ± {std_val:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cross-Validation Results Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'cv_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = {
        'metrics': {
            key: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
            for key, values in metrics.items()
        }
    }
    
    with open(os.path.join(summary_dir, 'cv_summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)


def training(model, device, epoch, fold, train_loader, optimizer, params, t):
    """Training function with proper error handling"""
    model.train()
    train_loss = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Training", leave=False)
    


    for ind, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        drug_data = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        
        y_hat, pred_dict = model(
            drug_data,
            batch["gep"].to(device),
            batch["cnv"].to(device),
            batch["mut"].to(device)
        )
        
        loss = model.loss(y_hat, batch["ic50"].to(device))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        train_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{train_loss / (ind + 1):.5f}"})
    
    progress_bar.close()
    
    avg_train_loss = train_loss / len(train_loader)
    logger.info(
        "**** TRAINING **** Fold[%d] Epoch [%d/%d], loss: %.5f. Time: %.1f secs.",
        fold+1, epoch + 1, params['epochs'], avg_train_loss, time() - t
    )
    
    return avg_train_loss


def evaluation(model, device, test_loader, params, epoch, fold):
    """Evaluation function with proper metrics calculation"""
    model.eval()
    test_loss = 0
    log_pres = []
    log_labels = []
    
    # Get min/max values for scaling
    min_value = params["drug_sensitivity_processing_parameters"]["parameters"]["min"]
    max_value = params["drug_sensitivity_processing_parameters"]["parameters"]["max"]
    
    # Progress bar
    progress_bar = tqdm(test_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Testing", leave=True)
    
    with torch.no_grad():
        for ind, batch in enumerate(progress_bar):
            if epoch == 0 and ind == 0:


            drug_data = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            y_hat, pred_dict = model(
                drug_data,
                batch["gep"].to(device),
                batch["cnv"].to(device),
                batch["mut"].to(device)
            )
            logger.info(f"Pred range: {y_hat.min():.3f} ~ {y_hat.max():.3f}")
            logger.info(f"Label range: {batch['ic50'].min():.3f} ~ {batch['ic50'].max():.3f}")
            wandb.log({"debug/pred_min": y_hat.min(), "debug/pred_max": y_hat.max(),
                    "debug/label_min": batch['ic50'].min(), "debug/label_max": batch['ic50'].max()}, step=epoch)
            loss = model.loss(y_hat, batch["ic50"].to(device))
            log_pre = pred_dict.get("log_micromolar_IC50")
            log_pres.append(log_pre)
            log_y = get_log_molar(batch["ic50"], ic50_max=max_value, ic50_min=min_value)
            log_labels.append(log_y.to(device))
            test_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{test_loss / (ind + 1):.5f}"})
    
    progress_bar.close()

    # Concatenate predictions and labels
    predictions = torch.cat(log_pres, dim=0).cpu()
    labels = torch.cat(log_labels, dim=0).cpu()
    
    # Ensure correct shape
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()
    
    # Calculate metrics
    test_pearson_a = pearsonr(predictions, labels)
    test_rmse_a = torch.sqrt(torch.mean((predictions - labels) ** 2))
    test_loss_a = test_loss / len(test_loader)
    test_r2_a = r2_score(predictions, labels)
    
    logger.info(
        "**** TEST **** Fold[%d] Epoch [%d/%d], loss: %.5f, Pearson: %.4f, RMSE: %.4f, R2: %.4f.",
        fold+1, epoch + 1, params['epochs'], test_loss_a, test_pearson_a, test_rmse_a, test_r2_a
    )
    
    return test_loss_a, test_pearson_a, test_rmse_a, test_r2_a, predictions, labels


if __name__ == "__main__":
    import wandb

    # Data paths
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_'
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    gene_filepath = 'data/MUDICUS_Omic_619_pathways.pkl'

    model_path = 'result/model'
    
    # Model parameters
    params = {
        'fold': 11,
        'lr': 0.001,
        'optimizer': "adam",
        'smiles_padding_length': 256,
        'smiles_embedding_size': 384,
        'number_of_pathways': 619,
        'smiles_attention_size': 256,
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
            'parameters': {"min": -8.658382, "max": 13.107465}
        },
        'loss_fn': 'mse'
    }
    
    training_name = f'maxlen_256_with_attention_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        # ────────────────── main() 맨 위 초기화 ──────────────────
    wandb.init(
        project="PASO_GEP_CNV_MUT",
        name="training_name",
        config=params,                # 하이퍼파라미터 자동 저장
        notes="Debug run with scale-check & grad logging"
    )
    wandb.watch_called = False        # 여러 번 watch 되는 것 방지
    # ────────────────────────────────────────────────────────
    # Run the training
    main(
        drug_sensitivity_filepath,
        gep_filepath,
        cnv_filepath,
        mut_filepath,
        smiles_filepath,
        gene_filepath,
        model_path,
        params,
        training_name
    )