# drug_rna_mili_pyg/utils/training_utils.py

import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
import datetime

from data_processing.datasets import MoleculeWithFragmentsDataset
from models.main_model import AffinityModelMiliPyG
from configs.model_config import (
    ITERATIVE_DRUG_ENCODER_CONFIG,
    MULTI_VIEW_RNA_ENCODER_CONFIG,
    CROSS_ATTENTION_CONFIG,
    RNA_MOTIF_ENCODER_CONFIG,
    MAIN_PREDICTOR_CONFIG,
    SSIM_LOSS_CONFIG,
    TRAIN_CONFIG, COLUMN_NAMES, PROCESSED_DATA_DIR, RAW_DATA_FILE_PATH,
    DRUG_RNA_SSIM_CONFIG,
    MODEL_VARIANT_CONFIG,
    AUX_LOSS_TYPE,
    SELF_SUPERVISED_LEARNING_CONFIG # 导入SSL配置
)
from data_processing.utils import NUCLEOTIDE_VOCAB_SIZE
from .logger_utils import log_print

TQDM_FILE_STREAM_TRAINING = sys.stdout
LOG_LEVEL_DEBUG_TRAINING = False

def set_training_module_params(tqdm_stream, log_level_debug):
    global TQDM_FILE_STREAM_TRAINING, LOG_LEVEL_DEBUG_TRAINING
    TQDM_FILE_STREAM_TRAINING = tqdm_stream
    LOG_LEVEL_DEBUG_TRAINING = log_level_debug

def set_seed(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log_print(f"Random seed set to {seed_value}", level="info")


def prepare_model_inputs_from_databatch(data_batch_from_loader, device, task_type):
    original_mol_batch_device = data_batch_from_loader.to(device)

    # 1. RNA批处理应该在外面，对所有任务都执行
    batched_rna_multiview = Batch().to(device)
    if hasattr(original_mol_batch_device, 'rna_data_obj'):
        rna_data_list = original_mol_batch_device.rna_data_obj
        if rna_data_list and all(isinstance(d, Data) for d in rna_data_list):
            try:
                batched_rna_multiview = Batch.from_data_list(rna_data_list).to(device)
            except Exception as e:
                log_print(f"ERROR_PREPARE RNA (Batching): {e}", level="error")

    # 2. 根据任务类型选择正确的标签，并且不再覆盖
    if task_type == 'regression':
        labels = original_mol_batch_device.y_reg.to(device)
    elif task_type == 'classification':
        labels = original_mol_batch_device.y_cls.to(device)
    else:
        # 提供一个默认或报错
        labels = original_mol_batch_device.y.to(device)

    # 3. descriptors 和 rna_dot_bracket_list 的处理保持不变
    descriptors = original_mol_batch_device.mol_descriptors.to(device)
    num_graphs = original_mol_batch_device.num_graphs
    if descriptors.numel() > 0 and num_graphs > 0:
        if descriptors.numel() % num_graphs == 0:
            num_descriptors_per_mol = descriptors.numel() // num_graphs
            descriptors = descriptors.view(num_graphs, num_descriptors_per_mol)
        else:
            log_print(
                f"WARN: Descriptors tensor size ({descriptors.numel()}) not divisible by num_graphs ({num_graphs}).",
                "warn")

    rna_dot_bracket_list = original_mol_batch_device.rna_dot_bracket

    return original_mol_batch_device, batched_rna_multiview, labels, descriptors, rna_dot_bracket_list
def train_one_epoch(model, dataloader, optimizer, device, criterion_affinity,
                    current_epoch_num, fold_idx_tqdm, task_type, # c
                    onecycle_scheduler=None):
    model.train()
    epoch_total_loss, epoch_affinity_loss, epoch_aux_loss, epoch_desc_loss, epoch_ssl_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_aux_metrics = []
    num_batches = len(dataloader)
    if num_batches == 0: return 0, 0, 0, 0, 0, float('nan')

    train_iterator = tqdm(dataloader, desc=f"Fold {fold_idx_tqdm}, Epoch {current_epoch_num} Trn", leave=False, disable=LOG_LEVEL_DEBUG_TRAINING, file=TQDM_FILE_STREAM_TRAINING, ascii=True, dynamic_ncols=True)

    for data_batch_from_loader in train_iterator:
        try:
            original_mol_batch, rna_batch, labels, descriptors, rna_db_list = prepare_model_inputs_from_databatch(data_batch_from_loader, device,task_type)

            optimizer.zero_grad(set_to_none=True)

            # --- 修改: 模型调用现在包含SSL配置 ---
            affinity_pred, descriptor_pred, similarity_matrix, drug_recon, rna_recon, orig_drug_emb, orig_rna_emb = model(
                original_mol_batch, rna_batch, rna_db_list, ssl_config=SELF_SUPERVISED_LEARNING_CONFIG
            )

            # --- 修改: 损失计算现在包含SSL相关的所有输入 ---
            total_loss, loss_affinity, aux_loss, loss_desc, loss_ssl, aux_metric = model.compute_loss(
                affinity_prediction=affinity_pred,
                descriptor_prediction=descriptor_pred,
                drug_recon=drug_recon,
                rna_recon=rna_recon,
                true_labels=labels,
                true_descriptors=descriptors,
                original_drug_embedding=orig_drug_emb,
                original_rna_embedding=orig_rna_emb,
                similarity_matrix=similarity_matrix,
                criterion_affinity=criterion_affinity,
                ssl_config=SELF_SUPERVISED_LEARNING_CONFIG
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                log_print(f"WARN NaN/Inf loss Trn E{current_epoch_num}. Skipping batch.", level="warn")
                continue

            total_loss.backward()
            optimizer.step()

            if onecycle_scheduler:
                onecycle_scheduler.step()

            epoch_total_loss += total_loss.item()
            epoch_affinity_loss += loss_affinity.item()
            epoch_aux_loss += aux_loss.item()
            epoch_desc_loss += loss_desc.item()
            epoch_ssl_loss += loss_ssl.item() # 新增
            if torch.is_tensor(aux_metric) and not torch.isnan(aux_metric):
                epoch_aux_metrics.append(aux_metric.item())

        except Exception as e:
            log_print(f"ERROR Trn E{current_epoch_num}: {e}", level="error")
            import traceback; log_print(traceback.format_exc(), "error")
            continue

    train_iterator.close()

    avg_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
    avg_affinity_loss = epoch_affinity_loss / num_batches if num_batches > 0 else 0
    avg_aux_loss = epoch_aux_loss / num_batches if num_batches > 0 else 0
    avg_desc_loss = epoch_desc_loss / num_batches if num_batches > 0 else 0
    avg_ssl_loss = epoch_ssl_loss / num_batches if num_batches > 0 else 0 # 新增
    avg_aux_metric = np.nanmean(epoch_aux_metrics) if epoch_aux_metrics else float('nan')

    return avg_total_loss, avg_affinity_loss, avg_aux_loss, avg_desc_loss, avg_ssl_loss, avg_aux_metric


@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion_affinity, current_epoch_num, fold_idx_tqdm, task_type):
    model.eval()
    total_eval_loss_epoch = 0.0
    valid_loss_count = 0
    all_predictions_list, all_true_labels_list = [], []
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0, {}, np.array([]), np.array([])

    eval_iterator = tqdm(dataloader, desc=f"Fold {fold_idx_tqdm}, Epoch {current_epoch_num} Eval", leave=False, disable=LOG_LEVEL_DEBUG_TRAINING, file=TQDM_FILE_STREAM_TRAINING, ascii=True, dynamic_ncols=True)

    for data_batch_from_loader in eval_iterator:
        try:
            original_mol_batch, rna_batch, labels_batch, descriptors_batch, rna_db_list_eval = prepare_model_inputs_from_databatch(data_batch_from_loader, device,task_type=task_type)

            affinity_pred_batch_eval, _, _, _, _, _, _ = model(
                original_mol_batch, rna_batch, rna_db_list_eval, ssl_config={'enable': False}
            )

            # 在评估时，我们只关心主任务的损失，所以简化compute_loss的调用
            loss_affinity_eval = criterion_affinity(affinity_pred_batch_eval.squeeze(), labels_batch.squeeze())

            if not torch.isnan(loss_affinity_eval):
                total_eval_loss_epoch += loss_affinity_eval.item()
                valid_loss_count += 1

            all_predictions_list.append(affinity_pred_batch_eval.cpu())
            all_true_labels_list.append(labels_batch.cpu())

        except Exception as e:
            log_print(f"ERROR Eval E{current_epoch_num}: {e}", level="error")
            import traceback; log_print(traceback.format_exc(), "error")
            continue

    eval_iterator.close()
    avg_loss = total_eval_loss_epoch / valid_loss_count if valid_loss_count > 0 else float('nan')

    if not all_predictions_list: return avg_loss, {}, np.array([]), np.array([])

    try:
        predictions_np_all = torch.cat([p.squeeze(-1) if p.dim() > 1 else p for p in all_predictions_list]).numpy()
        true_labels_np_all = torch.cat([l.squeeze(-1) if l.dim() > 1 else l for l in all_true_labels_list]).numpy()
    except Exception as e:
        log_print(f"Error concatenating predictions/labels in eval: {e}", "error")
        return avg_loss, {}, np.array([]), np.array([])

    all_metrics = {}

    if task_type == 'regression':
        from sklearn.metrics import mean_squared_error
        from scipy.stats import pearsonr, spearmanr
        mse, rmse, pcc, scc = np.nan, np.nan, np.nan, np.nan
        if len(true_labels_np_all) > 1:
            try:
                mse = mean_squared_error(true_labels_np_all, predictions_np_all)
                rmse = np.sqrt(mse)
                if np.std(true_labels_np_all) > 1e-9 and np.std(predictions_np_all) > 1e-9:
                    pcc, _ = pearsonr(true_labels_np_all, predictions_np_all)
                    scc, _ = spearmanr(true_labels_np_all, predictions_np_all)
            except Exception as e:
                log_print(f"Error calculating regression metrics: {e}", "error")
        all_metrics = {
            'val_loss': avg_loss, 'val_mse': mse, 'val_rmse': rmse, 'val_pcc': pcc, 'val_scc': scc
        }

    else:  # classification
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
        
        auc, bacc, specificity, recall, precision = np.nan, np.nan, np.nan, np.nan, np.nan
        
        predictions_prob = torch.sigmoid(torch.from_numpy(predictions_np_all)).numpy()
        predictions_binary = (predictions_prob > 0.5).astype(int)

        if len(true_labels_np_all) > 1 and len(np.unique(true_labels_np_all)) > 1:
            try:
                auc = roc_auc_score(true_labels_np_all, predictions_prob)
                precision = precision_score(true_labels_np_all, predictions_binary, zero_division=0)
                recall = recall_score(true_labels_np_all, predictions_binary, zero_division=0)
                
                tn, fp, fn, tp = confusion_matrix(true_labels_np_all, predictions_binary).ravel()
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bacc = (recall + specificity) / 2

            except Exception as e:
                log_print(f"Error calculating classification metrics: {e}", "error")
        
        all_metrics = {
            'val_loss': avg_loss, 
            'val_auc': auc, 
            'val_bacc': bacc,
            'val_specificity': specificity,
            'val_recall': recall,
            'val_precision': precision
        }

    return avg_loss, all_metrics, true_labels_np_all, predictions_np_all


def run_training_kfold(seed_for_run, device, current_lambda_aux,
                       log_level_debug_main,
                       tqdm_file_stream_main,
                       specific_fold_to_run=None):
    set_training_module_params(tqdm_file_stream_main, log_level_debug_main)
    set_seed(seed_for_run)

    # --- 核心修改 1: 将任务类型定义为顶层配置 ---
    # 在这里选择 'regression' 或 'classification'
    TASK_TYPE = 'regression'

    log_print(f"--- Running K-Fold Training for Task: {TASK_TYPE.upper()} ---", "info")

    full_dataset = MoleculeWithFragmentsDataset(
        root_dir=os.path.abspath(PROCESSED_DATA_DIR),
        raw_file_path=os.path.abspath(RAW_DATA_FILE_PATH),
        drug_smiles_col=COLUMN_NAMES['drug_smiles'],
        rna_seq_col=COLUMN_NAMES['rna_sequence'],
        rna_struct_col=COLUMN_NAMES.get('rna_structure', None),
        label_col=COLUMN_NAMES['label']
    )

    kf = KFold(n_splits=TRAIN_CONFIG['n_splits_kfold'], shuffle=True, random_state=seed_for_run)
    all_kf_splits_indices = list(kf.split(full_dataset))
    folds_to_iterate = list(enumerate(all_kf_splits_indices))
    if specific_fold_to_run is not None and 0 <= specific_fold_to_run < len(all_kf_splits_indices):
        folds_to_iterate = [(specific_fold_to_run, all_kf_splits_indices[specific_fold_to_run])]

    # --- 核心修改 2: 计算类别权重以处理不平衡问题 ---
    pos_weight_tensor = None
    if TASK_TYPE == 'classification':
        try:
            # 从整个数据集中计算正负样本比例
            all_labels = full_dataset.get_raw_dataframe()[COLUMN_NAMES['label']]
            pKd_threshold = 4.0  # 与datasets.py中的阈值保持一致
            num_pos = (all_labels >= pKd_threshold).sum()
            num_neg = (all_labels < pKd_threshold).sum()
            if num_pos > 0:
                pos_weight = num_neg / num_pos
                pos_weight_tensor = torch.tensor([pos_weight], device=device)
                log_print(
                    f"Classification task: num_pos={num_pos}, num_neg={num_neg}. Calculated pos_weight: {pos_weight:.2f}",
                    "info")
            else:
                log_print("WARN: No positive samples found in the dataset for classification.", "warn")
        except Exception as e:
            log_print(f"ERROR calculating pos_weight: {e}. Proceeding without weights.", "error")

    overall_best_val_metric_so_far = -float('inf')
    overall_best_model_state_so_far = None
    overall_best_model_fold_idx = -1
    overall_best_model_epoch = -1
    all_fold_best_metrics = []

    time_sfx_run = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    models_save_dir = os.path.join(os.getcwd(), f"kfold_models_s{seed_for_run}_{TASK_TYPE}_{time_sfx_run}")
    os.makedirs(models_save_dir, exist_ok=True)
    log_print(f"Models for each fold will be saved in: {os.path.abspath(models_save_dir)}", "info")

    for fold_idx, (train_indices, val_indices) in folds_to_iterate:
        log_print(f"\n--- Starting Training Fold {fold_idx + 1}/{len(folds_to_iterate)} ---", level="info")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        train_loader = PyGDataLoader(train_subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=0)
        val_loader = PyGDataLoader(val_subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=0)

        log_print(f"Current task for Fold {fold_idx + 1}: {TASK_TYPE}", "info")

        model_config = {
            'iterative_drug_config': ITERATIVE_DRUG_ENCODER_CONFIG,
            'multi_view_rna_encoder_config': {**MULTI_VIEW_RNA_ENCODER_CONFIG,
                                              'input_nucleotide_vocab_size': NUCLEOTIDE_VOCAB_SIZE},
            'rna_motif_encoder_config': RNA_MOTIF_ENCODER_CONFIG,
            'main_predictor_config': MAIN_PREDICTOR_CONFIG,
            'ssim_config': DRUG_RNA_SSIM_CONFIG,
            'ssim_loss_config': SSIM_LOSS_CONFIG,
            'cross_attention_config': CROSS_ATTENTION_CONFIG,
            'use_ssim': MODEL_VARIANT_CONFIG['enable_ssim_module'],
            'debug_mode': log_level_debug_main,
            'aux_loss_type': AUX_LOSS_TYPE,
            'task': TASK_TYPE
        }
        model = AffinityModelMiliPyG(**model_config).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                weight_decay=TRAIN_CONFIG['weight_decay'])

        if TASK_TYPE == 'regression':
            criterion_affinity = nn.MSELoss()
            early_stop_metric = 'val_pcc'
            best_val_metric_this_fold = -float('inf')
            metric_to_beat_overall = 'val_pcc'
        else:  # classification
            # --- 核心修改 2 (续): 将权重应用到损失函数 ---
            criterion_affinity = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            early_stop_metric = 'val_auc'  # 使用AUC作为早停和最佳模型选择指标
            best_val_metric_this_fold = -float('inf')
            metric_to_beat_overall = 'val_auc'

        epochs_no_improve_this_fold = 0
        best_model_state_this_fold = None
        metrics_at_best_val_metric_this_fold = {}
        best_epoch_this_fold = -1

        for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
            train_loss, train_aff_loss, _, train_desc_loss, train_ssl_loss, _ = train_one_epoch(
                model=model, dataloader=train_loader, optimizer=optimizer, device=device,
                criterion_affinity=criterion_affinity, current_epoch_num=epoch,
                fold_idx_tqdm=fold_idx + 1, task_type=TASK_TYPE
            )

            val_loss, val_metrics, _, _ = evaluate_model(
                model=model, dataloader=val_loader, device=device,
                criterion_affinity=criterion_affinity, current_epoch_num=epoch,
                fold_idx_tqdm=fold_idx + 1, task_type=TASK_TYPE
            )

            current_metric_for_stop = val_metrics.get(early_stop_metric, -float('inf'))
            metric_str = f"{current_metric_for_stop:.4f}" if not np.isnan(current_metric_for_stop) else "N/A"
            log_print(
                f"F{fold_idx + 1} E{epoch} | TrL:{train_loss:.3f} (Aff:{train_aff_loss:.3f} Desc:{train_desc_loss:.3f} SSL:{train_ssl_loss:.3f}) | ValL:{val_loss:.3f} {early_stop_metric}:{metric_str}",
                level="info")

            if not np.isnan(current_metric_for_stop) and current_metric_for_stop > best_val_metric_this_fold:
                best_val_metric_this_fold = current_metric_for_stop
                best_epoch_this_fold = epoch
                epochs_no_improve_this_fold = 0
                best_model_state_this_fold = {k: v.cpu() for k, v in model.state_dict().items()}
                metrics_at_best_val_metric_this_fold = val_metrics.copy()

                if current_metric_for_stop > overall_best_val_metric_so_far:
                    overall_best_val_metric_so_far = current_metric_for_stop
                    overall_best_model_state_so_far = best_model_state_this_fold
                    overall_best_model_fold_idx = fold_idx
                    overall_best_model_epoch = epoch
            else:
                epochs_no_improve_this_fold += 1

            if epochs_no_improve_this_fold >= TRAIN_CONFIG['patience_epochs']:
                log_print(f"F{fold_idx + 1}: EARLY STOP at E{epoch}.", level="info")
                break

        metrics_at_best_val_metric_this_fold['epoch'] = best_epoch_this_fold

        model_path_this_fold = None
        if best_model_state_this_fold:
            best_metric_val = metrics_at_best_val_metric_this_fold.get(metric_to_beat_overall, 0.0)
            model_filename = f"fold_{fold_idx + 1}_epoch_{best_epoch_this_fold}_{metric_to_beat_overall}_{best_metric_val:.4f}.pth"
            model_path_this_fold = os.path.join(models_save_dir, model_filename)
            try:
                torch.save(best_model_state_this_fold, model_path_this_fold)
                log_print(f"Saved best model for Fold {fold_idx + 1} to: {model_path_this_fold}", "info")
            except Exception as e:
                log_print(f"ERROR saving model for Fold {fold_idx + 1}: {e}", "error")
                model_path_this_fold = "Failed to save"

        metrics_at_best_val_metric_this_fold['model_path'] = model_path_this_fold
        all_fold_best_metrics.append(metrics_at_best_val_metric_this_fold)

    # --- 核心修改 3: 使K-Fold总结报告适应任务类型 ---
    log_print(f"\n--- K-Fold Cross-Validation Final Summary ({TASK_TYPE.upper()}) ---", "info")
    
    if TASK_TYPE == 'regression':
        log_print("=" * 80, "info")
        log_print(f"{'Fold':<6} | {'Best Epoch':<12} | {'Val PCC':<14} | {'Val RMSE':<14}", "info")
        log_print("-" * 80, "info")
        fold_results_pcc, fold_results_rmse = [], []
        for i, metrics in enumerate(all_fold_best_metrics):
            pcc = metrics.get('val_pcc', np.nan)
            rmse = metrics.get('val_rmse', np.nan)
            fold_results_pcc.append(pcc)
            fold_results_rmse.append(rmse)
            log_print(
                f"{i + 1:<6} | {metrics.get('epoch', 'N/A'):<12} | {pcc:.4f if not np.isnan(pcc) else 'N/A':<14} | {rmse:.4f if not np.isnan(rmse) else 'N/A':<12} | {os.path.basename(metrics.get('model_path', 'N/A')):<40}",
                "info")
        log_print("=" * 120, "info")
        log_print("\n--- Overall Performance (Mean ± Std Dev) ---", "info")
        log_print(f"  PCC:  Mean={np.nanmean(fold_results_pcc):.4f} ± {np.nanstd(fold_results_pcc):.4f}", level="info")
        log_print(f"  RMSE: Mean={np.nanmean(fold_results_rmse):.4f} ± {np.nanstd(fold_results_rmse):.4f}",
                  level="info")

    else: # classification
        # 定义您需要的五个核心指标
        metrics_to_report = ['auc', 'bacc', 'specificity', 'recall', 'precision']
        display_names = ['AUC', 'BACC', 'Specificity', 'Recall', 'Precision']
        
        header = f"{'Fold':<6} | {'Best Epoch':<12} | " + " | ".join([f"{name:<11}" for name in display_names])
        log_print("=" * len(header), "info")
        log_print(header, "info")
        log_print("-" * len(header), "info")

        all_metrics_lists = {key: [] for key in metrics_to_report}
        for i, metrics in enumerate(all_fold_best_metrics):
            row_values = [f"{i + 1:<6}", f"{metrics.get('epoch', 'N/A'):<12}"]
            for key in metrics_to_report:
                value = metrics.get(f'val_{key}', np.nan)
                all_metrics_lists[key].append(value)
                value_str = f"{value:.4f}" if not np.isnan(value) else "N/A"
                row_values.append(f"{value_str:<11}")
            log_print(" | ".join(row_values), "info")

        log_print("=" * len(header), "info")

        log_print("\n--- Overall Performance (Mean ± Std Dev) ---", "info")
        for i, key in enumerate(metrics_to_report):
            results = all_metrics_lists[key]
            display_name = display_names[i]
            if any(not np.isnan(r) for r in results):
                log_print(f"  {display_name:<12}: Mean={np.nanmean(results):.4f} ± {np.nanstd(results):.4f}", level="info")

    # 打印全局最佳模型信息
    if overall_best_model_state_so_far:
        best_metric_name_overall = 'PCC' if TASK_TYPE == 'regression' else 'AUC'
        log_print(
            f"\nOverall best model was from Fold {overall_best_model_fold_idx + 1} at Epoch {overall_best_model_epoch} ({best_metric_name_overall}: {overall_best_val_metric_so_far:.4f}).",
            "info")
        log_print("This model has already been saved as part of its fold's results.", "info")
    else:
        log_print("WARN: No overall best model was identified across all folds.", "warn")

    return overall_best_model_state_so_far, overall_best_model_fold_idx, current_lambda_aux, full_dataset, all_kf_splits_indices, seed_for_run