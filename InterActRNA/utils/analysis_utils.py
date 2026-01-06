# drug_rna_mili_pyg/utils/analysis_utils.py
import pandas as pd

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from data_processing.utils import NUCLEOTIDE_VOCAB_SIZE # <<<--- 添加这一行

import re
import  random
# (其他导入保持不变)
from data_processing.datasets import MoleculeWithFragmentsDataset
from models.main_model import AffinityModelMiliPyG
from configs.model_config import (
    ITERATIVE_DRUG_ENCODER_CONFIG,  # 替代原来的药物编码器配置
    MULTI_VIEW_RNA_ENCODER_CONFIG,
    CROSS_ATTENTION_CONFIG,  # 保留跨模态注意力配置
    MAIN_PREDICTOR_CONFIG,
    RNA_SUBGRAPH_CONFIG,
    AUX_LOSS_TYPE,
    TRAIN_CONFIG, COLUMN_NAMES, PROCESSED_DATA_DIR, RAW_DATA_FILE_PATH
)
from data_processing.utils import NUCLEOTIDE_VOCAB_SIZE
from .logger_utils import log_print

TQDM_FILE_STREAM_ANALYSIS = sys.stdout
LOG_LEVEL_DEBUG_ANALYSIS = False


def set_analysis_module_tqdm_params(tqdm_stream, log_level_debug):
    global TQDM_FILE_STREAM_ANALYSIS, LOG_LEVEL_DEBUG_ANALYSIS
    TQDM_FILE_STREAM_ANALYSIS = tqdm_stream
    LOG_LEVEL_DEBUG_ANALYSIS = log_level_debug


@torch.no_grad()
def get_raw_predictions(model, dataloader, device, desc_prefix="Analyzing",
                        prepare_batch_fn=None):
    if prepare_batch_fn is None:
        raise ValueError("prepare_batch_fn must be provided to get_raw_predictions.")
    model.eval()
    all_predictions_list, all_true_labels_list = [], []
    if not dataloader or len(dataloader) == 0:
        log_print(f"WARN: Dataloader for '{desc_prefix}' is empty or None.", "warn")
        return np.array([]), np.array([])

    eval_iterator = tqdm(dataloader, desc=f"{desc_prefix} Predicting", leave=False,
                         disable=LOG_LEVEL_DEBUG_ANALYSIS or (TQDM_FILE_STREAM_ANALYSIS is None),
                         file=TQDM_FILE_STREAM_ANALYSIS, ascii=True, dynamic_ncols=True)
    for data_batch_from_loader in eval_iterator:
        try:
            original_mol_batch, \
            all_fragments_smiles_flat, \
            original_mol_idx_for_fragments_flat, \
            fragment_smiles_per_original_mol, \
            rna_batch, labels = prepare_batch_fn(
                data_batch_from_loader, device
            )
        except Exception as e:
            log_print(f"ERROR Prep (get_raw_predictions) {desc_prefix}: {e}", level="error")
            import traceback
            log_print(traceback.format_exc(), "error")
            continue

        if not hasattr(original_mol_batch, 'num_graphs') or original_mol_batch.num_graphs == 0:
            if LOG_LEVEL_DEBUG_ANALYSIS: log_print(
                f"WARN (get_raw_predictions) {desc_prefix}: Skip empty orig_mol_batch.", level="debug")
            continue
        try:
            # <<< 核心修复：修改这里，接收 4 个返回值，第 4 个是 is_large_mol_batch_mask，在此处不使用 >>>
             affinity_pred_batch, _, _, _ = model(
                original_mol_batch,
                rna_batch,
                rna_dot_bracket_strings=rna_db_list
            )
            # <<< 修改结束 >>>
        except Exception as e_model_eval:
            log_print(f"ERROR ModelForward (get_raw_predictions) {desc_prefix}: {e_model_eval}", level="error")
            import traceback
            log_print(traceback.format_exc(), "error")
            continue

        all_predictions_list.append(affinity_pred_batch.cpu())
        all_true_labels_list.append(labels.cpu())
    eval_iterator.close()

    if not all_predictions_list:
        log_print(f"WARN (get_raw_predictions) {desc_prefix}: No predictions collected.", "warn")
        return np.array([]), np.array([])
    try:
        s_preds = [p.squeeze(-1) if p.ndim > 1 and p.size(-1) == 1 else p for p in all_predictions_list]
        s_labels = [l.squeeze(-1) if l.ndim > 1 and l.size(-1) == 1 else l for l in all_true_labels_list]
        preds_np = torch.cat(s_preds).numpy()
        labels_np = torch.cat(s_labels).numpy()
        if preds_np.ndim > 1: preds_np = preds_np.squeeze()
        if labels_np.ndim > 1: labels_np = labels_np.squeeze()
        if preds_np.ndim == 0: preds_np = np.expand_dims(preds_np, axis=0)
        if labels_np.ndim == 0: labels_np = np.expand_dims(labels_np, axis=0)
        return labels_np, preds_np
    except Exception as e:
        log_print(f"Error processing preds/labels in get_raw_predictions for '{desc_prefix}': {e}", "error")
        import traceback
        log_print(traceback.format_exc(), "error")
        return np.array([]), np.array([])

def plot_scatter_predictions(true_labels, predictions, pcc, scc, plot_title_prefix, save_dir, expected_pcc=None,
                             expected_scc=None):
    if len(true_labels) == 0 or len(predictions) == 0:
        log_print(f"WARN: No data to plot for {plot_title_prefix}.", "warn");
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(true_labels, predictions, alpha=0.5, label=f"Predictions (N={len(true_labels)})")

    min_val = min(np.min(true_labels), np.min(predictions)) if len(true_labels) > 0 and len(predictions) > 0 else 0
    max_val = max(np.max(true_labels), np.max(predictions)) if len(true_labels) > 0 and len(predictions) > 0 else 1
    padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-6 else 0.1
    line_min = min_val - padding;
    line_max = max_val + padding

    plt.plot([line_min, line_max], [line_min, line_max], 'r--', label="y=x (Ideal)")
    title = f"{plot_title_prefix}: True pKd vs. Predicted pKd\nCalculated: PCC={pcc:.4f}, SCC={scc:.4f}"
    if expected_pcc is not None and not np.isnan(expected_pcc): title += f"\nExpected PCC: {expected_pcc:.4f}"
    if expected_scc is not None and not np.isnan(expected_scc): title += f", Expected SCC: {expected_scc:.4f}"

    plt.title(title);
    plt.xlabel("True pKd");
    plt.ylabel("Predicted pKd")
    plt.xlim(line_min, line_max);
    plt.ylim(line_min, line_max)
    plt.legend();
    plt.grid(True);
    plt.tight_layout()

    sanitized_prefix = re.sub(r'[^\w\s-]', '', plot_title_prefix).strip().replace(' ', '_').replace('/', '_')
    plot_filename = f"{sanitized_prefix}_scatter.png"
    plot_path = os.path.join(save_dir, plot_filename);
    plt.savefig(plot_path);
    plt.close()
    log_print(f"Saved scatter plot for '{plot_title_prefix}' to {plot_path}", level="info")


def run_specific_model_analysis(model_state_dict_or_path, val_loader, device,
                                lambda_val_for_model_init, plot_title_prefix, analysis_seed,
                                prepare_batch_fn,
                                analysis_config_dict=None,
                                expected_pcc=None, expected_scc=None):
    if analysis_config_dict is None: analysis_config_dict = {}
    plot_save_dir = analysis_config_dict.get("plot_save_dir", "analysis_plots_scatter_default")
    os.makedirs(plot_save_dir, exist_ok=True)

    from configs.model_config import (
        ITERATIVE_DRUG_ENCODER_CONFIG, MULTI_VIEW_RNA_ENCODER_CONFIG,
        RNA_MOTIF_ENCODER_CONFIG, MAIN_PREDICTOR_CONFIG, DRUG_RNA_SSIM_CONFIG,
        SSIM_LOSS_CONFIG, CROSS_ATTENTION_CONFIG, MODEL_VARIANT_CONFIG,
        AUX_LOSS_TYPE
    )
    from data_processing.utils import NUCLEOTIDE_VOCAB_SIZE

    iterative_drug_cfg = ITERATIVE_DRUG_ENCODER_CONFIG.copy()
    rna_multiview_cfg = MULTI_VIEW_RNA_ENCODER_CONFIG.copy()
    rna_multiview_cfg['input_nucleotide_vocab_size'] = NUCLEOTIDE_VOCAB_SIZE
    rna_motif_cfg = RNA_MOTIF_ENCODER_CONFIG.copy()
    ssim_cfg = DRUG_RNA_SSIM_CONFIG.copy() if MODEL_VARIANT_CONFIG.get('enable_ssim_module', True) else None
    cross_attn_cfg = CROSS_ATTENTION_CONFIG.copy() if MODEL_VARIANT_CONFIG.get('enable_cross_attention', False) else None

    model = AffinityModelMiliPyG(
        iterative_drug_config=iterative_drug_cfg,
        multi_view_rna_encoder_config=rna_multiview_cfg,
        rna_motif_encoder_config=rna_motif_cfg,
        main_predictor_config=MAIN_PREDICTOR_CONFIG,
        ssim_config=ssim_cfg,
        ssim_loss_config=SSIM_LOSS_CONFIG,
        cross_attention_config=cross_attn_cfg,
        use_ssim=MODEL_VARIANT_CONFIG.get('enable_ssim_module', True),
        aux_loss_type=AUX_LOSS_TYPE,
        lambda_aux=lambda_val_for_model_init,
        debug_mode=LOG_LEVEL_DEBUG_ANALYSIS
    ).to(device)
    try:
        state_dict = torch.load(model_state_dict_or_path, map_location=device) if isinstance(model_state_dict_or_path,
                                                                                             str) else model_state_dict_or_path
        model.load_state_dict(state_dict)
        log_print(f"Successfully loaded model state for '{plot_title_prefix}'.", "info")
    except Exception as e:
        log_print(f"ERROR loading model state for '{plot_title_prefix}': {e}", "error");
        import traceback;
        log_print(traceback.format_exc(), "error");
        return

    true_labels, predictions = get_raw_predictions(model, val_loader, device, plot_title_prefix, prepare_batch_fn)
    if true_labels.size == 0:
        log_print(f"No predictions obtained for '{plot_title_prefix}'. Skipping metrics and plot.", "warn");
        return

    pcc, scc = np.nan, np.nan
    if len(true_labels) > 1 and np.std(true_labels) > 1e-9 and np.std(predictions) > 1e-9:
        try:
            pcc, _ = pearsonr(true_labels, predictions)
        except:
            pass
        try:
            scc, _ = spearmanr(true_labels, predictions)
        except:
            pass

    title_with_context = f"{plot_title_prefix} (S{analysis_seed} L{lambda_val_for_model_init:.2f})"
    log_print(f"Analysis '{title_with_context}': PCC={pcc:.4f}, SCC={scc:.4f}", "info")
    plot_scatter_predictions(true_labels, predictions, pcc, scc, title_with_context, plot_save_dir, expected_pcc,
                             expected_scc)


def run_independent_analysis_mode(analysis_cfg, device, prepare_batch_fn):
    log_print("--- RUNNING IN INDEPENDENT ANALYSIS MODE ---", "info")
    kf_seed = analysis_cfg["seed_for_kfold_in_analysis"]
    lambda_val = analysis_cfg["lambda_aux_for_models_in_analysis"]
    folds_data_to_analyze = analysis_cfg["folds_to_analyze_independently"]

    if not folds_data_to_analyze:
        log_print("WARN: No folds specified in 'folds_to_analyze_independently'. Exiting independent analysis.", "warn")
        return

    abs_processed_data_dir = os.path.abspath(PROCESSED_DATA_DIR)
    abs_raw_data_file_path = os.path.abspath(RAW_DATA_FILE_PATH)
    dataset_fragment_method = TRAIN_CONFIG.get('fragment_method', 'brics')
    try:
        full_dataset = MoleculeWithFragmentsDataset(root_dir=abs_processed_data_dir,
                                                    raw_file_path=abs_raw_data_file_path,
                                                    drug_smiles_col=COLUMN_NAMES['drug_smiles'],
                                                    rna_seq_col=COLUMN_NAMES['rna_sequence'],
                                                    rna_struct_col=COLUMN_NAMES.get('rna_structure', None),
                                                    label_col=COLUMN_NAMES['label'],
                                                    fragment_method=dataset_fragment_method)
    except Exception as e:
        log_print(f"ERROR Independent Analysis Dataset Init: {e}", "error");
        import traceback;
        log_print(traceback.format_exc(), "error");
        return
    if len(full_dataset) == 0:
        log_print("ERROR Independent Analysis: Dataset empty.", "error");
        return
    log_print(f"Independent Analysis: Full dataset size: {len(full_dataset)}", "info")

    n_splits_original = TRAIN_CONFIG.get('n_splits_kfold', 5)
    kf = KFold(n_splits=n_splits_original, shuffle=True, random_state=kf_seed)
    all_fold_indices_list = list(kf.split(full_dataset))

    for fold_idx_key, model_details in folds_data_to_analyze.items():
        try:
            fold_idx_to_process = int(fold_idx_key)
            if not (0 <= fold_idx_to_process < n_splits_original):
                log_print(f"WARN: Invalid fold_idx {fold_idx_to_process} in ANALYSIS_CONFIG. Skipping.", "warn");
                continue
        except ValueError:
            log_print(f"WARN: Invalid key {fold_idx_key} in folds_to_analyze_independently. Must be integer. Skipping.",
                      "warn");
            continue

        _, val_indices = all_fold_indices_list[fold_idx_to_process]
        model_path = model_details["model_path"]
        expected_pcc_val = model_details.get("expected_pcc", float('nan'))
        expected_scc_val = model_details.get("expected_scc", float('nan'))

        if not os.path.exists(model_path):
            log_print(f"ERROR: Model file not found for Fold {fold_idx_to_process + 1}: {model_path}", "error");
            continue

        val_subset = Subset(full_dataset, val_indices)
        val_loader = PyGDataLoader(val_subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=0)
        if len(val_loader) == 0:
            log_print(f"WARN: Val loader for Fold {fold_idx_to_process + 1} is empty. Skipping.", "warn");
            continue

        plot_title = f"Ind. Analysis Fold {fold_idx_to_process + 1}"
        run_specific_model_analysis(model_path, val_loader, device, lambda_val,
                                    plot_title, kf_seed, prepare_batch_fn,
                                    analysis_cfg,
                                    expected_pcc_val, expected_scc_val)
    log_print("--- Independent Analysis Mode Finished ---", "info")


def analyze_fragment_distribution(raw_file_path, smiles_col_name, method='brics', sample_size=50):
    log_print(f"--- Analyzing Fragment Distribution (Sample: {sample_size}, Method: {method}) ---", "info")
    try:
        df = None
        if raw_file_path.endswith('.csv') or raw_file_path.endswith('.tsv'):
            df = pd.read_csv(raw_file_path, sep='\t')
        else:
            log_print(f"Unsupported file type for fragment analysis: {raw_file_path}", "error")
            return

        if df is None or smiles_col_name not in df.columns:
            log_print(f"Could not load DataFrame or SMILES column '{smiles_col_name}' not found.", "error")
            return

        all_smiles_series = df[smiles_col_name].dropna().astype(str)
        if all_smiles_series.empty:
            log_print("No SMILES strings found for fragment analysis.", "warn")
            return
        
        unique_smiles_list = all_smiles_series.unique().tolist()
        actual_sample_size = min(sample_size, len(unique_smiles_list))
        
        if actual_sample_size == 0:
            log_print("No SMILES to sample for fragment analysis after unique filtering.")
            return
            
        sampled_smiles = random.sample(unique_smiles_list, actual_sample_size)

        targeted_smiles_to_check = [
            "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(CO)O)O)O", # Rank 1 from previous error analysis
            "C1[C@H](CN(C[C@H]1N)c1nc(nc(N2C[C@H](C[C@@H](C2)N)N)n1)Nc1ccc(cc1)NC(=O)c1c(cc2c(c1)cccc2)O)N", # Rank 2
            "C1=NC2=NC=NC(=C2N1)N" # Rank 7 (Adenine)
        ]

        smiles_to_process_set = set(sampled_smiles)
        for tsmi in targeted_smiles_to_check:
            if tsmi in unique_smiles_list:
                 smiles_to_process_set.add(tsmi)

        smiles_to_process = list(smiles_to_process_set)
        if not smiles_to_process:
            log_print("No SMILES to process for fragment analysis after combining sampled and targeted.", "warn")
            return

        fragment_counts = []
        disable_tqdm_frag = LOG_LEVEL_DEBUG_TRAINING or (TQDM_FILE_STREAM_TRAINING is None)
        
        log_print(f"Processing {len(smiles_to_process)} unique SMILES for fragment analysis...", "info")

        for smi in tqdm(smiles_to_process, desc=f"Analyzing frags ({method})", leave=False, 
                        file=TQDM_FILE_STREAM_TRAINING, ascii=True, disable=disable_tqdm_frag):
            if not smi or not smi.strip():
                fragment_counts.append(0)
                continue
            
            frags = None
            num_frags_for_current_smi = -1

            try:
                frags = get_molecule_fragments(smi, method=method)
                if isinstance(frags, (set, list)):
                    valid_frags = {f_item for f_item in frags if isinstance(f_item, str) and f_item.strip()}
                    num_frags_for_current_smi = len(valid_frags)
                    fragment_counts.append(num_frags_for_current_smi)
                else:
                    log_print(f"Warning: get_molecule_fragments for '{smi}' returned unexpected type: {type(frags)}. Treating as error.", "warn")
                    fragment_counts.append(-1)
                    
            except Exception as e_frag:
                log_print(f"Error fragmenting SMILES '{smi}': {e_frag}", level="warn")
                fragment_counts.append(-1)
                num_frags_for_current_smi = -1
            
            if smi in targeted_smiles_to_check:
                log_print(f"  TARGETED SMILES PROCESSED: {smi}", "info")
                log_print(f"    Number of fragments: {num_frags_for_current_smi}", "info")
                if frags and isinstance(frags, (set, list)) and 0 < num_frags_for_current_smi <= 10 : 
                     log_print(f"    Fragments: {list(valid_frags)}", "info")
                elif num_frags_for_current_smi == 0:
                     log_print(f"    Fragments: [] (No valid fragments or original SMILES used as fragment)", "info")
                elif num_frags_for_current_smi > 10:
                     log_print(f"    Fragments: (Too many to display: {num_frags_for_current_smi})", "info")

        if fragment_counts:
            count_distribution = Counter(fragment_counts)
            log_print(f"Fragment count distribution ('{method}', {len(smiles_to_process)} unique SMILES processed):", "info")
            for count_val, num_molecules in sorted(count_distribution.items()):
                percentage = (num_molecules / len(smiles_to_process)) * 100 if len(smiles_to_process) > 0 else 0
                label = f"{count_val} frags" if count_val != -1 else "ERROR in fragmentation"
                log_print(f"  {label}: {num_molecules} mols ({percentage:.2f}%)", "info")
        else:
            log_print("No fragment counts collected.", "warn")
            
        log_print(f"--- End Fragment Analysis ({method}) ---", "info")

    except FileNotFoundError:
        log_print(f"ERROR: Raw data file not found for fragment analysis: '{raw_file_path}'", "error")
    except KeyError as e_key:
        log_print(f"ERROR: SMILES column '{smiles_col_name}' not found in the data file. {e_key}", "error")
    except Exception as e:
        log_print(f"An unexpected error occurred during fragment analysis: {e}", "error")
        import traceback
        log_print(traceback.format_exc(), "error")