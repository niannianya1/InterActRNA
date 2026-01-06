# drug_rna_mili_pyg/main.py

import sys
import os
import datetime
import torch
import numpy as np
import random
import re
import pandas as pd
import torch
# --- Path Setup ---
current_dir_of_script = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir_of_script
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration Imports ---
try:
    from configs.model_config import (
        TRAIN_CONFIG, LAMBDA_AUX as DEFAULT_LAMBDA_AUX,
        ITERATIVE_DRUG_ENCODER_CONFIG,  # 替代原来的药物编码器配置
        MULTI_VIEW_RNA_ENCODER_CONFIG, CROSS_ATTENTION_CONFIG, MAIN_PREDICTOR_CONFIG,DRUG_RNA_SSIM_CONFIG ,
        RNA_SUBGRAPH_CONFIG, AUX_LOSS_TYPE, COLUMN_NAMES  # 移除fragment相关导入
    )
except ImportError as e:
    print(f"CRITICAL: Error importing base configurations: {e}\nSys.path: {sys.path}")
    sys.exit(1)

# --- Utility Function Imports ---
try:
    from utils.logger_utils import Logger, log_print, set_utils_logger, set_utils_log_levels
    from utils.training_utils import (
        set_training_module_params,
        set_seed as set_seed_training,
        prepare_model_inputs_from_databatch as prepare_batch_fn_for_training,  # 这个已经简化了
        run_training_kfold,
        evaluate_model  # 用于错误分析
    )
    from utils.analysis_utils import (
        set_analysis_module_tqdm_params,
        run_independent_analysis_mode,
        run_specific_model_analysis,
    )
    from torch_geometric.data import Batch, Data
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    # 导入迭代模型
    from models.main_model import AffinityModelMiliPyG
    import torch.nn as nn  # 用于错误分析中的criterion
    from data_processing.datasets import MoleculeWithFragmentsDataset  # 访问get_raw_dataframe
except ImportError as e:
    print(f"CRITICAL: Error importing utility modules: {e}\nSys.path: {sys.path}")
    sys.exit(1)

# --- Global Logging & Debug Configuration for main.py ---
LOG_LEVEL_DEBUG = False  # 设为True进行详细模型调试
LOG_LEVEL_INFO = True
LAMBDA_AUX_OVERRIDE = 0.0  # 保持为0.0，专注于主要损失

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
TQDM_FILE_STREAM = None
if hasattr(_ORIGINAL_STDERR, 'isatty') and _ORIGINAL_STDERR.isatty():
    TQDM_FILE_STREAM = _ORIGINAL_STDERR
elif hasattr(_ORIGINAL_STDOUT, 'isatty') and _ORIGINAL_STDOUT.isatty():
    TQDM_FILE_STREAM = _ORIGINAL_STDOUT

LOGGER_INSTANCE = None

# 更新分析配置，适配迭代模型
ANALYSIS_CONFIG = {
    "perform_analysis_only": False,
    "analyze_best_model_after_training": True,  # 控制训练后整体最佳模型分析
    "run_only_specific_fold_for_debug": None,  # 设为None运行所有K折，设为如0只运行第一折进行快速测试
    "seed_for_kfold_in_analysis": 42,
    "lambda_aux_for_models_in_analysis": 0.0,  # 匹配训练时的lambda
    "folds_to_analyze_independently": {},
    "plot_save_dir": "analysis_plots_scatter_iterative",  # 标记这是迭代版本的分析
    "fold_to_analyze_for_errors": 1,  # 0索引，所以1表示第2折。设为None跳过错误分析
    "num_error_samples_to_show": 10  # 显示的最差预测样本数量
}


def main():
    global LOGGER_INSTANCE, TQDM_FILE_STREAM

    initial_seed_for_run = 100  # 可以根据需要修改随机种子
    device_name = TRAIN_CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    is_independent_analysis_only = ANALYSIS_CONFIG.get("perform_analysis_only", False)
    specific_fold_for_debug_run = ANALYSIS_CONFIG.get("run_only_specific_fold_for_debug", None)
    if is_independent_analysis_only:
        specific_fold_for_debug_run = None

    if is_independent_analysis_only:
        current_lambda_for_run = ANALYSIS_CONFIG.get("lambda_aux_for_models_in_analysis", DEFAULT_LAMBDA_AUX)
    else:
        current_lambda_for_run = LAMBDA_AUX_OVERRIDE if LAMBDA_AUX_OVERRIDE is not None else DEFAULT_LAMBDA_AUX

    seed_for_log_filename = initial_seed_for_run
    if is_independent_analysis_only:
        seed_for_log_filename = ANALYSIS_CONFIG.get("seed_for_kfold_in_analysis", initial_seed_for_run)

    # 更新日志前缀，标记这是迭代版本
    mode_prefix_log = "_train"
    if is_independent_analysis_only:
        mode_prefix_log = f"_independent_analysis_iterative_s{seed_for_log_filename}_l{current_lambda_for_run:.2f}"
    else:
        mode_prefix_log = f"_train_iterative_s{seed_for_log_filename}_l{current_lambda_for_run:.2f}"
        if specific_fold_for_debug_run is not None:
            mode_prefix_log += f"_debugfold{specific_fold_for_debug_run + 1}"

    debug_str_for_fname = "_debug" if LOG_LEVEL_DEBUG else ""
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"run{mode_prefix_log}{debug_str_for_fname}_{current_time_str}.log"
    log_file_path_to_use = os.path.join("logs", log_file_name)

    # 初始化日志系统
    if LOGGER_INSTANCE:
        LOGGER_INSTANCE.close()
    LOGGER_INSTANCE = Logger(log_file_path_to_use, terminal_stream=_ORIGINAL_STDOUT)
    sys.stdout = LOGGER_INSTANCE
    sys.stderr = LOGGER_INSTANCE

    set_utils_logger(LOGGER_INSTANCE)
    set_utils_log_levels(LOG_LEVEL_INFO, LOG_LEVEL_DEBUG)
    set_training_module_params(TQDM_FILE_STREAM, LOG_LEVEL_DEBUG)
    set_analysis_module_tqdm_params(TQDM_FILE_STREAM, LOG_LEVEL_DEBUG)

    # 记录运行模式和配置
    log_print(
        f"Script mode: {'Independent Analysis Only' if is_independent_analysis_only else 'Training (Iterative Substructure Learning)'}",
        "info")
    log_print(f"GLOBAL DEBUG LOGGING IS: {'ENABLED' if LOG_LEVEL_DEBUG else 'DISABLED'}", "info")
    if not is_independent_analysis_only and specific_fold_for_debug_run is not None:
        log_print(f"DEBUG RUN: Training will run ONLY for Fold {specific_fold_for_debug_run + 1}.", "warn")
    log_print(f"Device: {device}", "info")
    log_print(f"Logs will be saved to: {os.path.abspath(log_file_path_to_use)}", "info")

    # 记录关键配置信息
    log_print(
        f"Iterative Drug Encoder Config: num_iterations={ITERATIVE_DRUG_ENCODER_CONFIG['num_iterations']}, hidden_dim={ITERATIVE_DRUG_ENCODER_CONFIG['hidden_dim']}",
        "info")
    log_print(
        f"Cross-Attention Config: drug_dim={CROSS_ATTENTION_CONFIG['drug_dim']}, rna_dim={CROSS_ATTENTION_CONFIG['rna_dim']}, num_heads={CROSS_ATTENTION_CONFIG['num_heads']}",
        "info")

    seed_to_set = initial_seed_for_run
    if is_independent_analysis_only:
        seed_to_set = ANALYSIS_CONFIG.get("seed_for_kfold_in_analysis", initial_seed_for_run)
    set_seed_training(seed_to_set)

    if is_independent_analysis_only:
        log_print(
            f"Independent Analysis: KFold seed: {seed_to_set}, Lambda for model init: {current_lambda_for_run:.2f}",
            "info")
        run_independent_analysis_mode(ANALYSIS_CONFIG, device, prepare_batch_fn_for_training)
    else:
        log_print(
            f"Training (Iterative Mode): KFold seed: {initial_seed_for_run}, Lambda_aux for run: {current_lambda_for_run:.2f}",
            "info")

        # 训练配置检查
        if 'num_epochs' not in TRAIN_CONFIG:
            TRAIN_CONFIG['num_epochs'] = 100
        if 'patience_epochs' not in TRAIN_CONFIG:
            TRAIN_CONFIG['patience_epochs'] = 50
        log_print(
            f"Training Config: Max Epochs={TRAIN_CONFIG['num_epochs']}, Early Stopping Patience={TRAIN_CONFIG['patience_epochs']}",
            "info")

        # 学习率调度器配置
        scheduler_type_main = TRAIN_CONFIG.get('lr_scheduler_type', 'none').lower()
        if scheduler_type_main == 'onecycle':
            onecycle_max_lr_val = TRAIN_CONFIG.get('onecycle_max_lr')
            if onecycle_max_lr_val is not None:
                log_print(f"Using OneCycleLR with max_lr: {onecycle_max_lr_val:.1e}", level="info")
            else:
                log_print(f"ERROR: OneCycleLR selected, but 'onecycle_max_lr' not found.", level="error")
        elif 'learning_rate' in TRAIN_CONFIG and TRAIN_CONFIG.get('learning_rate') is not None:
            log_print(f"Base/Fixed LR: {TRAIN_CONFIG['learning_rate']:.1e}", level="info")
        else:
            log_print("WARN: LR key not found/None in TRAIN_CONFIG.", level="warn")

        # 执行K-fold训练
        best_model_state, best_model_fold_idx, lambda_of_best_model, \
        full_dataset_obj, kf_splits_indices, seed_used_for_kfold_run = \
            run_training_kfold(seed_for_run=initial_seed_for_run, device=device,
                               current_lambda_aux=current_lambda_for_run,
                               log_level_debug_main=LOG_LEVEL_DEBUG,
                               tqdm_file_stream_main=TQDM_FILE_STREAM,
                               specific_fold_to_run=specific_fold_for_debug_run)

if __name__ == '__main__':
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    final_log_path_print = "log file (path not determined if error before logger init)"

    try:
        main()
        if LOGGER_INSTANCE and hasattr(LOGGER_INSTANCE, 'log_file_handle') and \
                LOGGER_INSTANCE.log_file_handle and not LOGGER_INSTANCE.log_file_handle.closed:
            log_print("Script main function completed successfully or as intended by mode.", level="info")
            if hasattr(LOGGER_INSTANCE.log_file_handle, 'name'):
                final_log_path_print = os.path.abspath(LOGGER_INSTANCE.log_file_handle.name)
        elif LOGGER_INSTANCE is None:
            _ORIGINAL_STDOUT.write("Script main function completed (Logger was not initialized by main).\n")
        else:
            _ORIGINAL_STDOUT.write("Script main function completed (Logger state uncertain or closed).\n")
    except KeyboardInterrupt:
        if LOGGER_INSTANCE and hasattr(LOGGER_INSTANCE, 'log_file_handle') and \
                LOGGER_INSTANCE.log_file_handle and not LOGGER_INSTANCE.log_file_handle.closed:
            log_print("Execution interrupted by user.", level="warn")
            if hasattr(LOGGER_INSTANCE.log_file_handle, 'name'):
                final_log_path_print = os.path.abspath(LOGGER_INSTANCE.log_file_handle.name)
        else:
            _ORIGINAL_STDOUT.write("Execution interrupted by user (Logger not available).\n")
        sys.exit(130)
    except Exception as e_global:
        if LOGGER_INSTANCE and hasattr(LOGGER_INSTANCE, 'log_file_handle') and \
                LOGGER_INSTANCE.log_file_handle and not LOGGER_INSTANCE.log_file_handle.closed:
            log_print(f"CRITICAL ERROR in script execution: {e_global}", level="error")
            import traceback

            log_print(traceback.format_exc(), level="error")
            if hasattr(LOGGER_INSTANCE.log_file_handle, 'name'):
                final_log_path_print = os.path.abspath(LOGGER_INSTANCE.log_file_handle.name)
        else:
            _ORIGINAL_STDERR.write(f"CRITICAL ERROR in script execution: {e_global}\n")
            import traceback

            _ORIGINAL_STDERR.write(traceback.format_exc())
            _ORIGINAL_STDERR.write("\n")
        sys.exit(1)
    finally:
        if sys.stdout == LOGGER_INSTANCE:
            sys.stdout = _ORIGINAL_STDOUT
        if sys.stderr == LOGGER_INSTANCE:
            sys.stderr = _ORIGINAL_STDERR
        if LOGGER_INSTANCE:
            if "path not determined" in final_log_path_print and \
                    hasattr(LOGGER_INSTANCE, 'log_file_handle') and LOGGER_INSTANCE.log_file_handle and \
                    hasattr(LOGGER_INSTANCE.log_file_handle, 'name') and not LOGGER_INSTANCE.log_file_handle.closed:
                final_log_path_print = os.path.abspath(LOGGER_INSTANCE.log_file_handle.name)
            LOGGER_INSTANCE.close()
            LOGGER_INSTANCE = None
        _ORIGINAL_STDOUT.write(f"\nLogging actions complete. Main log file should be at: {final_log_path_print}\n")
        _ORIGINAL_STDOUT.flush()