# InterActRNA
This is the official repository for the paper: **"InterActRNA: a co-evolutionary geometric deep learning framework for RNA-small molecule binding affinity prediction"**.

InterActRNA is an end-to-end deep learning framework designed to predict RNA-Small Molecule Affinity (RSMA). By introducing an **Interleaved Guided Encoding Module**, it computationally simulates the dynamic **"Induced Fit"** mechanism in molecular recognition, addressing the limitations of traditional static encoding methods.

![Model Framework](model.png)
*(Note: Please upload your `model.png` to the repository to display the framework here)*

## 🌟 Key Features

*   **Dynamic Interaction Simulation**: Simulates the "Induced Fit" process via a co-evolutionary attention mechanism between the Iterative Drug Encoder and Hierarchical RNA Encoder.
*   **Multi-scale Representation**: Captures both atomic-level and motif-level features for small molecules and RNAs.
*   **Interpretability**: Includes a Cross-Modality Substructure Similarity (SSIM) module to explicitly model pairwise interactions between functional groups and RNA motifs.
*   **SOTA Performance**: Significantly outperforms baselines (e.g., DeepRSMA, GraphDTA) in both cross-validation and rigorous **Cold-Start** scenarios.

## 🛠️ Requirements

The code is implemented in Python using PyTorch and PyTorch Geometric.

*   Python >= 3.8
*   PyTorch >= 1.10
*   PyTorch Geometric (PyG)
*   RDKit
*   Pandas
*   Numpy
*   Scikit-learn
*   Tqdm

## 📂 Dataset

The model is evaluated on the **R-SIM** dataset, which contains 1,439 RNA-small molecule pairs.
Please place the data files in the `data/` directory.

## 🚀 Usage

### 1. Standard Training (Cross-Validation)
To run the standard 5-fold cross-validation on the R-SIM dataset:

```bash
python main.py
