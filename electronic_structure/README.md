# MLES-Machine Learning Electronic Structure

## 1.Introduction

Machine Learning Electronic Structure (MLES) is an emerging paradigm in computational chemistry and materials science that leverages machine learning to accelerate or even replace traditional ab initio electronic structure methods. It aims to retain quantum accuracy while drastically reducing computational costs. Current research in MLES can be broadly categorized into several directions: Neural Quantum States, Graph-Based Electronic Structure Models, ML Hamiltonians, Neural XC, SCF Accelerators etc. MLES has demonstrated strong potential in predicting material properties, guiding molecular design, and understanding catalytic mechanisms, making it an increasingly important tool in computational materials science and quantum chemistry.

## 2.Models Matrix

| **Supported Functions**                      | **[InfGCN](./configs/infgcn/README.md)** | **[DeepH](./configs/deeph/README.md)** |
| -------------------------------------------- | :--------: | :--------: |
| **Forward Prediction · Materials Properties**|            |            |
| Electron density                             |      ✅    |      —     |
| Hamiltonian matrix elements                  |      —     |      ✅    |
| **ML Capabilities · Training**               |            |            |
| Single-GPU                                   |      ✅    |      ✅    |
| Distributed training                         |      ✅    |      —     |
| Mixed precision (AMP)                        |      —     |      —     |
| Fine-tuning                                  |      ✅    |      ✅    |
| Uncertainty / Active Learning                |      —     |      —     |
| Dynamic→Static graphs                        |      —     |      pending |
| Compiler (CINN) opt.                         |      —     |      pending |
| **ML Capabilities · Predict**                |            |            |
| Distillation / Pruning                       |      —     |      —     |
| Standard inference                           |      ✅    |      ✅    |
| Distributed inference                        |      —     |      —     |
| Compiler-level inference                     |      —     |      pending |
| **Datasets**                                 |            |            |
| Graphene Hamiltonian                         |      —     |      ✅    |
| TBG Hamiltonian                              |      —     |      subset |

DeepH status note:

- `ppmat/models/deeph`
- `ppmat/datasets/deeph_dataset.py`
- `ppmat/datasets/collate_fn.py::DeepHCollator`
- `electronic_structure/configs/deeph/deeph_graphene.yaml`

These pieces already allow DeepH to enter the standard PaddleMaterials
`build_dataloader -> build_model -> BaseTrainer` path. Predictor / sampler
entries are available, while compiler benchmarking still needs final GPU timing
numbers for the MIIT acceptance material.

**Notice**:🌟 represent originate research work published from paddlematerials toolkit
