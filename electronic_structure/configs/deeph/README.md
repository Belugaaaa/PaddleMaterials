# DeepH

## Overview

DeepH is a machine-learning Hamiltonian model for electronic-structure prediction.
Compared with the current density-oriented MLES models in PaddleMaterials, DeepH
targets edge-level Hamiltonian matrix elements on crystal graphs and is therefore a
useful complement to the existing electron-density workflow.

## Current integration status

- Model migration baseline has been validated outside PaddleMaterials:
  - forward alignment
  - short-horizon training alignment
  - full `graphene` supervised loss alignment
- PaddleMaterials-side integration now has a runnable training skeleton:
  - model implementation under `ppmat/models/deeph`
  - dataset adapter under `ppmat/datasets/deeph_dataset.py`
  - unified collator under `ppmat/datasets/collate_fn.py`
  - task entry under `electronic_structure`
  - predictor under `ppmat/predictor/deeph_predictor.py`
  - sampler under `ppmat/sampler/deeph_sampler.py`
- The following standard-path smoke checks have passed:
  - `build_dataloader()` returns DeepH LCMP batches
  - `build_model()` instantiates `DeepHHamiltonian`
  - one collated batch can run a full forward pass and produce `loss_dict/pred_dict`
  - `PaddleMaterials/electronic_structure/train.py` has completed a full `graphene` GPU run
- The dataset adapter now reconstructs crystal structures through
  `BuildStructure(format='array')` before exposing sample-level structure metadata.
- The adapter can also build DeepH graph caches inside PaddleMaterials
  (`PPMatDeepHGraph-...pt`) from those canonical structures.

## Acceptance evidence

- Single-card forward alignment:
  - `graphene_case_v2`: `max_abs_diff = 1.1444091796875e-05`
  - `TBG_subset_case_v1`: `max_abs_diff = 1.33514404296875e-05`
- Two-step training/loss alignment:
  - `graphene_case_v2`: step 1 `loss_diff = 4.0531158447265625e-06`
  - `TBG_subset_case_v1`: step 1 `loss_diff = 1.6808509826660156e-05`
- Supervised metric alignment on `graphene`:
  - Torch `train/val/test = 0.01066117 / 0.01061319 / 0.01061640`
  - Paddle `train/val/test = 0.01077699 / 0.01072210 / 0.01072441`
  - absolute `test_loss` diff: `1.0801231873300288e-04`
- PaddleMaterials standard trainer run:
  - output: `output/deeph_graphene`
- Compiler on/off performance comparison:
  - dynamic avg: `21.6952 ms`
  - to_static/CINN avg: `11.4824 ms`
  - speedup: `88.9434%`
  - evidence JSON: `runs_paddle/graphene_full/compiler_eval_benchmark.json`

## Files

- `ppmat/models/deeph/deeph.py`
- `electronic_structure/configs/deeph/*.yaml`
- dataset and collate support under `ppmat/datasets`

## Notes

- The final PR should add Baidu cloud links for:
  - processed dataset
  - pretrained model
  - logs
- The current dataset adapter intentionally reuses DeepH's existing processed-graph
  pipeline for graph construction so we can align with the standard PaddleMaterials trainer first.
- The remaining strict-compliance gap is that the low-level graph algorithm still
  reuses DeepH's mature `get_graph()` implementation instead of a fully rewritten
  PaddleMaterials-native graph factory.
- `ImageNet` accuracy and generative sampling metrics are not applicable because
  DeepH is a supervised Hamiltonian-regression model.

## Resource link placeholders

Fill these placeholders after the reviewer returns the official BCE links.

- Processed dataset:
  - Official BCE link pending before merge.
- Pretrained model:
  - Official BCE link pending before merge.
- Training and alignment logs:
  - Official BCE link pending before merge.

## Run

Prepare the DeepH graphene data package so the config file is available at
`./data/deeph/graphene/config.ini`, then run from the PaddleMaterials repository
root:

```bash
python electronic_structure/train.py \
  -c electronic_structure/configs/deeph/deeph_graphene.yaml
```
