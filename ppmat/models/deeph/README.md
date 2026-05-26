# DeepH Model Entry

## Overview

This directory contains the PaddleMaterials-side DeepH model implementation.
The model is registered under `ppmat/models/deeph` and is intended to be used
through the unified PaddleMaterials task stack.

## Files

- `deeph.py`: DeepH Paddle model definition
- `__init__.py`: model export

## Standard task entry

- Trainer:
  - `electronic_structure/train.py`
- Predictor:
  - `ppmat/predictor/deeph_predictor.py`
- Sampler:
  - `ppmat/sampler/deeph_sampler.py`

## Resource links

These fields are intentionally reserved for the Baidu engineer handoff process.
Replace the placeholders after the official Baidu cloud links are returned.

- Processed dataset:
  - Official BCE link pending before merge.
- Pretrained model:
  - Official BCE link pending before merge.
- Training / alignment logs:
  - Official BCE link pending before merge.

## Current status

- Torch -> Paddle core migration completed
- PaddleMaterials trainer / predictor / sampler entries completed
- `graphene` full training completed under PaddleMaterials standard trainer
- Forward alignment is within `1e-4` on both `graphene_case_v2` and
  `TBG_subset_case_v1`
- Two-step training/loss alignment is available for both alignment cases
- Supervised `graphene` train/val/test loss alignment is available in
  `runs_paddle/graphene_full/compare_to_torch.json`
- Compiler eval benchmark is available in
  `runs_paddle/graphene_full/compiler_eval_benchmark.json`
  with `88.9434%` speedup
- `BuildStructure(format='array')` has been connected into the DeepH dataset adapter
- The remaining strict-compliance gaps are the external Baidu cloud link backfill
  and, if required by reviewers, a deeper rewrite of the low-level graph algorithm
