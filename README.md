# Implementation of "On the Effect of Key Factors in Spurious Correlation: A Theoretical Perspective"

This repository contains the implementation of the paper "On the Effect of Key Factors in Spurious Correlation: A Theoretical Perspective," which has been accepted at AISTATS 2024.

## Prerequisites

Before executing the code, please ensure to have the following:

1. **Datasets**: 
   - CIFAR-10
   - CUB-200-2011
   - Places256
   - CelebA

2. **Hardware**: 
   - GPU with at least 16GB memory

## Datasets

The datasets are processed using the following scripts:
- `CIFAR_dataset.py`
- `waterbird_dataset.py`
- `CelebA_dataset.py`

## Quick Start

To get started, check out `quick_start.ipynb`. This notebook demonstrates:
- Training the original model using CIFAR-watermark dataset.
- Training a regularized model.
- Implementing Group-DRO, following the official release.

## Experiments

To replicate the experiments presented in the paper:
- Execute `train_CelebA.py` for the CelebA dataset.
- Execute `train_waterbird.py` for the Waterbird dataset.

Please modify the `data_root`, `CUB_root`, and `places_root` variables in the scripts to point to your local dataset paths. Note that for the Waterbird dataset, both CUB-200-2011 and Places256 datasets are required.
