# Wavefunction-Alignment-Net

A neural network framework for accelerating Density Functional Theory (DFT) calculations through machine learning. This repository implements state-of-the-art neural network architectures for predicting molecular properties, Hamiltonian matrices, and wavefunctions.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Configuration](#configuration)
- [Models](#models)
- [Datasets](#datasets)
- [Scripts](#scripts)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Features

- **Multiple Model Architectures**: Support for QHNet, Equiformer, ViSNet, and LSR variants
- **Hamiltonian Prediction**: Direct prediction of molecular Hamiltonian matrices
- **Multi-task Learning**: Simultaneous prediction of energies, forces, and Hamiltonian matrices
- **Symmetry-aware Models**: Incorporation of physical symmetries (SO(2), SO(3))
- **Distributed Training**: Support for multi-GPU training with PyTorch Lightning
- **Flexible Configuration**: Hydra-based configuration management
- **Experiment Tracking**: Integration with Weights & Biases (wandb)

## Architecture

The framework supports several neural network architectures:

1. **QHNet**: Quantum Hamiltonian Network for molecular property prediction
2. **Equiformer**: Equivariant transformer architecture with attention mechanisms
3. **ViSNet**: Vector-Scalar Interaction networks
4. **LSR Models**: Large-scale representation models with improved efficiency

### Key Components:
- **Representation Models**: Extract molecular features from atomic coordinates
- **Output Modules**: Task-specific heads for energy, forces, and Hamiltonian prediction
- **Hamiltonian Heads**: Specialized modules for predicting quantum mechanical Hamiltonians

## Installation

### Prerequisites
- CUDA driver version ≥ 520 (for CUDA Toolkit 12 support)
- Python 3.10+
- Conda/Mamba package manager

### Environment Setup

Create a virtual environment with all necessary packages:

```bash
# Clone the repository
git clone https://github.com/your-repo/Wavefunction-Alignment-Net.git
cd Wavefunction-Alignment-Net

# Create conda environment
conda env create -n madft_nn -f environment.yaml

# Activate environment
conda activate madft_nn

# Install the custom CUDA DFT wheel (if needed)
pip install cudft-0.2.6-cp310-cp310-linux_x86_64.whl
```

### Key Dependencies
- PyTorch 2.1.0 with CUDA 12.1
- PyTorch Lightning
- e3nn (for equivariant neural networks)
- PyTorch Geometric
- PySCF 2.4.0
- Weights & Biases
- Hydra configuration framework

## Quick Start

### Basic Training Example

Train a Hamiltonian prediction model on QH9 dataset:

```bash
python pipelines/train.py \
    --config-name=config.yaml \
    --wandb=True \
    --wandb-group="test" \
    --dataset-path="/path/to/QH9_new.db" \
    --ngpus=4 \
    --lr=0.0005 \
    --enable-hami=True \
    --hami-weight=1 \
    --batch-size=32 \
    --lr-warmup-steps=1000 \
    --lr-schedule="polynomial" \
    --max-steps=300000 \
    --used-cache=True \
    --train-ratio=0.9 \
    --val-ratio=0.06 \
    --test-ratio=0.04 \
    --gradient-clip-val=5.0 \
    --dataset-size=100000
```

## Training

### Model Training Pipeline

The training pipeline (`pipelines/train.py`) supports:
- Multi-GPU distributed training
- Automatic mixed precision
- Gradient clipping
- Learning rate scheduling (cosine, polynomial, reduce-on-plateau)
- Early stopping
- Model checkpointing

### Training Parameters

Key training parameters:
- `model_backbone`: Choose from QHNet, Equiformerv2, LSR_QHNet, ViSNet
- `enable_hami`: Enable Hamiltonian prediction
- `enable_energy`: Enable energy prediction
- `enable_forces`: Enable force prediction
- `hami_weight`, `energy_weight`, `forces_weight`: Loss function weights
- `lr_schedule`: Learning rate scheduler type
- `max_steps`: Maximum training steps
- `batch_size`: Training batch size

### Multi-task Training

Enable multiple prediction tasks simultaneously:

```bash
python pipelines/train.py \
    --enable-hami=True \
    --enable-energy=True \
    --enable-forces=True \
    --hami-weight=1.0 \
    --energy-weight=0.1 \
    --forces-weight=0.01
```

## Configuration

### Configuration System

The project uses Hydra for configuration management. Configuration files are located in `config/`:

- `config.yaml`: Default configuration
- `config_equiformer.yaml`: Equiformer-specific settings
- `config_lsrqhnet.yaml`: LSR-QHNet configuration
- `config_lsrm.yaml`: LSRM model configuration
- `model/`: Model-specific configurations
- `schedule/`: Learning rate schedule configurations

### Custom Configuration

Create custom configurations by:
1. Creating a new YAML file in `config/`
2. Overriding parameters via command line
3. Using Hydra's composition feature

Example custom config:
```yaml
defaults:
  - model: equiformerv2
  - schedule: polynomial
  - _self_

model_backbone: Equiformerv2SO2
batch_size: 64
lr: 1e-3
max_steps: 500000
```

## Models

### Available Models

1. **QHNet Variants**:
   - `QHNet_backbone`: Basic QHNet architecture
   - `QHNet_backbone_No`: QHNet without specific features
   - `QHNetBackBoneSO2`: SO(2) symmetric QHNet

2. **Equiformer Variants**:
   - `Equiformerv2`: Equivariant transformer v2
   - `Equiformerv2SO2`: SO(2) symmetric Equiformer
   - `GraphAttentionTransformer`: Graph attention-based transformer

3. **LSR Models**:
   - `LSR_QHNet_backbone`: Large-scale representation QHNet
   - `LSR_Equiformerv2SO2`: LSR variant of Equiformer
   - `LSRM`: Standalone LSR model

4. **Other Models**:
   - `ViSNet`: Vision-inspired molecular network
   - `spherical_visnet`: Spherical harmonics-based ViSNet

### Model Components

- **Representation Models**: Extract invariant/equivariant features
- **Output Modules**: 
  - `EquivariantScalar_viaTP`: Tensor product-based scalar output
  - Energy/force prediction heads
- **Hamiltonian Heads**:
  - `HamiHead`: Basic Hamiltonian prediction
  - `HamiHeadSymmetry`: Symmetry-aware Hamiltonian prediction

## Datasets

### Supported Datasets

1. **QH9**: Quantum Hamiltonian dataset with ~130k molecules
2. **Water Clusters**: Various water cluster configurations (W3, W6, W15, etc.)
3. **PubChem**: Subset of PubChem molecules
4. **Malondialdehyde**: Conformational dataset

### Dataset Format

Datasets are stored in SQLite or LMDB format with the following structure:
- Atomic coordinates
- Atomic numbers
- Hamiltonian matrices (if available)
- Energy values
- Force vectors
- Additional quantum mechanical properties

### Data Preparation

Use the data preparation utilities in `src/madftnn/data_prepare/` to:
- Convert raw data to database format
- Generate train/validation/test splits
- Apply data augmentation

## Scripts

### Training Scripts

Located in `scripts/`:
- `run_qh9.sh`: Train on QH9 dataset
- `run_water.sh`: Train on water clusters
- `run_pubchem.sh`: Train on PubChem molecules
- `run_malondialdehyde.sh`: Train on malondialdehyde conformations

### Utility Scripts

- `test_script.sh`: Testing and validation scripts
- `job.sh`: HPC job submission script

## Project Structure

```
Wavefunction-Alignment-Net/
├── config/                 # Configuration files
│   ├── model/             # Model-specific configs
│   └── schedule/          # LR schedule configs
├── src/madftnn/           # Main source code
│   ├── models/            # Neural network models
│   ├── training/          # Training utilities
│   ├── dataset/           # Dataset handling
│   ├── utility/           # Helper functions
│   └── analysis/          # Analysis tools
├── pipelines/             # Training and testing pipelines
├── scripts/               # Utility scripts
├── dataset/               # Dataset storage
├── outputs/               # Training outputs
├── wandb/                 # Weights & Biases logs
├── local_files/           # Local configurations (git-ignored)
└── environment.yaml       # Conda environment specification
```

## Advanced Usage

### Custom Model Implementation

To implement a custom model:

1. Create a new model class in `src/madftnn/models/`
2. Inherit from appropriate base class
3. Register the model in `model.py`
4. Create corresponding configuration

### Distributed Training

For multi-node training:

```bash
python pipelines/train.py \
    --ngpus=8 \
    --num-nodes=4 \
    --strategy=ddp
```

### Experiment Tracking

All experiments are tracked using Weights & Biases:
- Set `wandb.open=True` to enable tracking
- Configure project and group names
- Monitor training progress in real-time

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
li2025enhancing,
title={Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems},
author={Yunyang Li and Zaishuo Xia and Lin Huang and Xinran Wei and Samuel Harshe and Han Yang and Erpai Luo and Zun Wang and Jia Zhang and Chang Liu and Bin Shao and Mark Gerstein},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=twEvvkQqPS}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on PyTorch and PyTorch Lightning
- Uses e3nn for equivariant neural networks
- Incorporates ideas from QHNet, Equiformer, and ViSNet architectures

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
