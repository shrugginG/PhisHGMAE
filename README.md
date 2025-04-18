# PhisHGMAE: Phishing Detection Using Heterogeneous Graph Masked Autoencoder

## Overview
PhisHGMAE is a deep learning-based project for phishing website detection that utilizes Heterogeneous Graph Masked Autoencoder techniques. The model leverages metapath-based approaches to learn effective representations of URL relationships and features, enabling accurate classification of phishing vs. legitimate websites.

## Features
- Heterogeneous graph-based approach for phishing detection
- Masked autoencoder architecture for robust feature learning
- MetaPath2Vec implementation for metapath-based node embeddings
- Support for multiple metapaths and heterogeneous graph structures
- Classification tasks with comprehensive evaluation metrics

## Requirements
- Python 3.12+
- PyTorch 2.3.1
- PyTorch Geometric 2.6.1
- Scikit-learn
- Loguru

## Installation

### Using uv (Recommended)
PhisHGMAE uses [uv](https://github.com/astral-sh/uv) as its package management and virtual environment system.

1. Install uv (if not already installed):
```bash
brew install uv
```

2. Clone the repository:
```bash
git clone <repository-url>
cd PhisHGMAE
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install -e .
```

## Usage

### Data Preparation
Prepare your dataset in the appropriate format and place it in the `data/` directory. The system expects heterogeneous graph data with specific metapath structures.

### Configuration
Edit the `configs.yml` file to set model hyperparameters and experiment configurations. The configuration includes parameters for:

- Model architecture (hidden dimensions, layers, etc.)
- Training settings (learning rate, epochs, etc.)
- MetaPath2Vec configurations
- Masking rates for the autoencoder

### Training and Evaluation
To train the model and evaluate its performance:

```bash
uv run main.py --task classification --use_cfg
```

For the MLP-based evaluation with different training ratios:
```bash
python evaluate_phishgmae.py
```

### Outputs
- Trained model embeddings will be saved to the `embeddings/` directory
- Evaluation results will be displayed in the console and saved to the `results/` directory
- Logs are stored in the `logs/` directory

## Project Structure
- `main.py`: Main entry point for training and evaluation
- `models/`: Contains model architecture definitions
  - `edcoder.py`: Encoder-decoder architecture
  - `han.py`: Hierarchical Attention Network implementation
  - `gat.py`: Graph Attention Network implementation
- `utils/`: Utility functions and helpers
- `data/`: Directory for storing datasets
- `configs.yml`: Configuration file for experiments
- `run_claude_mlp_pro.py`: Script for MLP-based evaluation