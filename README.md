# PhisHGMAE
## Python ENV
```bash
# cuda
cuda_version = 12.1
# Creata conda virtual env
conda create -n phishgmae python=3.10
# Activate conda virtual env
conda activate phishgmae
# Install dependencies
# DGL can only support up to pytorch 2.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install torch_geometric
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip3 install -U scikit-learn
```
