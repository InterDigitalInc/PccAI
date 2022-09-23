#!/usr/bin/env bash
# PccAI installation example
# Run "echo y | conda create -n pccai python=3.8 && conda activate pccai && ./install_torch-1.8.1+cu-11.2.sh"

# 1. Basic installation for PccAI
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==2.8.0
pip install plyfile==0.7.4
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric==2.0.3

# 2. Additional installation for the examples

# Optional: nndistance for fast Chamfer Distance computation
cd third_party/nndistance
export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
python build.py install
cd ../..

# Optional: CompressAI for entropy modeling and coding, necessary for the benchmarking example "bench_ford_hetero.sh"
pip install compressai==1.1.1

# Optional: Open3D for visualization
pip install open3d==0.14.1
