#!/usr/bin/env bash
# pccAI installation example
# Run "echo y | conda create -n pccai python=3.6 && conda activate pccai && ./install_torch-1.7.0+cu-10.1.sh"

# 1. Basic installation for pccAI
echo y | conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install tensorboard==2.9.0
pip install open3d==0.15.2
pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install torch-geometric==2.0.3
pip install numba==0.53.1

# 2. Additional installation for the examples

# Optional: nndistance for fast Chamfer Distance computation
cd third_party/nndistance
export PATH="/usr/local/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
python build.py install
cd ../..

# Optional: CompressAI for entropy modeling and coding, necessary for the benchmarking example "bench_ford_hetero.sh"
pip install compressai==1.1.1
