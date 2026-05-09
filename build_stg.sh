#!/usr/bin/env bash
set -e  # stop on error

echo "🚀 Setting up CG-STVG environment..."

# -------------------------
# 1. Create conda env
# -------------------------
ENV_NAME=stg

echo "📦 Creating conda env: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.11

echo "🔄 Activating env"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# -------------------------
# 2. Install PyTorch (CUDA 11.7)
# -------------------------
echo "🔥 Installing PyTorch 2.0.1 (CUDA 11.7)"
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# -------------------------
# 3. Install project deps
# -------------------------
echo "📚 Installing requirements"
pip install -r requirements.txt

# Fix common issues
pip install "numpy<2"
pip install einops easydict pytorch-pretrained-bert

# -------------------------
# 4. Fix transformers compatibility
# -------------------------
echo "🤗 Installing compatible transformers"
pip uninstall -y transformers tokenizers || true
pip install "transformers==4.30.2" "tokenizers<0.14"

# -------------------------
# 6. Fix tokenizer path bug (repo bug)
# -------------------------
echo "🔧 Fixing roberta path bug"
ln -sf model_zoo/roberta-base model_zoo/roberta

# -------------------------
# 7. Set PYTHONPATH
# -------------------------
export PYTHONPATH=$PWD:$PYTHONPATH

echo "✅ Setup complete!"
echo ""
echo "👉 To train:"
echo "conda activate $ENV_NAME"
echo "python -m torch.distributed.launch --nproc_per_node=1 scripts/train_net.py \\"
echo "  --config-file experiments/hcstvg.yaml \\"
echo "  INPUT.RESOLUTION 420 \\"
echo "  OUTPUT_DIR output/hcstvg \\"
echo "  TENSORBOARD_DIR output/hcstvg"