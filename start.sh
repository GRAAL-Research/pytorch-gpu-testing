nvidia-smi

rm -rf venv
python3 -m venv venv
. venv/bin/activate
echo "Using..."
which python
which pip
echo "Let's install PyTorch..."
pip install torch

echo "Let's test some things..."
python test_pytorch.py

echo "Let's train something..."
pip install torchvision poutyne
python train.py
rm -rf MNIST

deactivate
rm -rf venv
