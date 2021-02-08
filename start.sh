rm -rf venv
python3 -m virtualenv venv
. venv/bin/activate
echo "Using..."
which python
which pip
echo "Let's install PyTorch..."
pip install torch

echo "Let's test some things..."
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Number of devices:', torch.cuda.device_count())"
python -c """import torch
print('Devices:')
for device in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(device))
"""

echo "Let's train something..."
pip install torchvision poutyne
python train.py
rm -rf MNIST

deactivate
rm -rf venv
