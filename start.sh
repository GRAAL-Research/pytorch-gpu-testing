rm -rf venv
python3 -m venv venv
. venv/bin/activate
echo "Using..."
which python
which pip
echo "Let's install PyTorch..."
pip install torch

echo "Let's test some things..."
python -c """import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Number of devices:', torch.cuda.device_count())
print('Devices:')
for device in range(torch.cuda.device_count()):
    print(f'{device}:', torch.cuda.get_device_name(device))
"""

echo "Let's train something..."
pip install torchvision poutyne
python train.py
rm -rf MNIST

deactivate
rm -rf venv
