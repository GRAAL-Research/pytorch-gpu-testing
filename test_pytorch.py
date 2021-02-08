import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available()
print('Number of devices:', torch.cuda.device_count())
assert torch.cuda.device_count() > 0
print('Devices:')
for device in range(torch.cuda.device_count()):
    print(f'{device}:', torch.cuda.get_device_name(device))
