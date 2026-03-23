import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    raise RuntimeError('No GPU available.')
total_memory = torch.cuda.get_device_properties(0).total_memory
try:
    # 70%
    tensor = torch.empty(int(total_memory * 0.95), dtype=torch.uint8, device=device)
    print(f'Allocated {tensor.numel()} bytes on GPU')
except RuntimeError as e:
    print(f'Failed to allocate memory: {e}')


while True:
    time.sleep(5)