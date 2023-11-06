import torch
import os
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
