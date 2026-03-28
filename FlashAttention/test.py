import torch
import triton

# 查询当前 GPU 的 shared memory 大小
device = torch.cuda.current_device()
shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
print(f"Max shared memory: {shared_mem / 1024:.0f} KB")