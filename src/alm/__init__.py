import torch
import os

try:
   torch.ops.load_library(os.path.realpath(os.path.join(__file__, "../../alm_ext_gpu.so")))
   CUDA_EXT_AVAILABLE = True
except OSError:
   torch.ops.load_library(os.path.realpath(os.path.join(__file__, "../../alm_ext_cpu.so")))
   CUDA_EXT_AVAILABLE = False
