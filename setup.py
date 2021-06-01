from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

exts = []

try:
    exts.append(CUDAExtension(
        name='alm_ext_gpu', 
        sources=[
            'geometry/native/sutherland_hodgman_gpu.cpp',
            'geometry/native/sutherland_hodgman_gpu_kernel.cu'
        ],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-Xptxas", "-O3", "-use_fast_math"]}))
except OSError:
    pass

exts.append(CppExtension(name='alm_ext_cpu', 
    sources=['geometry/native/sutherland_hodgman_cpu.cpp'],
    extra_compile_args=["-O3", "-funroll-loops"]))

setup(name='alm_ext', ext_modules=exts, cmdclass={'build_ext': BuildExtension})
