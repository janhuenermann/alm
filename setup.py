import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


ext_modules = []

try:
    ext_modules.append(CUDAExtension('alm_ext_gpu', [
        "src/alm/ext/polygon_cpu.cpp",
        "src/alm/ext/polygon_gpu.cpp",
        "src/alm/ext/polygon_gpu_kernel.cu",
    ]))
except OSError:
    ext_modules.append(CppExtension('alm_ext_cpu', ["src/alm/ext/polygon_cpu.cpp"]))


setuptools.setup(
    name="alm",
    version="0.0.1",
    author="Jan Hünermann",
    author_email="mail@janhuenermann.com",
    description="Metrics for ML",
    url="https://github.com/janhuenermann/alm",
    project_urls={"Bug Tracker": "https://github.com/janhuenermann/alm/issues",},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["torch>=1.8.0"],
    python_requires=">=3.6",
    package_data={'': ['./geometry/native/*']},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
)
