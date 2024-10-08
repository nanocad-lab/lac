from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import shutil
import os

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang"
os.environ['CFLAGS'] = '-march=native -fopenmp'

#os.environ['CFLAGS'] = '-march=native -fopenmp'
# shutil.rmtree("build")
# shutil.rmtree("mult_approx.egg-info")
# shutil.rmtree("dist")

setup(name='mult_approx',
      ext_modules=[CppExtension('mult_approx', ['applications/mult_approx.cpp'])],
     # extra_compile_args ={'cxx': ['-march=native', '-g3', '-fopenmp', 'std=c++20']},
      cmdclass={'build_ext': BuildExtension})