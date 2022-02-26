from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CppExtension


setup(name='my_lib_cuda',
     ext_modules=[CppExtension('my_lib_cuda', ['src/my_lib.cpp'])],
     cmdclass={'build_ext': BuildExtension})


#if __name__ == '__main__':
#    ffi.build()
