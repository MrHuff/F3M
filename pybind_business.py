from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='ffm_3d_float',
      ext_modules=[CUDAExtension('ffm_3d_float', sources=['n_tree.cu'],extra_compile_args={'cxx':['-g'],
                                                                                    'nvcc': ['-Xptxas', '-O3','--use_fast_math','-arch=sm_61']})],
      cmdclass={'build_ext': BuildExtension})