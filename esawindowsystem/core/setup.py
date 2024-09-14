from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["*.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(['demodulation_functions_c.pyx', 'max_star.pyx', 'get_num_events.pyx'])
)
