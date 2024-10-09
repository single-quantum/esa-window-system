from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["*.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(['max_star.pyx'])
)
