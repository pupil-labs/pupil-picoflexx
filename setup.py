from Cython.Build import cythonize
from Cython.Distutils import Extension
from distutils.core import setup

extensions = [
    Extension(
        "roypycy",
        ["roypycy.pyx"],
        library_dirs=["."],
        libraries=["royale"],
        include_dirs=["include"],
        language="c++",
        extra_compile_args=["-std=c++0x"],
    )
]

setup(name="Pico Flexx", ext_modules=cythonize(extensions))
