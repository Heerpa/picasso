pyGpufit (vendored) - DEPRECATED, SCHEDULED FOR REMOVAL
=======================================================

Picasso no longer uses this binding. GPU fitting is implemented in Numba CUDA
kernels (picasso/splinefit_cuda.py, picasso/gaussfit_cuda.py,
picasso/lmfit_cuda.py), which need no compiled library and work on Linux as
well as Windows - so nothing below has to be done any more. This folder is kept
for one release only, so that a result can still be compared against the binary
Picasso used to ship, by setting PICASSO_SPLINE_GPU_BACKEND=gpufit. It will be
deleted; do not write new code against it.

Everything that follows describes the old, deprecated arrangement.

---

This folder contains the Python binding (gpufit.py) for Gpufit
(https://github.com/gpufit/Gpufit), a CUDA Levenberg-Marquardt curve fitting
library. Picasso uses it as the GPU fitting backend for Picasso: Localize,
accelerating several of its fitting algorithms on a CUDA-capable NVIDIA GPU.
More algorithms will also be implemented in v0.11.0.

The binding loads a compiled Gpufit library that must sit next to gpufit.py:

- Windows: Gpufit.dll  - shipped here, so GPU fitting works out of the box.
- Linux:   libGpufit.so - NOT shipped, because the binary depends on your CUDA
           toolkit and GPU. You have to build it yourself (see below).

IMPORTANT: Picasso needs fit models that are not part of upstream Gpufit. Build
the library from our fork, https://github.com/rafalkowalewski1/Gpufit.

When no library can be loaded (or no CUDA GPU is available), Picasso hides the
GPU fitting option and falls back to the equivalent CPU implementation. Building
the library is optional and only possible if you have an NVIDIA (CUDA-capable)
GPU.


Building libGpufit.so on Linux
------------------------------

Prerequisites: an NVIDIA GPU with a matching CUDA toolkit
(https://developer.nvidia.com/cuda-downloads), CMake >= 3.11, a C/C++ compiler
(GCC) and git.

    git clone https://github.com/rafalkowalewski1/Gpufit.git Gpufit
    mkdir Gpufit-build
    cd Gpufit-build
    cmake -DCMAKE_BUILD_TYPE=RELEASE ../Gpufit
    make

If make fails with "unsupported GNU version! gcc versions later than X are not
supported", your CUDA toolkit needs an older GCC. Install one (e.g. gcc-5) and
point CMake at it:

    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_COMPILER=gcc-5 ../Gpufit

After a successful build, libGpufit.so is created inside the build directory
(under Gpufit-build/Gpufit/). Copy it into THIS folder:

    cp Gpufit-build/Gpufit/libGpufit.so /path/to/picasso/picasso/ext/pygpufit/

Restart Picasso: Localize - the GPU fitting option will appear next to the
supported fit methods once the library loads and a CUDA GPU is detected.

See the full Gpufit build documentation (section "Building from source code") at
https://gpufit.readthedocs.io, and the Picasso documentation at
https://picassosr.readthedocs.io/en/latest/localize.html#gpu-fitting-on-linux
