pyGpuspline (vendored)
======================

This folder contains the Python binding (gpuspline.py) for Gpuspline
(https://github.com/gpufit/Gpuspline), a library for computing multidimensional
cubic-spline coefficients. Picasso uses it to build cubic-spline PSF
calibrations from an averaged bead z-stack; those coefficients are then fitted
per spot with Gpufit's SPLINE_2D / SPLINE_3D models (see picasso.localize and
picasso/ext/pygpufit).

Note: despite the "Gpu" in the name (kept for consistency with Gpufit),
Gpuspline is a plain CPU C++ library - it does NOT use CUDA or a GPU. Computing
the coefficients runs entirely on the CPU. Only the subsequent *fitting* step
(Gpufit) needs a CUDA GPU.

Division of labour:

- Gpuspline (this package, CPU): turns a measured/averaged PSF template into
  cubic spline coefficients (spline_coefficients) and evaluates them
  (spline_values). Needed only when GENERATING a spline calibration.
- Gpufit (../pygpufit, CUDA GPU): fits the spline model to the data using those
  coefficients. Needed to fit with an existing calibration.

Because the coefficients are stored inside the calibration file, fitting an
existing calibration needs only Gpufit - Gpuspline is required only to create a
new calibration.

The compiled library
--------------------

The binding loads a compiled Gpuspline library that must sit next to
gpuspline.py:

- Windows: splines.dll
- Linux:   libsplines.so
- macOS:   libsplines.dylib is not searched for by the upstream binding
           (it looks for libsplines.so under os.name == "posix").

The binary is not shipped here. Build it yourself (or drop in a prebuilt one)
to enable spline calibration generation. When no library can be loaded, Picasso
sets GPUSPLINE_INSTALLED = False and the calibration-generation step is
disabled; fitting with an already-generated calibration still works as long as
Gpufit is available.

Building
--------

Gpuspline uses CMake >= 3.11, a C/C++ compiler and git:

    git clone https://github.com/gpufit/Gpuspline.git Gpuspline
    mkdir Gpuspline-build
    cd Gpuspline-build
    cmake -DCMAKE_BUILD_TYPE=RELEASE ../Gpuspline
    cmake --build . --config Release

Copy the resulting splines.dll (Windows) or libsplines.so (Linux) into this
folder, next to gpuspline.py. Build it single-precision (REAL = float) so its
coefficients match the single-precision Gpufit build used for fitting.

License
-------

See LICENSE.txt in this folder. Replace it with the exact LICENSE shipped with
the Gpuspline source you build from.
