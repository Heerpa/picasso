"""PyInstaller hook: bundle numba_cuda's vendored compiled ``cext`` extensions.

numba-cuda's CUDA target ships its own compiled C/C++ extensions -- ``_dispatcher``,
``_typeconv``, ``_helperlib``, ``mviewbuf`` -- under ``numba_cuda/numba/cuda/cext``.
They share basenames and ``PyInit_<name>`` symbols with base Numba's identically
named extensions, so PyInstaller keeps Numba's copies and silently drops
numba_cuda's vendored ones.

But they are NOT interchangeable: numba_cuda's ``_dispatcher`` adds CUDA-specific
entry points (e.g. ``swap_current_launch_args``) that base Numba's lacks, so the
``numba.cuda`` redirector falling back to Numba's copy makes kernel launches fail
at runtime with ``AttributeError: module 'numba.cuda.cext._dispatcher' has no
attribute 'swap_current_launch_args'``.

Copy numba_cuda's ``cext`` extensions verbatim to their real path -- as *data*, so
PyInstaller does not relocate them by module name and re-trigger the basename
collision -- so ``_numba_cuda_compat.py`` finds numba_cuda's *own* extensions
(PathFinder, its first strategy) instead of borrowing Numba's.
"""

import glob
import os

from PyInstaller.utils.hooks import get_package_paths

# numba_cuda is a normal package (no runtime __path__ rewriting), so
# get_package_paths is reliable: it returns (base, pkg_dir) with base being the
# site-packages directory.
_site_packages, _pkg_dir = get_package_paths("numba_cuda")
_cext_dir = os.path.join(_pkg_dir, "numba", "cuda", "cext")
_cext_rel = os.path.relpath(
    _cext_dir, _site_packages
)  # numba_cuda/numba/cuda/cext

datas = [
    (src, _cext_rel)
    for src in glob.glob(os.path.join(_cext_dir, "*.pyd"))
    + glob.glob(os.path.join(_cext_dir, "*.so"))
]

# numba_cuda's wheel is delvewheel-repaired: its cext extensions link a private,
# hash-mangled copy of msvcp140 vendored in a top-level ``numba_cuda.libs``
# directory (e.g. msvcp140-<hash>.dll). Copying the .pyd as data above skips
# PyInstaller's dependency analysis, so that DLL is otherwise left out and the
# extensions fail with "DLL load failed while importing _typeconv". numba_cuda's
# __init__ has no add_dll_directory patch, so bundle the DLL right next to the
# extensions -- Windows resolves a module's dependencies from its own directory.
_libs_dir = os.path.join(_site_packages, "numba_cuda.libs")
datas += [
    (dll, _cext_rel) for dll in glob.glob(os.path.join(_libs_dir, "*.dll"))
]
