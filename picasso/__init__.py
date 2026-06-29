"""
picasso.__init__.py
~~~~~~~~~~~~~~~~~~~

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

import os.path
import yaml
from .version import __version__  # noqa: F401

# In frozen (PyInstaller) builds the numba-cuda redirect (a site-packages .pth)
# never runs, so "from numba import cuda" would fall back to Numba's built-in
# CUDA stub and GPU-accelerated (numba.cuda) code silently vanishes. Install the
# redirect here, before anything imports numba.cuda. No-op outside frozen builds.
from . import _numba_cuda_compat as _numba_cuda_compat

_numba_cuda_compat.install()

_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)
try:
    with open(os.path.join(_this_dir, "config.yaml"), "r") as config_file:
        CONFIG = yaml.full_load(config_file)
    if CONFIG is None:
        CONFIG = {}
except FileNotFoundError:
    CONFIG = {}
