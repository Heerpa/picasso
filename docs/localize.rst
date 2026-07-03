localize
========

.. image:: ../docs/localize.png
   :scale: 50 %
   :alt: UML Localize

Localize allows performing super-resolution reconstruction of image stacks. For spot detection, a gradient-based approach is used. For Fitting, the following algorithms are implemented:

- MLE, integrated Gaussian (based on `Smith et al., 2010 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2862147/>`_.)
- LQ, Gaussian (least squares).
- Average of ROI (finds summed intensity of spots)

Picasso uses `Gpufit <https://github.com/gpufit/Gpufit>`_ for fitting on CUDA-capable GPUs (see `GPU fitting`_ below). On Windows the pre-compiled library (``Gpufit.dll``) is vendored into Picasso (``picasso/ext/pygpufit/``) and works automatically — no extra install step. On Linux there is no pre-compiled binary; you have to build ``libGpufit.so`` yourself and drop it next to the Windows DLL (see `GPU fitting on Linux`_ below). When no GPU library is available, the GPU fitting option simply does not appear and Picasso uses the accessible CPU algorithms.

**Please note:** Picasso Localize supports file formats:

- ``.ome.tif`` and plain ``.tif`` / ``.tiff`` image stacks,
- MicroManager "separate image files" acquisitions, saved as one ``img_*.tif`` per frame in a folder (see ``File`` > ``Open MicroManager image folder`` below),
- ``NDTiffStack`` with extension ``.tif``,
- BigTIFF, with extensions ``.tif``, ``.btf``, ``.tf8`` or ``.tf2``,
- Zeiss ``.lsm``,
- Zeiss ``.czi`` (requires ``pip install picassosr[czi]``, Python ≥ 3.12),
- Leica ``.lif`` (requires ``pip install picassosr[lif]``, Python ≥ 3.12),
- ``.raw``,
- ``.ims`` (supported only on Windows),
- ``.nd2``,
- ``.stk``.

TIFF-family files (``.tif``, ``.tiff``, ``.ome.tif``, ``.btf``, ``.tf8``, ``.tf2``, ``.lsm``) are read via the `tifffile <https://github.com/cgohlke/tifffile>`_ library. **Picasso expects grayscale image stacks with one frame per TIFF page; multi-channel, RGB or tiled whole-slide TIFF variants are not supported.**

Zeiss ``.czi`` and Leica ``.lif`` movies are read via the optional `czifile <https://github.com/cgohlke/czifile>`_ and `liffile <https://github.com/cgohlke/liffile>`_ libraries, installed with the ``czi`` / ``lif`` extras (e.g. ``pip install picassosr[czi,lif]``; both require Python ≥ 3.12). These files are reduced to a single-channel ``(frames, height, width)`` movie: when a file contains more than one channel a dialog asks which channel to load (a ``.lif`` file may also contain several acquisitions, in which case the one with the most frames is used). We are open to feature requests regarding support for other file formats, please visit our `GitHub page <https://github.com/jungmannlab/picasso>`_.

GPU fitting
-----------

Picasso can run several of its fitting algorithms on a CUDA-capable NVIDIA GPU via `Gpufit <https://github.com/gpufit/Gpufit>`_, a CUDA Levenberg-Marquardt library that serves as Picasso's GPU fitting backend. Picasso loads it through a small Python binding in ``picasso/ext/pygpufit/``, which expects a compiled Gpufit library next to it:

- ``Gpufit.dll`` on Windows — **shipped with Picasso**, so GPU fitting works out of the box.
- ``libGpufit.so`` on Linux — **not shipped**, because the binary depends on your CUDA toolkit and GPU. You have to compile it yourself (see `GPU fitting on Linux`_), and copy it into ``picasso/ext/pygpufit/``.

When the library is present and a CUDA GPU is detected, the GPU fitting option becomes available in the ``Parameters`` dialog (Picasso checks ``gpufit.cuda_available()`` at startup). Otherwise the option stays hidden and Picasso uses the equivalent CPU implementation, so results are the same — only slower. Using the GPU is entirely optional and only worthwhile if you have an NVIDIA (CUDA-capable) GPU.

GPU fitting on Linux
~~~~~~~~~~~~~~~~~~~~~

The remainder of this section explains how to build ``libGpufit.so`` so that GPU fitting becomes available on Linux. (On Windows nothing needs to be done.)

Prerequisites
^^^^^^^^^^^^^

- An NVIDIA GPU with the matching `CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`_ installed.
- ``CMake`` 3.11 or later.
- A C/C++ compiler (GCC). CUDA only supports GCC up to a certain version; if ``make`` later complains *"unsupported GNU version! gcc versions later than X are not supported"*, install an older GCC and point CMake at it (see below).
- ``git``.

Building ``libGpufit.so``
^^^^^^^^^^^^^^^^^^^^^^^^^

From a terminal::

   git clone https://github.com/gpufit/Gpufit.git Gpufit
   mkdir Gpufit-build
   cd Gpufit-build
   cmake -DCMAKE_BUILD_TYPE=RELEASE ../Gpufit
   make

If ``make`` aborts with an *"unsupported GNU version"* error, your CUDA toolkit needs an older GCC. Install one (e.g. ``gcc-5``) and pass it to CMake::

   cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_COMPILER=gcc-5 ../Gpufit

After a successful build, ``libGpufit.so`` is created inside the build directory (under ``Gpufit-build/Gpufit/``).

Installing the library into Picasso
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the freshly built ``libGpufit.so`` into Picasso's ``picasso/ext/pygpufit/`` folder (the same folder that already contains ``Gpufit.dll`` and ``gpufit.py``)::

   cp Gpufit-build/Gpufit/libGpufit.so /path/to/picasso/picasso/ext/pygpufit/

To locate that folder for a ``pip``-installed Picasso, run ``pip show picassosr`` and look at the ``Location:`` line; the target is ``<Location>/picasso/ext/pygpufit/``. For a cloned repository it is simply ``picasso/ext/pygpufit/`` inside your Picasso folder.

Restart Picasso: Localize. If the library loaded and CUDA is available, the GPU fitting option appears in the ``Parameters`` dialog next to the supported fit methods, as described in `GPU fitting`_ above.

Identification and fitting of single-molecule spots
---------------------------------------------------

1. In ``Picasso: Localize``, open a movie file by dragging the file into the window or by selecting ``File`` > ``Open movie``. If the movie is split into multiple μManager .tif files, open only the first file. Picasso will automatically detect the remaining files according to their file names. Similarly, for consecutive .stk files (e.g. ``name_001.stk``, ``name_002.stk``, …), open the first file of the desired range and Picasso will automatically include all subsequent files with a higher numeric suffix. When opening a .raw file, a dialog will appear for file specifications. When opening an IMS file it should be displayed immediately in the localize window. When opening an IMS file with multiple channels, a dialog window will appear allowing you to select the channel that should be loaded. You can navigate through the file using the arrow keys on your keyboard. The current frame is displayed in the lower right corner.
2. Adjust the image contrast (select ``View`` > ``Contrast``) so that the single-molecule spots are clearly visible.
3. To adjust spot identification and fit parameters, open the ``Parameters`` dialog (select ``Analyze`` > ``Parameters``).
4. In the ``Identification`` group, set the ``Box side length`` to the rounded integer value of 6 × σ + 1, where σ is the standard deviation of the PSF. In an optimized microscope setup, σ is one pixel, and the respective ``Box side length`` should be set to 7. The value of ``Min. net gradient`` specifies a minimum threshold above which spots should be considered for fitting. The net gradient value of a spot is roughly proportional to its intensity, independent of its local background. By checking ``Preview``, the spots identified with the current settings will be marked in the displayed frame. Adjust ``Min. net gradient`` to a value at which only spots are detected (no background).
5. (Optional) Restrict the analysis to one or more regions of interest (ROIs) instead of the whole frame; see *Regions of interest (ROIs)* below.
6. In the ``Photon conversion`` group, adjust ``EM Gain``, ``Baseline``, ``Sensitivity`` and ``Quantum Efficiency`` according to your camera specifications and the experimental conditions. Set ``EM Gain`` to 1 for conventional output amplification. ``Baseline`` is the average dark camera count. ``Sensitivity`` is the conversion factor (electrons per analog-to-digital (A/D) count). ``Quantum Efficiency`` is not used since version 0.6.0 and is kept for backward compatibility only. These parameters are critical to converting camera counts to photons correctly. The quality of the upcoming maximum likelihood fit strongly depends on a Poisson photon noise model, and thus on the absolute photon count. For simulated data, generated with ``Picasso: Simulate``, set the parameters as follows: ``EM Gain`` = 1, ``Baseline`` = 0, ``Sensitivity`` = 1.
7. From the menu bar, select ``Analyze`` > ``Localize (Identify & Fit)`` to start spot identification and fitting in all movie frames. The status of this computation is displayed in the window's status bar. After completion, the fit results will be saved in a new file in the same folder as the movie, in which the filename is the base name of the movie file with the extension ``_locs.hdf5``. Furthermore, information about the movie and analysis procedure will be saved in an accompanying file with the extension ``_locs.yaml``; this file can be inspected using a text editor.

Regions of interest (ROIs)
--------------------------

By default, Picasso analyzes the whole frame. If you are only interested in certain parts of the movie, you can restrict the analysis to one or more rectangular regions of interest (ROIs). Spots outside the ROIs are ignored, which also speeds up the analysis. There are two ways to work with ROIs:

- **With the mouse, directly on the image.** Drag a rectangle with the left mouse button to add a ROI; repeat to add as many as you like. To remove a ROI, double-click inside it. ROIs are outlined in blue, and the one currently selected is highlighted in cyan.

- **Numerically, in the Parameters dialog.** Open ``Analyze`` > ``Parameters``. The ``ROIs`` field in the ``Identification`` group summarizes the current selection:

  - empty (``Whole frame``) means the entire frame is analyzed,
  - a single ROI is shown as its four coordinates ``y_min, x_min, y_max, x_max`` (in camera pixels), which you can edit directly in the field,
  - several ROIs are shown as a count (e.g. ``3 ROIs``).

  Click ``Edit ROIs...`` to open a small dialog where you can add, edit, remove, or clear all ROIs in a table.

To go back to analyzing the whole frame, simply remove all ROIs (double-click them, empty the single-ROI field, or use ``Clear`` in the ``Edit ROIs...`` dialog). If ROIs overlap, Picasso automatically trims them so that no spot is detected twice, so you do not need to draw them precisely. As with the rest of the identification settings, turn on ``Preview`` to check which spots fall inside your ROIs before running the full analysis.

Extra features
--------------

- ``File`` > ``Open one multichannel movie``: Opens a single multichannel file (``.ims``, ``.czi``, ``.lif`` or ``.nd2``) and loads **every** channel at once, one per channel, rather than prompting for a single channel to load.
- ``File`` > ``Open channels from several movies``: Opens several separate movie files and loads each as one channel. The channel name is taken from the file's metadata where available, otherwise from the file name.
- ``File`` > ``Open MicroManager image folder``: Opens a MicroManager acquisition that was saved as **separate image files** (one single-page ``img_*.tif`` per frame in a folder, e.g. ``img_channel000_position000_time000000000_z000.tif`` in MicroManager 2.0 or ``img_000000000_Default_000.tif`` in MicroManager 1.4), rather than as a single multi-page stack. Select the acquisition folder and Picasso assembles the whole sequence into one movie, ordered by frame index. Channel, position and z are held fixed at the first frame's values, so a multi-channel or multi-position acquisition is **not** interleaved into a single movie. Only the first frame is read when the movie is opened (the rest are read on demand during localization), so even acquisitions of tens of thousands of files open quickly. You can also reach the same result through ``File`` > ``Open movie`` by selecting any one ``img_*.tif`` file in the folder — Picasso detects the remaining frames automatically, exactly as it does for split μManager stacks.
- ``File`` > ``Save identifications``: Saves the current set of identifications (frame, x, y, net gradient and identification id, where applicable) to an HDF5 file with a companion YAML metadata file. By default the suggested filename is ``<movie_base>_identifications.hdf5``. The accompanying YAML stores the original movie metadata together with the ``Box Size`` and ``Min. Net Gradient`` used at the time of saving, so the parameters can be restored when the identifications are loaded again.
- ``File`` > ``Load identifications``: Loads identifications previously saved with ``Save identifications``. The identifications are clipped to the current movie's bounds (using the current ``Box Size``) and the identification parameters stored in the YAML sidecar (``Box Size``, ``Min. Net Gradient``) are restored. *As with the other identification loading actions, changing any identification parameter (box size, min. net gradient, etc.) will reset the loaded identifications, and ``Analyze`` > ``Fit`` should be used (rather than ``Localize (Identify & Fit)``) to fit them without resetting.*
- ``File`` > ``Load picks as identifications``: Allows the user to load circular picks (from Picasso Render) as identifications. Additionally, the drift correction file (.txt) can be loaded to adjust the positions of the identifications throughout acquisition. The current box size will be used to make the identification, however, min. net gradient will **not** be applied to the identifications. *Note that changing any of the identification parameters (box size, min. net gradient, etc) will reset the loaded identifications. Furthermore, use ``Analyze`` > ``Fit``, rather than ``Analyze`` > ``Localize (Identify & Fit)``, to fit the loaded identifications without reseting them.*
- ``File`` > ``Load locs as identifications``: Similar to loading picks as identifications (see above) but uses localizations as input. The user is asked to provide the number of frames around localizations to be used for the identifications, i.e., how many frames before and after the frame of the localization should be included in the identifications. For each localization, 2 * n_frames + 1 identifications will be assigned, thus if localizations are close together the identifications may overlap. *Note that changing any of the identification parameters (box size, min. net gradient, etc) will reset the loaded identifications. Furthermore, use ``Analyze`` > ``Fit``, rather than ``Analyze`` > ``Localize (Identify & Fit)``, to fit the loaded identifications without reseting them.*
- ``File`` > ``Save spots``: Cuts out and saves the identified spots (NxBxB array, with N spots and B being the box side length). The spots can be saved as a .npy file or as a .tif file.

When more than one channel is loaded (by either of the two actions above), a channel selector appears below the image so you can switch between channels; identification, fitting and saving then operate on the currently active channel.

Loading runs in the background, so the window stays responsive while the files are read, and a progress dialog with a ``Cancel`` button is shown. Cancelling stops before the next file begins (a file already being read is finished first). This also applies to opening a single movie. Because the load no longer blocks the interface, large or multi-file datasets can be opened without freezing Picasso.

Camera Config
-------------

Picasso can remember default cameras and will use saved camera parameters. In order to use camera configs, create a file named ``config.yaml`` in the ``picasso`` folder. See below on how to locate it.

To start with a template, modify ``config_template.yaml`` that can be found in the folder by default. Picasso will compare the entries with Micro-Manager-Metadata and match the sensitivity values. If no matching entries can be found (e.g., if the file was not created with Micro-Manager) the config file will still be used to create a dropdown menu to select the different categories. The camera config can also be used to define a default camera that will always be used. Indentions are used for definitions.

One click installer (Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you downloaded an .exe Picasso file from the `release page <https://github.com/jungmannlab/picasso/releases>`_:

- Navigate to the installation folder, by default, it's ``C:/Picasso``. *Before version 0.8.3, the default location was* ``C:/Program Files/Picasso``.
- Go to the folder ``_internal/picasso``. *Before version 0.9.6, the folder was simply* ``picasso``.
- Add your config file there.

One click installer (macOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you downloaded a .dmg Picasso file from the `release page <https://github.com/jungmannlab/picasso/releases>`_:

- Navigate to your Applications folder and right-click on the picasso app, then select "Show Package Contents".
- Go to the folder ``Contents/Frameworks/picasso``.
- Add your config file there.

PyPI
~~~~
If you installed Picasso using ``pip install picassosr``:

- Activate your conda environment where ``picassosr`` is installed by typing ``conda activate YOUR_ENVIRONMENT``.
- To find the location of the package, type ``pip show picassosr`` and look for the line starting with ``Location:``.
- Navigate to this location and go to ``picasso``.
- Add your config file there.

GitHub
~~~~~~
If you cloned the GitHub repository, you can add plugins by following these steps:
- Find the directory where you cloned the GitHub repository with Picasso.
- Go to ``picasso/picasso/``.
- Copy the config file to this folder.

Example: Default Camera
~~~~~~~~~~~~~~~~~~~~~~~

::

   Cameras:
     Camera1:
       Baseline: 100
       Sensitivity: 0.5
       Quantum Efficiency: 1.0

If there is only one camera entry, picasso will create a dropdown menu that has always selected this camera. 

Gain
^^^^
If the string ``Gain Property Name`` can be found in the config, picasso will search for a value for this key in the Micro-Manager metadata and match if found.

Sensitivity
^^^^^^^^^^^

If the string ``Sensitivity Categories`` can be found in the config, picasso will create a dropdown menu for each entry, and if the property can be located in the Micro-Manager Metadata, it will be automatically set.

::

   Cameras:
     Camera1:
       Baseline: 100
       Quantum Efficiency:
         525: 0.5
       Sensitivity Categories:
         - PixelReadoutRate
         - Sensitivity/DynamicRange
       Sensitivity:
         540 MHz - fastest readout:
           12-bit (high well capacity): 7.18
           12-bit (low noise): 0.29
           16-bit (low noise & high well capacity): 0.46
         200 MHz - lowest noise:
           12-bit (high well capacity): 7.0
           12-bit (low noise): 0.26
           16-bit (low noise & high well capacity): 0.45

Here, two Sensitivity Categories are given ``PixelReadoutRate`` and ``Sensitivity/DynamicRange``. In the upper dropdown menu, one now will be able to choose from ``540 MHz - fastest readout`` and
``200 MHz - lowest noise``. Within 540 MHz it will be ``12-bit (high well capacity): 7.18``, ``12-bit (low noise): 0.29`` and ``16-bit (low noise & high well capacity): 0.46``. Accordingly for the 200 MHz entry. The dropdown menus can be further nested, e.g., when considering Gain modes:

::

       Sensitivity:
         Electron Multiplying:
           17.000 MHz:
             Gain 1: 15.9
             Gain 2: 9.34
             Gain 3: 5.32

Quantum Efficiency
^^^^^^^^^^^^^^^^^^

This feature is not used since Picasso 0.6.0. It is kept for backward compatibility only.

Several Cameras
^^^^^^^^^^^^^^^

::

   Cameras:
     Camera1:
     Camera2:
     Camera3:

Once there are several cameras present, Picasso will select the camera who's name matches the Micro-Manager Metadata. If no camera is found, the first one is automatically selected. In the dropdown menu, the configured cameras are displayed in alphabetical order.

Camera Priorities
^^^^^^^^^^^^^^^^^

::

   CameraPriority:
      - Camera3
      - Camera1

If many cameras are configured, the dropdown can become cluttered. For that reason, the config can additionally include a "CameraPriority" field. It describes a list of camera names which must match names in the "Cameras" field. The listed cameras are then displayed on top of the dropdown menu while the non-listed cameras are shown below in alphabetical order.

3D-Calibration
--------------

Theory
~~~~~~

3D Calibration is performed by an adapted version of `Huang et al., 2008 <https://www.ncbi.nlm.nih.gov/pubmed/18174397/>`_.


Calibrating z
~~~~~~~~~~~~~

After entering the step size, picasso will calculate the mean and the variance for sigma_x and sigma_y for each z position. Localizations that are not within one standard deviation are discarded. A six-degree polynomial is fitted to the mean values of x and y.

-  mean_sx = cx[6]z0 + cx[5]z1 .. + cx[0]z6
-  mean_sy = cy[6]z0 + cy[5]z1 .. + cy[0]z6

The calibration coefficients are stored in the YAML file and contain the parameters of cx and cy. The first entry being c[0], the last being c[6].

Fitting z
~~~~~~~~~

For each localization, sigma_x and sigma_y is determined. Similar to the Science paper, the following equation is used to minimize the Distance D:  ``D = (sx0.5 - wx0.5)^2 + (sy0.5 - wy0.5)^2`` with w being ``c[6]z0 +
c[5]z1 .. + c[0]z6``.

Incorporating calibrations in config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The calibration depends on the microscope, camera, and emission wavelength used. It can become tedious to navigate to and select the correct calibration yaml file. Therefore, the config file can include a field to map camera and emission wavelength to path of the z calibration yaml file:

::

   z-calibrations:
      Camera1:
         525: /path/to/Camera1-GFP-zcalibration.yaml
         595: /path/to/Camera1-Cy3B-zcalibration.yaml

If the camera names and emission wavelengths match the settings in Micromanager, the correct z-calibration is automatically loaded. In any case an alternative calibration yaml file can be loaded by button.
