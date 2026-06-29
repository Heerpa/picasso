call DEL /F/Q/S build > NUL
call DEL /F/Q/S dist > NUL
call RMDIR /Q/S build
call RMDIR /Q/S dist

call cd %~dp0\..\..

call conda create -n picasso_installer_gpu python=3.14.4 -y
call conda activate picasso_installer_gpu
call pip install build
call python -m build

for /f %%i in ('python -c "exec(open('picasso/version.py').read()); print(__version__)"') do set PICASSO_VERSION=%%i
REM installer_gpu adds numba-cuda[cu12], which pulls the nvidia-*-cu12 CUDA
REM runtime wheels needed for GPU features such as COMET.
call pip install "dist/picassosr-%PICASSO_VERSION%-py3-none-any.whl[installer_gpu]"
call cd release/one_click_windows_gui

REM GPU build differs from the CPU build by the extra flags below:
REM   --collect-all numba_cuda : the CUDA target package for numba
REM   --collect-all nvidia     : the bundled CUDA runtime libraries (cu12)
REM   --collect-all cuda       : NVIDIA's official CUDA binding (cuda.bindings
REM                              and cuda.core). numba-cuda >= 0.24 uses this
REM                              binding to talk to the driver; without it
REM                              numba.cuda.is_available() fails and the app
REM                              silently reports "no GPU".
REM   --copy-metadata numba-cuda / cuda-bindings / cuda-core : these packages
REM                              are discovered via importlib metadata, so the
REM                              dist-info must ship in the frozen app.
REM   --additional-hooks-dir ../pyinstaller/hooks : cuda.core implements its CUDA
REM                              runtime as compiled Cython extensions under a
REM                              cu12/cu13 subpackage it merges in at import time
REM                              via __path__; --collect-all drops those .pyd, so
REM                              hook-cuda.core.py copies them to their real path.
REM Note: numba-cuda redirects "numba.cuda" to its own target via a
REM site-packages .pth file that PyInstaller does not run, so the frozen app
REM would keep Numba's built-in CUDA stub and GPU features such as COMET would
REM vanish. picasso/_numba_cuda_compat.py reinstalls that redirect at import
REM time (shipped via --collect-all picasso), so no extra PyInstaller flag is
REM needed here.
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picasso ^
    --icon "../logos/localize.ico" ^
    --noconfirm
call pyinstaller "../pyinstaller/picasso_pyinstaller.py" ^
    --onedir ^
    --windowed ^
    --collect-all picasso ^
    --collect-all PyImarisWriter ^
    --collect-all streamlit ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all numba_cuda ^
    --collect-all nvidia ^
    --collect-all cuda ^
    --additional-hooks-dir "../pyinstaller/hooks" ^
    --copy-metadata numba-cuda ^
    --copy-metadata cuda-bindings ^
    --copy-metadata cuda-core ^
    --collect-submodules matplotlib.backends ^
    --copy-metadata streamlit ^
    --copy-metadata imageio ^
    --name picassow ^
    --icon "../logos/localize.ico" ^
    --noconfirm

call DEL /F/Q picasso.spec
call DEL /F/Q picassow.spec

call conda deactivate
call conda env remove -n picasso_installer_gpu -y

copy dist\picassow\picassow.exe dist\picasso\picassow.exe
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAPP_VERSION=%PICASSO_VERSION% /DVARIANT=-GPU picasso_innoinstaller.iss
