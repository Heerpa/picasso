from multiprocessing import freeze_support


if __name__ == "__main__":
    # The windowed build (picassow.exe) has no console, so PyInstaller sets
    # sys.stdout/sys.stderr to None. Repair them before anything else runs:
    # this happens before freeze_support() so that the worker processes of
    # the process pools - which exit inside freeze_support() and never reach
    # main() - get usable streams too. Without it, tqdm raises inside every
    # worker and their tracebacks are lost.
    from picasso.diagnostics import ensure_std_streams

    ensure_std_streams()

    # This line is needed so that pyinstaller can handle multiprocessing
    freeze_support()
    from picasso.__main__ import main

    main()
