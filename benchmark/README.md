# gobbli Benchmarks

This directory contains benchmarking code that can be run interactively in a Docker container via Jupyter Lab.

To run the Jupyter server:

    ./run_jupyter.sh interactive
    
To run with GPU support enabled:

    export GOBBLI_USE_GPU=1
    ./run_jupyter.sh interactive
    
You can access and rerun individual notebooks.  You may obtain slightly different accuracy numbers if you rerun the models.

If you want to regenerate the benchmarks on a server, you can use headless mode:

    ./run_jupyter.sh headless <notebook name>

If you want to rerun all notebooks in headless mode (after a functionality change or adding a new model to a large group of them), you can use `run_all_benchmarks.sh`:

    ./run_all_benchmarks.sh
    
**NOTE:** Some of the benchmarks are very time-consuming and may take days to run.
