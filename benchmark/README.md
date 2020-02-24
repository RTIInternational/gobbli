# gobbli Benchmarks

This directory contains benchmarking code and output for various aspects of gobbli model performance.

To run the benchmarks (note -- this may take several days depending on available computing resources):

    ./run_benchmarks.sh

To run with GPU support enabled:

    export GOBBLI_USE_GPU=1
    ./run_benchmarks.sh
    
Use `--help` to see additional arguments in case you want to debug individual benchmarks, force re-running, etc.

    ./run_benchmarks.sh --help
