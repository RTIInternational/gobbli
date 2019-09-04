#!/bin/bash

# Run all benchmarks using headless mode.
# NOTE: This will probably take several days, depending on your computing resources.

for benchmark in *.ipynb; do
    echo "Starting $benchmark"
    ./run_jupyter.sh headless "./${benchmark}"
    echo "Exit code $? - finished $benchmark"
done
