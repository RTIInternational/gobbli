#!/bin/bash

# Run all benchmarks that haven't been run.
# NOTE: This may take several days depending on your available resources.

image_name="gobbli-benchmark"

if [[ -n "$GOBBLI_USE_GPU" ]]; then
    image_name="${image_name}-gpu"
    echo "GPU enabled."
else
    echo "GPU disabled; running on CPU."
fi

# Set working directory so the container starts in our working directory
# Otherwise it starts in the repository root
docker-compose run --rm "$image_name" $@
