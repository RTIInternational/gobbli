#!/bin/bash

function usage() {
    echo "Usage: $0 <mode> [<nb_file>]"
    echo "  mode: 'headless' (run notebook on the command line via nbconvert). Requires a 'nb_file'."
    echo "        'interactive' (run notebook server)"
}

if [[ $# -ne 1 && $# -ne 2 ]]; then
    usage
    exit 1
fi

image_name="gobbli-benchmark"

if [[ -n "$GOBBLI_USE_GPU" ]]; then
    image_name="${image_name}-gpu"
    echo "GPU enabled."
else
    echo "GPU disabled; running on CPU."
fi


mode="$1"
if [[ "$mode" == "headless" ]]; then
    if [[ $# -ne 2 ]]; then
        usage
        exit 1
    fi
    nb_file="$2"

    # Set working directory so the container starts in our working directory
    # Otherwise it starts in the repository root
    docker-compose run -T -w "$(pwd)" --rm "$image_name" \
                   jupyter nbconvert \
                   --inplace \
                   --to notebook \
                   --log-level INFO \
                   --ExecutePreprocessor.timeout=-1 \
                   --execute "$nb_file"

elif [[ "$mode" == "interactive" ]]; then

    # Run Jupyter Lab in the Python container.
    docker-compose up "$image_name"
else
    echo "Unrecognized mode '$mode'"
    usage
    exit 1
fi
