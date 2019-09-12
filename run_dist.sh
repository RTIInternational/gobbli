#!/bin/bash

function usage() {
    echo "Usage: $0 [test|live]"
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

mode="$1"

if [[ "$mode" != "test" && "$mode" != "live" ]]; then
    usage
    exit 1
fi

rm -r ./dist/

python setup.py sdist bdist_wheel

if [[ "$mode" == "test" ]]; then
    python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
elif [[ "$mode" == "live" ]]; then
    python -m twine upload  dist/*
fi
