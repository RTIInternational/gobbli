#!/bin/bash

# Run the various processes needed for CI.
# Pass additional script arguments to py.test.

set -e

isort -rc --check-only ./gobbli
black ./gobbli
mypy ./gobbli --ignore-missing-imports
flake8 ./gobbli --config setup.cfg
py.test -vs $@ ./gobbli
