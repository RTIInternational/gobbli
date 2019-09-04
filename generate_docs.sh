#!/bin/bash

set -e

rm -f docs/auto/*
sphinx-apidoc --no-toc --separate -o docs/auto gobbli 'gobbli/model/*/src/'
cd docs
make html
