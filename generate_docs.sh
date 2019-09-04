#!/bin/bash

set -e

rm -f docs/auto/*
cd docs
make html
