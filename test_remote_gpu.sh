#!/bin/bash

# Run tests on a GPU machine over SSH.  Assumes the remote user is a member of the
# Docker group (i.e. no sudo required for Docker commands), and that
# Docker/docker-compose/nvidia-docker are already installed on the remote server.

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <ssh_string> <remote_repo_dir> <visible_devices>"
    echo
    echo "    ssh_string: SSH connection string for the remote server."
    echo
    echo "    remote_repo_dir: Path to use as the repository root on the "
    echo "      remote server.  Files will be copied here."
    echo
    echo "    visible_devices: Value to use for the NVIDIA_VISIBLE_DEVICES environment "
    echo "      variable controlling which GPUs are made available to the container "
    echo "      for testing."
    exit 1
fi

ssh_string="$1"
remote_repo_dir="$2"
visible_gpus="$3"

if ssh "$ssh_string" "[[ -e $remote_repo_dir ]]"; then
    echo "Directory '$remote_repo_dir' already exists on the remote server;"
    echo "can't run tests pointing at an existing directory."
    exit 1
fi

rsync -raz \
      --exclude .git \
      --filter=':- .gitignore' \
      ./ "$ssh_string:$remote_repo_dir"

ssh "$ssh_string" "cd $remote_repo_dir/ci-gpu \
    && export NVIDIA_VISIBLE_DEVICES=$visible_gpus \
    && export PYTHON_VERSION=3.7 \
    && docker-compose build gobbli-ci-gpu \
    && docker-compose run --rm gobbli-ci-gpu"
