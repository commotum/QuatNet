#!/usr/bin/env bash

# Run CI build and tests on a remote server via SSH.
# Requires the following environment variables:
#   CI_SSH_HOST - Hostname or IP of the remote server
#   CI_SSH_USER - SSH user to connect as
#   CI_REPO_PATH - Path to the repository on the remote server

set -e

# Default configuration. Override by setting the environment variables before
# calling this script.
CI_SSH_HOST="${CI_SSH_HOST:-100.111.229.14}"
CI_SSH_USER="${CI_SSH_USER:-jake}"
CI_REPO_PATH="${CI_REPO_PATH:-~/Developer/QuatNet/commotum-quatnet}"

ssh "${CI_SSH_USER}@${CI_SSH_HOST}" bash -s <<EOF2
set -e
cd "${CI_REPO_PATH}"
rm -rf build
mkdir build
cd build
cmake .. && make -j\$(nproc)
python -m pytest -q
EOF2

exit_code=$?
exit $exit_code

