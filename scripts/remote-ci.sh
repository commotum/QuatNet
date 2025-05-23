#!/usr/bin/env bash

# Run CI build and tests on a remote server via SSH.
# Requires the following environment variables:
#   CI_SSH_HOST - Hostname or IP of the remote server
#   CI_SSH_USER - SSH user to connect as
#   CI_REPO_PATH - Path to the repository on the remote server

set -e

if [[ -z "$CI_SSH_HOST" || -z "$CI_SSH_USER" || -z "$CI_REPO_PATH" ]]; then
    echo "CI_SSH_HOST, CI_SSH_USER, and CI_REPO_PATH must be set" >&2
    exit 1
fi

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

