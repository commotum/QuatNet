# Remote CI Agent

This repository uses a remote CI build and test system.

## Trigger
Run the remote CI whenever you see instructions like "run remote CI" or "build+test on server".

## Action
Execute `scripts/remote-ci.sh` locally. The script SSHs into the remote machine and performs the build and tests there.

## Environment variables
- `CI_SSH_HOST`: Hostname or IP address of the remote server.
- `CI_SSH_USER`: SSH user to log in as.
- `CI_REPO_PATH`: Path to this repository on the remote server.

## Example
```bash
export CI_SSH_HOST=100.111.229.14
export CI_SSH_USER=jake
export CI_REPO_PATH=~/Developer/QuatNet/commotum-quatnet
bash scripts/remote-ci.sh
```
