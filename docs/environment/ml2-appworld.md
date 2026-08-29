# AppWorld setup and verification on `ml2`

This runbook reproduces the AppWorld checkout recorded in the [AppWorld `ml2` manifest](../../data/manifests/appworld-ml2.yaml). It deliberately contains no machine-specific shared path, proxy endpoint, credential, decrypted protected content, or full command log.

## Runtime-only network configuration

Store all repositories, environments, LFS objects, and downloaded AppWorld data below `${TOOLSHIFT_SHARED_ROOT}`. The operator supplies that variable in the shell or scheduler environment; do not persist its value in this repository.

For external GitHub and Python-package downloads, ask the operator to start the operator-provided `ml2-proxy` and inject the required proxy variables into the download shell at runtime. Never print, log, or write the proxy values to a dotfile, Git configuration, command transcript, or scheduler script committed to Git.

ModelScope and other approved domestic mirrors must run without the external proxy. Clear `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and their lowercase forms before using those sources.

## Pinned checkout and isolated Git LFS storage

Run all code blocks below in one shell session. Run the networked commands only in the operator-configured external-download shell:

```bash
set -euo pipefail
: "${TOOLSHIFT_SHARED_ROOT:?TOOLSHIFT_SHARED_ROOT must be set and non-empty}"
case "${TOOLSHIFT_SHARED_ROOT}" in
  /*) ;;
  *)
    echo "TOOLSHIFT_SHARED_ROOT must be an absolute path" >&2
    exit 1
    ;;
esac
TOOLSHIFT_SHARED_ROOT="$(realpath -m -- "${TOOLSHIFT_SHARED_ROOT}")"
if [ "${TOOLSHIFT_SHARED_ROOT}" = "/" ]; then
  echo "TOOLSHIFT_SHARED_ROOT must not resolve to filesystem root" >&2
  exit 1
fi
readonly TOOLSHIFT_SHARED_ROOT
mkdir -p "${TOOLSHIFT_SHARED_ROOT}/repos" "${TOOLSHIFT_SHARED_ROOT}/git-lfs"
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://github.com/StonyBrookNLP/appworld.git \
  "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" config --local \
  lfs.storage "${TOOLSHIFT_SHARED_ROOT}/git-lfs/appworld"
git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" lfs install --local
GIT_LFS_SKIP_SMUDGE=1 git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" \
  checkout --detach a072b7a86e7c1d5b1d7175659d750ebb9b79f10a
git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" lfs pull
git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" rev-parse HEAD
git lfs version
```

Confirm that the checkout is detached at the manifest commit and that Git LFS is version `3.8.0`. Keep LFS storage repository-local and under the shared root; do not install a global LFS configuration for this run.

## Python 3.11 environment and AppWorld data

Create an isolated `uv` environment using the verified Python runtime, then install from the pinned checkout:

```bash
test "$(uv --version)" = "uv 0.11.8"
mkdir -p "${TOOLSHIFT_SHARED_ROOT}/venvs"
uv venv --python 3.11.15 "${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0"
uv pip install \
  --python "${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0/bin/python" \
  -e "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
cd "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
"${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0/bin/appworld" \
  download data --version 0.2.0 --mode minimal
"${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0/bin/appworld" install --repo
```

This verification installation uses upstream editable dependency resolution. The project training dependency lock is not complete, and no lock checksum is recorded.

The two data version files must both report `0.2.0`. The path/size inventory and encrypted bundle hashes are recorded in the manifest; the path/size inventory is a structural reproducibility summary, not a content-integrity checksum.

## Official verification

Before starting AppWorld verification, remove every external proxy variable from the verification shell:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

This step is mandatory because AppWorld starts localhost services during verification. If a proxy remains active, local health checks can be routed through it and fail even when the local service is healthy.

Run the official checks from the detached checkout with the isolated environment:

```bash
cd "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
"${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0/bin/appworld" verify tests
"${TOOLSHIFT_SHARED_ROOT}/venvs/appworld-0.2.0/bin/appworld" \
  verify tasks --num-processes 4
```

Record only exit codes, elapsed seconds, suite counts, immutable revisions, and checksums in the repository manifest. Keep full logs and any protected data outside Git under `${TOOLSHIFT_SHARED_ROOT}`.
