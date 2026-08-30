# AppWorld setup and verification on `ml2`

This runbook reproduces the AppWorld checkout recorded in the [AppWorld `ml2` manifest](../../data/manifests/appworld-ml2.yaml). It deliberately contains no machine-specific shared path, proxy endpoint, credential, decrypted protected content, or full command log.

## Fixed sources and runtime-only network configuration

The reproducibility boundary is the official AppWorld repository at
[`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`](https://github.com/StonyBrookNLP/appworld/tree/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a).
The pinned [`cli.py`](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/cli.py)
defines the `install`, `download data --root`, and `verify tests --root` commands; the pinned
[`path_store.py`](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/common/path_store.py)
defines how `APPWORLD_ROOT` isolates data and runtime output. The pinned
[`verify.py`](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/verify.py)
also shows that a repository installation reads its unpacked package tests below that root.

Store repositories, LFS objects, and downloaded AppWorld data below `${TOOLSHIFT_SHARED_ROOT}`.
Build and run the Python environment below the node-local `${TOOLSHIFT_NODE_LOCAL}` instead
of the shared filesystem. The operator supplies both variables in the shell or scheduler
environment; do not persist either value in this repository.

For external GitHub and Python-package downloads, ask the operator to start the
operator-provided `ml2-proxy` and inject the required proxy variables into the download shell
at runtime. Never print, log, or write the proxy values to a dotfile, Git configuration,
command transcript, or scheduler script committed to Git.

ModelScope and other approved domestic mirrors must run without the external proxy. Use the
fail-closed `clear_proxy_environment` helper defined below immediately before invoking any
approved domestic mirror. It removes and audits every environment name whose lowercase form
ends in `_proxy`, without printing either names or values.

## Pinned checkout and isolated Git LFS storage

Ensure the Git LFS executable is installed and available on `PATH` before cloning. Do not run
a global Git LFS install for this workflow: initialize it only in the cloned repository with
`lfs install --local`. Run all code blocks below in one Bash shell session, and run networked
commands only in the operator-configured external-download shell:

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
clear_proxy_environment() {
  local proxy_variable
  while IFS= read -r proxy_variable; do
    case "${proxy_variable,,}" in
      *_proxy)
        if ! unset -v "${proxy_variable}" 2>/dev/null; then
          echo "proxy environment cleanup failed" >&2
          return 1
        fi
        ;;
    esac
  done < <(compgen -e)
  while IFS= read -r proxy_variable; do
    case "${proxy_variable,,}" in
      *_proxy)
        echo "proxy environment cleanup failed" >&2
        return 1
        ;;
    esac
  done < <(compgen -e)
}
clear_git_environment() {
  local git_variable
  while IFS= read -r git_variable; do
    case "${git_variable}" in
      GIT_*)
        if ! unset -v "${git_variable}" 2>/dev/null; then
          echo "Git environment cleanup failed" >&2
          return 1
        fi
        ;;
    esac
  done < <(compgen -e)
  while IFS= read -r git_variable; do
    case "${git_variable}" in
      GIT_*)
        echo "Git environment cleanup failed" >&2
        return 1
        ;;
    esac
  done < <(compgen -e)
}
mkdir -p "${TOOLSHIFT_SHARED_ROOT}/repos" "${TOOLSHIFT_SHARED_ROOT}/git-lfs"
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://github.com/StonyBrookNLP/appworld.git \
  "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
chmod 0700 "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
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

Use node-local storage for the complete environment and cache so `uv` never has to perform
atomic environment persistence on the shared NFS mount. `--link-mode copy` keeps the
environment independent of cache entries. Create the dedicated private `APPWORLD_ROOT` under
umask `077`, make it mode `0700`, and keep it outside every Git worktree. It is deliberately
not the AppWorld source checkout.

```bash
set -euo pipefail
: "${TOOLSHIFT_NODE_LOCAL:?TOOLSHIFT_NODE_LOCAL must be set and non-empty}"
case "${TOOLSHIFT_NODE_LOCAL}" in
  /*) ;;
  *)
    echo "TOOLSHIFT_NODE_LOCAL must be an absolute path" >&2
    exit 1
    ;;
esac
TOOLSHIFT_NODE_LOCAL="$(realpath -m -- "${TOOLSHIFT_NODE_LOCAL}")"
if [ "${TOOLSHIFT_NODE_LOCAL}" = "/" ]; then
  echo "TOOLSHIFT_NODE_LOCAL must not resolve to filesystem root" >&2
  exit 1
fi
readonly TOOLSHIFT_NODE_LOCAL
umask 077
export APPWORLD_ROOT="${TOOLSHIFT_SHARED_ROOT}/private/appworld-m3a"
mkdir -p "${APPWORLD_ROOT}"
chmod 0700 "${APPWORLD_ROOT}"
test -d "${APPWORLD_ROOT}" && test ! -L "${APPWORLD_ROOT}" && test -O "${APPWORLD_ROOT}"
test "$(stat -c '%a' -- "${APPWORLD_ROOT}")" = "700"
if git -C "${APPWORLD_ROOT}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "APPWORLD_ROOT must be outside every Git worktree" >&2
  exit 1
fi
export APPWORLD_CACHE="${APPWORLD_ROOT}/.cache"
mkdir -p "${APPWORLD_CACHE}"
chmod 0700 "${APPWORLD_CACHE}"
test -d "${APPWORLD_CACHE}" && test ! -L "${APPWORLD_CACHE}" && test -O "${APPWORLD_CACHE}"
test "$(stat -c '%a' -- "${APPWORLD_CACHE}")" = "700"
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
unset APPWORLD_DB_ARGS APPWORLD_DATE_TIME LOAD_ON_STARTUP
export UV_CACHE_DIR="${TOOLSHIFT_NODE_LOCAL}/uv-cache"
export APPWORLD_ENV="${TOOLSHIFT_NODE_LOCAL}/venvs/appworld-0.2.0"
readonly APPWORLD_ENV
uv_runtime_version="$(uv --version)"
case "${uv_runtime_version}" in
  "uv 0.11.8" | "uv 0.11.8 "*) ;;
  *)
    echo "Unexpected uv version" >&2
    exit 1
    ;;
esac
unset uv_runtime_version
mkdir -p "${UV_CACHE_DIR}" "$(dirname -- "${APPWORLD_ENV}")"
test ! -e "${APPWORLD_ENV}"
uv venv --python 3.11.15 "${APPWORLD_ENV}"
uv pip install \
  --link-mode copy \
  --python "${APPWORLD_ENV}/bin/python" \
  "python-dotenv==1.2.2" \
  -e "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
cd "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
"${APPWORLD_ENV}/bin/appworld" install --repo
test ! -e "${APPWORLD_ROOT}/tests"
cp -a "${TOOLSHIFT_SHARED_ROOT}/repos/appworld/tests" "${APPWORLD_ROOT}/tests"
"${APPWORLD_ENV}/bin/appworld" \
  download data --version 0.2.0 --mode minimal --root "${APPWORLD_ROOT}"
test -d "${APPWORLD_ROOT}/data"
test ! -L "${APPWORLD_ROOT}/data"
test -O "${APPWORLD_ROOT}/data"
chmod 0700 "${APPWORLD_ROOT}/data"
test -d "${APPWORLD_ROOT}/data/base_dbs"
test ! -L "${APPWORLD_ROOT}/data/base_dbs"
test -O "${APPWORLD_ROOT}/data/base_dbs"
chmod 0700 "${APPWORLD_ROOT}/data/base_dbs"
test -f "${APPWORLD_ROOT}/data/version.txt"
test ! -L "${APPWORLD_ROOT}/data/version.txt"
test -O "${APPWORLD_ROOT}/data/version.txt"
chmod 0600 "${APPWORLD_ROOT}/data/version.txt"
test -f "${APPWORLD_ROOT}/data/base_dbs/version.txt"
test ! -L "${APPWORLD_ROOT}/data/base_dbs/version.txt"
test -O "${APPWORLD_ROOT}/data/base_dbs/version.txt"
chmod 0600 "${APPWORLD_ROOT}/data/base_dbs/version.txt"

APPWORLD_CHECKOUT="${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
readonly APPWORLD_CHECKOUT
test -d "${APPWORLD_CHECKOUT}/.git" && test ! -L "${APPWORLD_CHECKOUT}/.git"
clear_git_environment
set +e
"${APPWORLD_ENV}/bin/python" - "${APPWORLD_CHECKOUT}" <<'PY'
import ast
import hashlib
import io
import os
import pathlib
import stat
import subprocess
import sys
import zipfile

try:
    checkout = pathlib.Path(sys.argv[1]).resolve(strict=True)
    git_directory = checkout / ".git"
    git_stat = git_directory.lstat()
    if not stat.S_ISDIR(git_stat.st_mode) or stat.S_ISLNK(git_stat.st_mode):
        raise SystemExit(2)
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    prefix = [
        "git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        f"--git-dir={git_directory}",
        f"--work-tree={checkout}",
    ]

    def run(arguments, *, input_bytes=None, timeout=30):
        return subprocess.run(
            [*prefix, *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=input_bytes,
            env=environment,
            timeout=timeout,
        )

    def git_blob_sha1(payload):
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {len(payload)}\0".encode())
        digest.update(payload)
        return digest.hexdigest()

    def lfs_pointer(payload):
        lines = payload.splitlines(keepends=True)
        if (
            len(lines) != 3
            or lines[0] != b"version https://git-lfs.github.com/spec/v1\n"
            or not lines[1].startswith(b"oid sha256:")
            or not lines[1].endswith(b"\n")
            or not lines[2].startswith(b"size ")
            or not lines[2].endswith(b"\n")
        ):
            return None
        digest = lines[1][len(b"oid sha256:") : -1]
        size = lines[2][len(b"size ") : -1]
        if (
            len(digest) != 64
            or any(byte not in b"0123456789abcdef" for byte in digest)
            or not size
            or any(byte not in b"0123456789" for byte in size)
        ):
            raise SystemExit(4)
        return digest.decode("ascii"), int(size)

    binding = run(["rev-parse", "--show-toplevel", "--absolute-git-dir"])
    binding_lines = binding.stdout.decode("utf-8", errors="strict").splitlines()
    if (
        binding.returncode != 0
        or len(binding_lines) != 2
        or pathlib.Path(binding_lines[0]).resolve(strict=True) != checkout
        or pathlib.Path(binding_lines[1]).resolve(strict=True) != git_directory
    ):
        raise SystemExit(2)
    head = run(["rev-parse", "HEAD"])
    if (
        head.returncode != 0
        or head.stdout.strip() != b"a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
    ):
        raise SystemExit(2)
    detached = run(["symbolic-ref", "-q", "HEAD"])
    if detached.returncode != 1:
        raise SystemExit(2)
    object_format = run(["rev-parse", "--show-object-format"])
    if object_format.returncode != 0 or object_format.stdout.strip() != b"sha1":
        raise SystemExit(4)

    index_flags = run(["ls-files", "-v", "-z"])
    if index_flags.returncode != 0:
        raise SystemExit(4)
    for entry in index_flags.stdout.split(b"\0"):
        tag = entry[:1]
        if tag == b"S" or tag.islower():
            raise SystemExit(3)
    untracked = run(["ls-files", "--others", "-z", "--"])
    if untracked.returncode:
        raise SystemExit(4)
    untracked_paths = set()
    for raw_path in untracked.stdout.split(b"\0"):
        if not raw_path:
            continue
        parts = raw_path.split(b"/")
        if (
            raw_path.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in parts)
            or raw_path in untracked_paths
        ):
            raise SystemExit(3)
        untracked_paths.add(raw_path)
    if os.path.lexists(git_directory / "info" / "attributes"):
        raise SystemExit(3)
    if run(
        [
            "diff-index",
            "--cached",
            "--quiet",
            "--no-ext-diff",
            "--ignore-submodules=none",
            "HEAD",
            "--",
        ]
    ).returncode:
        raise SystemExit(3)

    tree = run(["ls-tree", "-r", "-z", "--full-tree", "HEAD"])
    if tree.returncode:
        raise SystemExit(4)
    entries = []
    paths = set()
    for record in tree.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        parts = raw_path.split(b"/")
        if (
            separator != b"\t"
            or len(fields) != 3
            or fields[0] not in (b"100644", b"100755", b"120000")
            or fields[1] != b"blob"
            or len(fields[2]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[2])
            or not raw_path
            or raw_path.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in parts)
            or raw_path in paths
        ):
            raise SystemExit(4)
        paths.add(raw_path)
        entries.append((fields[0], fields[2], raw_path))
    if not entries:
        raise SystemExit(4)

    object_ids = tuple(dict.fromkeys(oid for _, oid, _ in entries))
    batch = run(
        ["cat-file", "--batch"],
        input_bytes=b"".join(oid + b"\n" for oid in object_ids),
    )
    if batch.returncode:
        raise SystemExit(4)
    payloads = {}
    offset = 0
    for expected_oid in object_ids:
        header_end = batch.stdout.find(b"\n", offset)
        if header_end < 0:
            raise SystemExit(4)
        header = batch.stdout[offset:header_end].split(b" ")
        if len(header) != 3 or header[:2] != [expected_oid, b"blob"]:
            raise SystemExit(4)
        size = int(header[2])
        start = header_end + 1
        end = start + size
        payload = batch.stdout[start:end]
        if (
            end >= len(batch.stdout)
            or batch.stdout[end : end + 1] != b"\n"
            or git_blob_sha1(payload).encode("ascii") != expected_oid
        ):
            raise SystemExit(4)
        payloads[expected_oid] = payload
        offset = end + 1
    if offset != len(batch.stdout):
        raise SystemExit(4)

    head_payloads = {relative: payloads[oid] for _, oid, relative in entries}
    for mode, oid, relative in entries:
        path = checkout / os.fsdecode(relative)
        metadata = path.lstat()
        expected = payloads[oid]
        if mode == b"120000":
            if not stat.S_ISLNK(metadata.st_mode):
                raise SystemExit(3)
            if git_blob_sha1(os.fsencode(os.readlink(path))) != oid.decode("ascii"):
                raise SystemExit(3)
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or bool(metadata.st_mode & 0o111) != (mode == b"100755")
        ):
            raise SystemExit(3)
        pointer = lfs_pointer(expected)
        if pointer is None:
            expected_digest = oid.decode("ascii")
            expected_size = len(expected)
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {metadata.st_size}\0".encode())
        else:
            expected_digest, expected_size = pointer
            digest = hashlib.sha256()
        if metadata.st_size != expected_size:
            raise SystemExit(3)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise SystemExit(3)

    constants = ast.parse(
        head_payloads[b"src/appworld/common/constants.py"].decode(
            "utf-8", errors="strict"
        )
    )
    literals = {}
    for statement in constants.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in ("PASSWORD", "SALT")
        ):
            name = statement.targets[0].id
            if name in literals:
                raise SystemExit(4)
            literals[name] = ast.literal_eval(statement.value)
    bundle_phrase = literals.get("PASSWORD")
    kdf_salt = literals.get("SALT")
    if type(bundle_phrase) is not str or type(kdf_salt) is not bytes:
        raise SystemExit(4)

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    bundle_layout = (
        (b"src/appworld/.source/apps.bundle", b"src/appworld"),
        (b"src/appworld/.source/tests.bundle", b"tests"),
        (b"generate/.source/tasks.bundle", b"generate/tasks"),
        (b"generate/.source/data.bundle", b"generate"),
    )
    expected_untracked = {}
    for bundle_path, base_directory in bundle_layout:
        pointer = lfs_pointer(head_payloads[bundle_path])
        if pointer is None:
            raise SystemExit(4)
        expected_digest, expected_size = pointer
        encrypted = (checkout / os.fsdecode(bundle_path)).read_bytes()
        if (
            len(encrypted) != expected_size
            or hashlib.sha256(encrypted).hexdigest() != expected_digest
            or len(encrypted) < 17
        ):
            raise SystemExit(3)
        key = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=kdf_salt,
            iterations=100000,
            backend=default_backend(),
        ).derive(bundle_phrase.encode("utf-8"))
        decryptor = Cipher(
            algorithms.AES(key),
            modes.CFB(encrypted[:16]),
            backend=default_backend(),
        ).decryptor()
        archive_bytes = decryptor.update(encrypted[16:]) + decryptor.finalize()
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            for info in archive.infolist():
                raw_member = info.filename.encode("utf-8", errors="strict")
                parts = raw_member.split(b"/")
                if (
                    info.is_dir()
                    or not raw_member
                    or raw_member.startswith(b"/")
                    or b"\\" in raw_member
                    or any(part in (b"", b".", b"..") for part in parts)
                ):
                    raise SystemExit(4)
                target = base_directory + b"/" + raw_member
                if target in expected_untracked or target in head_payloads:
                    raise SystemExit(4)
                expected_untracked[target] = archive.read(info)
    if untracked_paths != set(expected_untracked):
        raise SystemExit(3)
    for raw_path, expected in expected_untracked.items():
        path = checkout / os.fsdecode(raw_path)
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or path.read_bytes() != expected
        ):
            raise SystemExit(3)
except Exception:
    raise SystemExit(4)
PY
checkout_audit_status=$?
set -e
case "${checkout_audit_status}" in
  0) ;;
  2)
    echo "Pinned AppWorld checkout binding validation failed" >&2
    exit 1
    ;;
  3)
    echo "Pinned AppWorld tracked checkout is dirty" >&2
    exit 1
    ;;
  *)
    echo "Pinned AppWorld checkout audit failed" >&2
    exit 1
    ;;
esac
```

This verification installation uses upstream editable dependency resolution except for an
exact `python-dotenv==1.2.2` pin. The pin makes `PYTHON_DOTENV_DISABLED=1` reproducible; Gate 0
also checks the installed version and probes the disable behavior before importing AppWorld.
The project training dependency lock is not complete, and no lock checksum is recorded.

The two data version files below the dedicated private `APPWORLD_ROOT` must both report
`0.2.0`. The path/size inventory and encrypted bundle hashes are recorded in the manifest;
the path/size inventory is a structural reproducibility summary, not a content-integrity
checksum. The audit decrypts the four already-verified LFS bundles in memory and requires the
ignored unpacked files to match their exact path set and bytes; any additional untracked file,
including caches or dotenv configuration, fails closed. It persists no protected content or
derived fingerprint. At this revision, `verify tests --root` expects the unpacked repository
tests below the selected root, so the commands copy that tree into the mode-`0700` private root
before verification. The protected test copy never enters ToolShift Git.

## Official verification

The proxy cleanup is mandatory because AppWorld starts localhost services during
verification. If a proxy remains active, local health checks can be routed through it and fail
even when the local service is healthy. Run only the official no-task test verification from
the detached checkout, explicitly bind it to the private root, and send all detailed output to
one mode-`0600` private log:

```bash
clear_proxy_environment
mkdir -p "${APPWORLD_ROOT}/operator-records"
chmod 0700 "${APPWORLD_ROOT}/operator-records"
test -d "${APPWORLD_ROOT}/operator-records"
test ! -L "${APPWORLD_ROOT}/operator-records"
test -O "${APPWORLD_ROOT}/operator-records"
test "$(stat -c '%a' -- "${APPWORLD_ROOT}/operator-records")" = "700"
VERIFY_LOG="${APPWORLD_ROOT}/operator-records/official-tests.log"
if [ -e "${VERIFY_LOG}" ] || [ -L "${VERIFY_LOG}" ]; then
  echo "Official AppWorld private verification log already exists" >&2
  exit 1
fi
: > "${VERIFY_LOG}"
chmod 0600 "${VERIFY_LOG}"
test -f "${VERIFY_LOG}" && test ! -L "${VERIFY_LOG}" && test -O "${VERIFY_LOG}"
test "$(stat -c '%a' -- "${VERIFY_LOG}")" = "600"

set +e
cd "${APPWORLD_CHECKOUT}"
"${APPWORLD_ENV}/bin/appworld" \
  verify tests --root "${APPWORLD_ROOT}" >"${VERIFY_LOG}" 2>&1
verify_status=$?
set -e
if [ "${verify_status}" -ne 0 ]; then
  echo "Official AppWorld test verification failed; inspect the private log" >&2
  exit 1
fi

if ! "${APPWORLD_ENV}/bin/python" - "${APPWORLD_ROOT}" "${VERIFY_LOG}" <<'PY'
import json
import os
import re
import stat
import sys

root_fd = records_fd = log_fd = temporary_fd = -1
temporary_name = ".official-tests-audit.tmp"
try:
    root_path = os.path.realpath(sys.argv[1])
    log_path = os.path.abspath(sys.argv[2])
    records_path = os.path.join(root_path, "operator-records")
    expected_log_path = os.path.join(records_path, "official-tests.log")
    if root_path == "/" or log_path != expected_log_path:
        raise ValueError

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root_path, directory_flags)
    records_fd = os.open("operator-records", directory_flags, dir_fd=root_fd)
    records_stat = os.fstat(records_fd)
    if records_stat.st_uid != os.geteuid() or stat.S_IMODE(records_stat.st_mode) != 0o700:
        raise ValueError

    log_name = "official-tests.log"
    log_fd = os.open(log_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=records_fd)
    log_stat = os.fstat(log_fd)
    if (
        not stat.S_ISREG(log_stat.st_mode)
        or log_stat.st_uid != os.geteuid()
        or stat.S_IMODE(log_stat.st_mode) != 0o600
    ):
        raise ValueError
    with os.fdopen(os.dup(log_fd), "r", encoding="utf-8", errors="replace") as handle:
        details = handle.read()
    summaries = [
        (int(passed), int(skipped or 0))
        for passed, skipped in re.findall(
            r"(?m)(\d+) passed(?:,\s*(\d+) skipped)?", details
        )
    ]
    expected_summaries = [(1652, 0), (76, 0), (110, 2)]
    if summaries != expected_summaries:
        raise ValueError

    audit = (
        json.dumps(
            {
                "exit_code": 0,
                "suite_summaries": [
                    {"passed": passed, "skipped": skipped}
                    for passed, skipped in summaries
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=records_fd,
    )
    os.write(temporary_fd, audit)
    os.fsync(temporary_fd)
    os.close(temporary_fd)
    temporary_fd = -1
    os.replace(
        temporary_name,
        "official-tests-audit.json",
        src_dir_fd=records_fd,
        dst_dir_fd=records_fd,
    )
    os.fsync(records_fd)
    os.unlink(log_name, dir_fd=records_fd)
    os.fsync(records_fd)
except Exception:
    if records_fd >= 0:
        try:
            os.unlink(temporary_name, dir_fd=records_fd)
        except OSError:
            pass
    raise SystemExit(1)
finally:
    for descriptor in (temporary_fd, log_fd, records_fd, root_fd):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
PY
then
  echo "Official AppWorld test verification audit failed; inspect the private log" >&2
  exit 1
fi
```

The private audit keeps only exit status and aggregate pass/skip counts. The detailed log is
deleted through its already-validated private directory descriptor only after all expected
aggregates pass audit. A failed command or audit retains the mode-`0600` log and emits only a
static public-shell error. The pinned official task verifier is not part of M3A: it opens both
train and dev and does not implement the one-worker selection protocol.

## Private, non-formal M3A integration smoke

M3A is private, non-formal M3A integration evidence. It searches only `train` in official
loader order and uses one worker; it never enumerates or opens `dev`, `test_normal`, or
`test_challenge`. A missing runtime, wrong pin, missing data, absence of an eligible L2 task,
contract failure, repeated-save failure, or cleanup failure is a hard failure.

Supply the ToolShift checkout at runtime, install it into the isolated environment, clear all
proxy variables, and place the aggregate-only result below the private root:

```bash
set -euo pipefail
: "${TOOLSHIFT_REPOSITORY_ROOT:?TOOLSHIFT_REPOSITORY_ROOT must be set and non-empty}"
case "${TOOLSHIFT_REPOSITORY_ROOT}" in
  /*) ;;
  *)
    echo "TOOLSHIFT_REPOSITORY_ROOT must be an absolute path" >&2
    exit 1
    ;;
esac
TOOLSHIFT_REPOSITORY_ROOT="$(realpath -m -- "${TOOLSHIFT_REPOSITORY_ROOT}")"
readonly TOOLSHIFT_REPOSITORY_ROOT
uv pip install \
  --link-mode copy \
  --python "${APPWORLD_ENV}/bin/python" \
  -e "${TOOLSHIFT_REPOSITORY_ROOT}[dev]"
clear_proxy_environment
mkdir -p "${APPWORLD_ROOT}/operator-records"
chmod 0700 "${APPWORLD_ROOT}/operator-records"
cd "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"
"${APPWORLD_ENV}/bin/python" \
  "${TOOLSHIFT_REPOSITORY_ROOT}/scripts/verify_appworld_gate0.py" \
  --output "${APPWORLD_ROOT}/operator-records/m3a-smoke.json"

TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE=1 \
  "${APPWORLD_ENV}/bin/pytest" \
  "${TOOLSHIFT_REPOSITORY_ROOT}/tests/smoke/test_appworld_gate0.py" -q
```

The JSON contains only the fixed aggregate allowlist and remains under mode `0600`. Actual M3A
records, screened/eligible/admitted counts, diagnostics, task-derived fingerprints, logs, and
payloads must never be added to Git or quoted in a public pull request. Publication requires
an AppWorld maintainer-approved encrypted workflow. A successful smoke remains
`gate_evaluable=false` and `formal_gate_passed=false`.

## Historical Gate 0a: deterministic oracle replay

The historical Gate 0a record predates M3A and does not set precedent for M3A publication or
privacy policy. It remains a separate environment-lifecycle record, not adapter-contract
evidence.

AppWorld resolves its data relative to the process working directory. The smoke and
formal verifier therefore **must run from the pinned AppWorld checkout**, even though
the ToolShift script and test live in a separate repository. The verifier rejects any
checkout whose `HEAD` is not the manifest revision.

Gate 0a treats reset as construction of a fresh AppWorld instance. Every repetition
uses a distinct `experiment_name`, runs the compiled train/dev oracle through native
AppWorld APIs, evaluates it, and closes the world through a context manager. Do not use
`load_state()` for this gate: the pinned revision can create a double time-freezer when
state loading is combined with repeated world lifecycles. A fresh-world reset avoids
that compatibility issue and tests the lifecycle used by later paired rollouts.
Each successful repetition must contain at least one native request with the pinned
`method`/`url`/`data` shape. The verifier hashes that private request list in memory and
persists only its aggregate consistency rate.

Supply the ToolShift checkout only at runtime. Clear every proxy-related variable
before either smoke or formal execution so localhost AppWorld services are never
routed away from the node:

```bash
set -euo pipefail
: "${TOOLSHIFT_SHARED_ROOT:?TOOLSHIFT_SHARED_ROOT must be set and non-empty}"
: "${TOOLSHIFT_REPOSITORY_ROOT:?TOOLSHIFT_REPOSITORY_ROOT must be set and non-empty}"
case "${TOOLSHIFT_REPOSITORY_ROOT}" in
  /*) ;;
  *)
    echo "TOOLSHIFT_REPOSITORY_ROOT must be an absolute path" >&2
    exit 1
    ;;
esac
TOOLSHIFT_REPOSITORY_ROOT="$(realpath -m -- "${TOOLSHIFT_REPOSITORY_ROOT}")"
readonly TOOLSHIFT_REPOSITORY_ROOT
mkdir -p "${TOOLSHIFT_SHARED_ROOT}/artifacts/gate0a"
uv pip install \
  --link-mode copy \
  --python "${APPWORLD_ENV}/bin/python" \
  -e "${TOOLSHIFT_REPOSITORY_ROOT}[dev]"
clear_proxy_environment
cd "${TOOLSHIFT_SHARED_ROOT}/repos/appworld"

TOOLSHIFT_RUN_APPWORLD_SMOKE=1 \
  "${APPWORLD_ENV}/bin/pytest" \
  "${TOOLSHIFT_REPOSITORY_ROOT}/tests/smoke/test_environment.py" -v

"${APPWORLD_ENV}/bin/python" \
  "${TOOLSHIFT_REPOSITORY_ROOT}/scripts/verify_appworld_replay.py" \
  --smoke \
  --output "${TOOLSHIFT_SHARED_ROOT}/artifacts/gate0a/smoke-summary.json"

"${APPWORLD_ENV}/bin/python" \
  "${TOOLSHIFT_REPOSITORY_ROOT}/scripts/verify_appworld_replay.py" \
  --output "${TOOLSHIFT_SHARED_ROOT}/artifacts/gate0a/formal-summary.json"
```

The formal command is intentionally fixed to all 90 train and 57 dev tasks, seed 100,
three fresh worlds per task, and one worker. A formal report is Gate-evaluable only
under that complete 147-task protocol. Its `task_consistency_rate` is a task-level AND
over initial state, final state, evaluator, private request trace, oracle success, and
zero execution failures; it must be at least 99%. The explicit smoke runs one train task
twice;
exit code zero means only that create, fresh reset, native oracle execution, evaluation,
and close completed consistently. A smoke report always has `gate_evaluable=false` and
`gate_passed=false`.

The JSON report contains aggregate counts, rates, the pinned revision, and a SHA256
fingerprint of the task set. It never contains raw task IDs, checkout paths, compiled
solutions, execution output, API arguments, evaluator requirements, or request traces.

## Recorded Gate 0a environment evidence

The committed [formal aggregate summary](../../data/evidence/appworld/gate0a/formal-2026-08-30.json)
records the fixed environment replay protocol over all 90 train and 57 dev tasks. Three
fresh worlds per task produced 441 episodes with all six aggregate rates equal to 1.0,
zero execution failures, zero exceptions, a Gate-evaluable result, and a passed Gate 0a.
The manifest separately records the verifier process's OS exit code as zero. This is
environment evidence for the pinned oracle replay lifecycle; it is not a model result or
a held-out evaluation.

For a successful audited run, the outer wrapper order is fixed: wait for the verifier to
finish and capture its status; atomically write a mode-0600 OS-exit record; validate exit
code zero and the exact summary integrity and hash; capture only safe process-log byte
metadata; delete the exact resolved private process-log path; atomically write aggregate
runtime metadata with `process_log_deleted=true`; only then promote the evidence and
checksums. The deletion target must be one exact, resolved, private regular-file path,
never a glob or symbolic link.

A temporary or nonzero run retains its private process log for diagnosis, must not claim
`process_log_deleted=true`, and must not promote evidence. The process log from the
recorded successful run was nonempty but was not content-reviewed; only its byte count
and source-metadata digest were retained. Neither the process log nor other raw runtime
artifacts are committed.

Schema v1 binds the AppWorld commit, task-set fingerprint, fixed protocol, and aggregate
results. It does not bind the ToolShift commit, wrapper version, runtime version, command
digest, or timestamps; the verifier commit is supported by the audited run label and
operational chain rather than by the summary itself. A future schema v2 must emit an
atomic completion record that binds those execution-provenance fields to the summary.
The evidence and this runbook intentionally contain no machine-specific checkout or
shared-filesystem path.
