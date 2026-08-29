# Repository Instructions

- Follow the approved design and implementation plans under `docs/plans/`.
- Use test-driven development for production behavior.
- Do not modify benchmark test data or tune on held-out splits.
- Keep external network proxy settings out of committed files; configure them only in the shell or scheduler environment.
- Keep large artifacts on the ml2 shared filesystem and commit only small manifests with checksums.
- Use AppWorld through an external adapter layer; do not redistribute decrypted protected bundles.
- Report official BFCL v4 and derived BFCL-Shift results separately.
- Run formatting, linting, unit, contract, and smoke tests before requesting review.
