# ToolShift-RL

Research code for studying task-conditioned tail reliability of tool-using language-model agents under behaviorally equivalent tool interfaces.

The project uses AppWorld as the primary stateful training and evaluation environment and BFCL v4 as an external function-calling evaluation suite.

## Status

The project is in the environment-validation and pilot-design phase. Results and model artifacts will be published only after reproducibility and data-license checks.

The local W1 project scaffold and system-configuration layer are under development. The remote AppWorld installation and smoke test on `ml2` have **not** been completed yet.

## Local development

ToolShift requires Python 3.10 or newer. Create an isolated environment and install the package with its development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to an ignored `.env` or export `TOOLSHIFT_SHARED_ROOT` in the runtime shell. Never commit the machine-specific shared path, proxy settings, or credentials.

Run the local quality gates with:

```bash
ruff check .
pytest
```

The dependency candidates and their unresolved revisions are recorded in `data/manifests/dependencies.yaml`. Revisions must be replaced with verified immutable identifiers before the corresponding remote smoke test is declared reproducible.

## Development principles

- Work on topic branches and merge through reviewed pull requests.
- Add tests before production code and keep experiments configuration-driven.
- Pin environment, model, evaluator, and dataset versions.
- Never commit credentials, decrypted benchmark bundles, model weights, or raw trajectories.
- Keep official BFCL results separate from any derived `BFCL-Shift` evaluation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.
