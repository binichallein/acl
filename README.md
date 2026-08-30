# ToolShift-RL

Research code for studying task-conditioned tail reliability of tool-using language-model agents under behaviorally equivalent tool interfaces.

The project uses AppWorld as the primary stateful training and evaluation environment and BFCL v4 as an external function-calling evaluation suite.

## Status

The project is in the environment-validation and pilot-design phase. Results and model artifacts will be published only after reproducibility and data-license checks. Model training has not started.

The local W1 scaffold and system-configuration layer are complete. On `ml2`, the pinned AppWorld checkout has completed the official test and task verification commands. The fixed Gate 0a replay protocol also passed over all 90 train and 57 dev tasks: 147 tasks × 3 repetitions = 441 episodes; all six aggregate rates were `1.0`, with zero execution failures and zero exceptions. This is environment evidence only, not a model result or a held-out evaluation. The reproducibility records and approved plans are:

- [AppWorld `ml2` manifest](data/manifests/appworld-ml2.yaml)
- [Gate 0a aggregate environment evidence](data/evidence/appworld/gate0a/formal-2026-08-30.json)
- [BFCL v4 manifest](data/manifests/bfcl-v4.yaml)
- [Research design](docs/plans/2026-08-29-tool-interface-tail-rl-design.md)
- [Implementation plan](docs/plans/2026-08-29-tool-interface-tail-rl-implementation-plan.md)

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

Verified AppWorld and BFCL revisions, together with the still-unresolved training-framework and base-model candidates, are recorded in `data/manifests/dependencies.yaml`. A candidate must receive a verified immutable identifier before its corresponding remote work is declared reproducible.

## Development principles

- Work on topic branches and merge through reviewed pull requests.
- Add tests before production code and keep experiments configuration-driven.
- Pin environment, model, evaluator, and dataset versions.
- Never commit credentials, decrypted benchmark bundles, model weights, or raw trajectories.
- Keep official BFCL results separate from any derived `BFCL-Shift` evaluation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.
