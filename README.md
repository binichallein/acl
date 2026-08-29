# ToolShift-RL

Research code for studying task-conditioned tail reliability of tool-using language-model agents under behaviorally equivalent tool interfaces.

The project uses AppWorld as the primary stateful training and evaluation environment and BFCL v4 as an external function-calling evaluation suite.

## Status

The project is in the environment-validation and pilot-design phase. Results and model artifacts will be published only after reproducibility and data-license checks.

## Development principles

- Work on topic branches and merge through reviewed pull requests.
- Add tests before production code and keep experiments configuration-driven.
- Pin environment, model, evaluator, and dataset versions.
- Never commit credentials, decrypted benchmark bundles, model weights, or raw trajectories.
- Keep official BFCL results separate from any derived `BFCL-Shift` evaluation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

