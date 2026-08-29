# Contributing

## Branches and pull requests

1. Keep `main` releasable.
2. Create a focused branch such as `research/w1-foundation` or `feat/paired-sampler`.
3. Use small, atomic commits with imperative Conventional Commit messages.
4. Open a pull request with motivation, scope, verification commands, and experiment impact.
5. Merge only after specification and code-quality review.

## Testing

New behavior follows red-green-refactor: add a failing test, verify the expected failure, implement the minimum change, then run the focused and full test suites.

## Experiments

- Treat the base task as the statistical unit; do not count rollouts as independent tasks.
- Record code, model, environment, evaluator, configuration, and seed identifiers.
- Match rollout, token, tool-call, and optimization budgets across methods.
- Do not tune on AppWorld test splits or official BFCL evaluation data.
- Store large artifacts outside Git and publish manifests and hashes instead.

## Security and data handling

Never commit API keys, SSH material, tokens, decrypted AppWorld bundles, protected task data, model weights, or raw user data. Use `.env` locally and provide only `.env.example` with placeholder values.

