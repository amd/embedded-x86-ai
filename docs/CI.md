# CI

Locked for `github.com/amd/embedded-x86-ai`. Two workflows, two different jobs.

## Every pull request

`.github/workflows/ci.yml` on GitHub-hosted `ubuntu-24.04`:

- Python files parse
- Ruff, syntax and undefined names only (`ruff.toml`)
- Notebooks are valid JSON
- YAML loads
- Relative markdown links resolve
- Changed Python files have an AMD copyright header

After this workflow is green once, require it under **Settings → Branches** for `main`.

## Labs: one functional check and one performance check

`.github/workflows/demo-checks.yml`. A pull request runs only the labs it touches. A weekly run covers every lab that has opted in.

A lab opts in with `labs/<name>/ci/manifest.yml`:

```yaml
functional: python ci/functional.py
performance: python ci/performance.py
budget_ms: 80
```

`functional` exits 0 when the result matches the golden files in that lab. For a vision demo, bounding boxes match the golden JSON and the output image stays within tolerance of the golden image.

`performance` exits 0 and prints `PERF_MS=<milliseconds>`. The job fails when that number is above `budget_ms`. The budget is for the `strix-halo` runner only.

`labs/reachymini-lab` has no headless fixture yet, so it is skipped. Any other lab a pull request changes must have the manifest.

Workshops are not in this workflow. A workshop that needs this guarantee should put the runnable demo in `labs/<name>/`.

Skills are not in this workflow. Unit tests and a golden output belong next to the skill. The `amd/skillscope` catalog eval is a separate workflow, added when `evals/evals.py` has been converted to `evals/evals.json`.

## Runner

The hardware job uses `runs-on: [self-hosted, linux, strix-halo]`. It does not start until a lab has a manifest, so merging this setup does not wait on a machine.

Before adding the first manifest:

1. **Settings → Actions → Runners → New self-hosted runner.** Label it `strix-halo`. Install the ROCm and Ryzen AI stack that the lab README pins.
2. **Settings → Actions → General → Fork pull request workflows → Require approval for all outside collaborators.**

## Version tags

CI does not create tags. Tag the public repo (`v2026.11`) when you publish a snapshot. Each lab pins its stack in its README and `requirements.txt`. Skills carry their own version in the skill metadata. Workshops are not tagged.
