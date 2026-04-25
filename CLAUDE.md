# TacAda Local Workflow

This repository includes an isolated TacAda documentation and validation workflow for
local development on machines that do not have a full IsaacLab installation.

## Local TacAda docs

The authoritative local docs for this project live under `tacada_docs/` and
build with a standalone Sphinx configuration.

Build them from the repository root with:

```bash
make tacada-docs
```

The generated landing page is:

```text
tacada_docs/_build/html/index.html
```

This validates that the project notes render cleanly as Sphinx documentation,
but it does not validate the upstream Isaac Lab docs tree or autosummary import
paths.

## Local TacAda checks

The root `Makefile` exposes TacAda-only validation commands:

```bash
make tacada-check-fast
make tacada-check-ruff
make tacada-check-format
make tacada-check-pyright
make tacada-check
```

These are intentionally scoped to:

- `source/isaaclab_tasks/.../trajectory_stabilization/...`
- `source/isaaclab_mimic/.../trajectory_stabilization_*`
- `tacada_docs/`

`tacada-check-fast` only requires Python and compiles the changed TacAda Python files.
The Ruff and Pyright checks require those tools to be installed into `env_isaaclab/`.

## What this local workflow does not cover

This Mac workflow does not replace a real IsaacLab runtime environment. It does
not validate:

- Isaac Sim startup
- environment boot or rollouts
- upstream `docs/` Sphinx build
- end-to-end Mimic execution
- full repo test coverage

Use it as a cheap pre-commit safety net and readable project documentation path.
