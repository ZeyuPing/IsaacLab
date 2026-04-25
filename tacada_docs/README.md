# TacAda standalone docs

This directory contains a standalone Sphinx project for the TacAda
trajectory-stabilization work. It is separate from Isaac Lab's upstream
`docs/` tree so it can be built on a Mac without importing IsaacLab runtime
modules.

## Build

From the repository root:

```bash
make tacada-docs
```

Or from this directory directly:

```bash
make html
```

## What this validates

- Sphinx rendering for the TacAda project notes
- internal links between the TacAda project pages
- a readable, shareable documentation bundle for local development

## What this does not validate

- upstream Isaac Lab docs builds
- autosummary imports
- Isaac Sim runtime behavior
- environment execution
