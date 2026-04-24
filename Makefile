# Project-level developer helpers.
#
# This intentionally wraps the existing Sphinx setup under `docs/` so contributors
# can build the design and implementation notes for the trajectory-stabilization
# task from the repository root.

.PHONY: docs docs-current docs-multi docs-clean docs-hx5

docs: docs-current

docs-current:
	@$(MAKE) -C docs current-docs

docs-multi:
	@$(MAKE) -C docs multi-docs

docs-clean:
	@rm -rf docs/_build

# Convenience alias for this project-specific documentation section.
docs-hx5: docs-current
	@printf "\nBuilt docs. Start at:\n"
	@printf "  docs/_build/current/source/overview/developer-guide/hx5_traj_stabilization/index.html\n\n"
