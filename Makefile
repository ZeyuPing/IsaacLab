# Project-level developer helpers.
#
# This file only adds repository-local convenience targets. Upstream Isaac Lab
# docs and contributor workflows remain unchanged.

PYTHON ?= $(CURDIR)/env_isaaclab/bin/python
TACADA_DOCS_DIR := tacada_docs
TACADA_DOCS_BUILD := $(TACADA_DOCS_DIR)/_build/html
TACADA_PY_FILES := \
	source/isaaclab_mimic/isaaclab_mimic/envs/trajectory_stabilization_mimic_env.py \
	source/isaaclab_mimic/isaaclab_mimic/envs/trajectory_stabilization_mimic_env_cfg.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/interfaces.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/trajectory_stabilization_env_cfg.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/actions.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/curriculums.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/observations.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/rewards.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/terminations.py \
	source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/config/hx5_d20/hx5_d20_env_cfg.py
TACADA_DOC_FILES := \
	$(TACADA_DOCS_DIR)/conf.py \
	$(TACADA_DOCS_DIR)/index.rst \
	$(TACADA_DOCS_DIR)/design_goals.rst \
	$(TACADA_DOCS_DIR)/architecture.rst \
	$(TACADA_DOCS_DIR)/implementation_status.rst \
	$(TACADA_DOCS_DIR)/code_map.rst \
	$(TACADA_DOCS_DIR)/README.md \
	$(TACADA_DOCS_DIR)/requirements.txt

.PHONY: docs docs-current docs-multi docs-clean docs-hx5 \
	tacada-docs tacada-docs-clean tacada-check-fast tacada-check-ruff tacada-check-format \
	tacada-check-pyright tacada-check

docs: docs-current

docs-current:
	@$(MAKE) -C docs current-docs

docs-multi:
	@$(MAKE) -C docs multi-docs

docs-clean:
	@rm -rf docs/_build

# Convenience alias for the upstream integrated project documentation section.
docs-hx5: docs-current
	@printf "\nBuilt docs. Start at:\n"
	@printf "  docs/_build/current/source/overview/developer-guide/hx5_traj_stabilization/index.html\n\n"

tacada-docs:
	@$(MAKE) -C $(TACADA_DOCS_DIR) html PYTHON=$(PYTHON)
	@printf "\nBuilt local TacAda docs. Start at:\n"
	@printf "  $(TACADA_DOCS_BUILD)/index.html\n\n"

tacada-docs-clean:
	@$(MAKE) -C $(TACADA_DOCS_DIR) clean

tacada-check-fast:
	@$(PYTHON) -m py_compile $(TACADA_PY_FILES)
	@printf "TacAda syntax check passed.\n"

tacada-check-ruff:
	@$(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)" || \
		(printf "ruff is not installed in env_isaaclab. Install it with:\n  %s -m pip install ruff\n" "$(PYTHON)" && exit 1)
	@$(PYTHON) -m ruff check $(TACADA_PY_FILES) $(TACADA_DOC_FILES)

tacada-check-format:
	@$(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)" || \
		(printf "ruff is not installed in env_isaaclab. Install it with:\n  %s -m pip install ruff\n" "$(PYTHON)" && exit 1)
	@$(PYTHON) -m ruff format --check $(TACADA_PY_FILES) $(TACADA_DOC_FILES)

tacada-check-pyright:
	@$(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyright') else 1)" || \
		(printf "pyright is not installed in env_isaaclab. Install it with:\n  %s -m pip install pyright\n" "$(PYTHON)" && exit 1)
	@$(PYTHON) -m pyright $(TACADA_PY_FILES)

tacada-check: tacada-check-fast tacada-check-ruff tacada-check-format tacada-check-pyright tacada-docs
