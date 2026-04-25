# Standalone TacAda Sphinx docs.

PYTHON ?= ../env_isaaclab/bin/python
SPHINXBUILD ?= $(PYTHON) -m sphinx
SOURCEDIR := .
BUILDDIR := _build/html

.PHONY: html clean

html:
	@$(SPHINXBUILD) -W --keep-going -b html $(SOURCEDIR) $(BUILDDIR)

clean:
	@rm -rf _build
