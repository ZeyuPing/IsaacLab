"""Standalone Sphinx configuration for TacAda project notes."""

from __future__ import annotations

project = "TacAda Trajectory Stabilization"
copyright = "2026, TacAda Project Contributors"
author = "TacAda Project Contributors"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = project
html_static_path = []

master_doc = "index"
