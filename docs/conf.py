# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import datetime as dt
import json
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import gobbli

file_loc = os.path.split(__file__)[0]
sys.path.insert(0, os.path.join(os.path.dirname(file_loc), "."))


# -- Project information -----------------------------------------------------

with open(os.path.join(file_loc, "..", "meta.json"), "r") as f:
    meta = json.load(f)

project = meta["name"]
author = meta["author"]
copyright = f"{dt.date.today().year}, {author}"

# The full version, including alpha/beta/rc tags
version = meta["version"]
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_paramlinks",
]

autodoc_default_options = {
    "members": None,
    "inherited-members": None,
    "show-inheritance": None,
}

autoclass_content = "both"
autosummary_generate = True

intersphinx_mapping = {
    "docker": ("https://docker-py.readthedocs.io/en/stable/", None),
    "ray": ("https://ray.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"
html_theme_options = {
    "description": "Deep learning doesn't have to be scary",
    "logo": "gobbli_lg.svg",
    "touch_icon": "gobbli_app.svg",
    "github_banner": "true",
    "github_button": "true",
    "github_repo": "gobbli",
    "github_user": "RTIInternational",
    "page_width": "1040px",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]
}

html_favicon = os.path.join("_static", "gobbli_app.svg")
html_title = f"gobbli {version} documentation"

# Autogenerate API docs
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    # Repository root
    base_dir = Path(__file__).parent.parent.resolve()

    output_path = base_dir / "docs" / "auto"
    main(
        [
            "--no-toc",
            "--separate",
            "-o",
            str(output_path),
            str(base_dir / project),
            str(base_dir / project / "model" / "*" / "src"),
        ]
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)
