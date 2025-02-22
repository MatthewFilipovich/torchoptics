# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchOptics"
copyright = "2024-2025, Matthew Filipovich"
author = "Matthew Filipovich"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",
]
nb_execution_mode = "off"

autoapi_member_order = "alphabetical"
autoapi_dirs = ["../../torchoptics"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autodoc_typehints = "description"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
intersphinx_mapping = {"torch": ("https://pytorch.org/docs/stable/", None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_favicon = "_static/favicon.png"


# -- Custom configuration ----------------------------------------------------
# Skip modules in the autoapi extension to avoid duplication errors
def skip_modules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_modules)
