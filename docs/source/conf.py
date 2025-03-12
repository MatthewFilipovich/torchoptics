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
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

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
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples", "user_guide"],
    "gallery_dirs": ["auto_examples", "auto_user_guide"],
    "reference_url": {"torchoptics": None},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "TorchOptics"

html_favicon = "_static/favicon.png"
html_logo = "_static/torchoptics_logo.png"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MatthewFilipovich/torchoptics",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchoptics",
            "icon": "fa-solid fa-box",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "sg_download_links", "sg_launcher_links"],
}

html_static_path = ["_static"]
html_sidebars = {"auto_user_guide/plot_quickstart": []}  # Disable sidebar for specific pages


# -- Custom configuration ----------------------------------------------------
# Skip modules in the autoapi extension to avoid duplication errors
def skip_modules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_modules)
    app.add_css_file("hide_links.css")  # Custom CSS to hide jupyter links
