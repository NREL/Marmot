# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import pkg_resources


# -- Project information -----------------------------------------------------

project = 'Marmot'
copyright = '2022, Alliance for Sustainable Energy, LLC'
author = 'Daniel Levie, Marty Schwarz, Brian Sergi, Ryan Houseman'

# The full version, including alpha/beta/rc tags
version = '0.10.0'
release = version

# release = '0.10.0'

#
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autosummary','sphinx.ext.napoleon','sphinx.ext.autodoc', 
#               'sphinx.ext.coverage',
#               'sphinx.ext.githubpages','sphinx_rtd_theme']

# extensions = ['sphinx.ext.autosummary','sphinx.ext.napoleon','sphinx.ext.autodoc']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    'sphinx_click.ext',
    "sphinx.ext.autosectionlabel",
    "sphinx_panels",]

intersphinx_mapping = {
    "dateutil": ("https://dateutil.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    "py": ("https://pylib.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "statsmodels": ("https://www.statsmodels.org/devel/", None),
}

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

html4_writer=True

html_theme_options = {
    'collapse_navigation': False,
    # 'sticky_navigation': True,
    # 'titles_only': False,
    "navigation_depth": 4,
    "show_nav_level": 1,
    # 'display_version': True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NREL/Marmot",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },]
}


html_context = {
    "display_github": True,
    "github_user": "nrel",
    "github_repo": "Marmot",
    "github_version": "gh-pages",
    "conf_py_path": "/docsrc/source/",
}

# -- Extension configuration -------------------------------------------------

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_member_order = 'bysource'

autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
add_module_names = False  # Remove namespaces from class/method signatures
# # Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = True
# # numpy_show_class_member = True
# napoleon_google_docstring = True
# napoleon_use_param = True
# napoleon_preprocess_types = True

# napoleon_use_ivar = False
# napoleon_use_rtype = False