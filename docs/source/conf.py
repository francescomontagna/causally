# Configuration file for the Sphinx documentation builder.

import os
import sys

#Location of Sphinx files
sys.path.insert(0, os.path.abspath('../..'))

# skip init
def skip_init(app, what, name, obj, skip, options):
    if name == "__init__":
        return True
    return None

def setup(app):
    app.connect("autodoc-skip-member", skip_init)


# -- Project information

project = 'Causally'
copyright = '2023, Francesco Montagna'
author = 'Francesco Montagna'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance', 'inherited-members']


# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Napoleon settings (Napoleon needed to parse Numpy like docstrings)
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True