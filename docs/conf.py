import sys
import os

sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../../src'))
#project = 'alhazen-py'
html_title = 'Alhazen'
html_theme = 'sphinx_book_theme'

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']

