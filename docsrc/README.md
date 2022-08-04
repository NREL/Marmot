# How to edit and build docs

## Required Modules

The docs are built with sphinx. The following modules are required:

- sphinx==4.5.0
- sphinx_click==4.3.0
- sphinx_panels==0.6.0
- pydata_sphinx_theme==0.9.0

## Building docs

To build docs run **.\make github** from the Marmot/docsrc folder
If you have added new files or modules you will need to run **.\make clean** first, if you have only made modifications to existing files running clean is not required. 

## Editing files

All source files are found in the Marmot/docsrc/source folder

All files have a rst format, it is like markdown.
Files are laid out in folders with the same names as the online html pages.
All folders have a index.rst file. This is the initial landing page of each tab on the webpage and controls contents and links to files in the same grouping, e.g get_started

## Publishing

Once all edits are ready, commit changes and push to remote branch. GitHub will automatically update the docs within a few minutes. 

