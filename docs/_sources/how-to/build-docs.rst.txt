.. raw:: html

    <style>
        h2  {border-bottom: 1px solid gray;}
    </style>

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=======================================================
How to edit and build Sphinx documentation for Marmot
=======================================================

This is a short guide on how to edit and build the Sphinx documentation that
you are currently reading for Marmot.

Required Modules
~~~~~~~~~~~~~~~~~~

The documentation is built with the python Sphinx package. The following modules are required:

- sphinx==4.5.0
- sphinx_click==4.3.0
- sphinx_panels==0.6.0
- sphinx-csv-filter==0.3.0
- pydata_sphinx_theme==0.9.0

.. note::
    To ensure the style of the documentation theme does not change,
    using the exact package versions listed above is very important!

gh-pages branch
~~~~~~~~~~~~~~~~~

All documentation lives on the gh-pages branch of Marmot. This is where all final edits and additions should be 
pushed to before publishing. 
It is advised to create a new branch from this one if you are making edits before merging back in. 

Editing files
~~~~~~~~~~~~~~~

All source files are found in the Marmot/docsrc/source directory

All files have a reStructuredText (reST) format and use the extension .rst, it is similar to markdown.
For an introduction to reStructuredText see 
`reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

Files are laid out in folders with the same names as the online html pages.
All directories have a index.rst file. This is the initial landing page of each tab on the webpage 
and controls contents and links to files in the same grouping, e.g get_started directory

Building documentation
~~~~~~~~~~~~~~~~~~~~~~~~~
To build docs run the following from the from the Marmot/docsrc directory::

        .\make github

If you have added new files or modules you will need to clean the documentation first::

        .\make clean

If you have only made modifications to existing files running clean is not required. 

Publishing
~~~~~~~~~~~~

Once all edits are ready, commit changes and push to remote branch. 
GitHub will automatically update the docs within a few minutes. 
