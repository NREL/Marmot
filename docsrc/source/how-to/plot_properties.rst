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

==========================
Plotting Properties Index
==========================

This is a reference guide which shows which properties are **required** and which are **optional** 
when creating plots with Marmot. If a property is required, you must first process that property for your 
desired model (e.g. PLEXOS, ReEDS, SIIP) so that it is contained within the formatted.h5 file. Ensure that
the required property is set to **TRUE** in your models :doc:`property.csv <../references/input-files/files>`

This table is laid out as follows:

- **Plotting Module:** The Marmot python plotting module, equivalent to the *Marmot Module* in the :ref:`Marmot_plot_select: csv file`.
- **Plotting Method:** The name of the method containing the specific plot logic, equivalent to the *Method* in the :ref:`Marmot_plot_select: csv file`.
- **Required Marmot Property:** These properties are required to create the plot.
- **Optional Marmot Property:** These properties are optional but add extra information to a plot.


.. csv-filter:: 
   :file: ../tables/plotting_properties.csv
   :widths: 10, 30, 30, 30
   :header-rows: 1
