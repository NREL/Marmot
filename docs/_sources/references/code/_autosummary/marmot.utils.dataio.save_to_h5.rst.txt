.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

   <script>
      var arr2 = document.getElementsByTagName('h1');
      for(var i = 0; i < arr2.length; i++) {
      arr2[i].innerHTML = arr2[i].innerHTML.replace(/\./g, '.<br />');
      }
   </script>


marmot.utils.dataio.save\_to\_h5
================================

.. currentmodule:: marmot.utils.dataio


.. autofunction:: save_to_h5