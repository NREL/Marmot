.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>



{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
    {%- if item not in inherited_members %}
    {%- if not item.startswith('_') %}
        ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}