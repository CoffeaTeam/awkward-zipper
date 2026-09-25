.. this is a readme file that is rendered in the documentation html page.

.. image:: img/logo_unfilled_with_text.png
    :align: center
    :width: 500px
    :alt: logo

Awkward-Zipper
==============

Most of this code has been originally been written by the
`coffea <https://github.com/scikit-hep/coffea/tree/master>`__
developers.

Developing
----------

For installation using `uv <https://github.com/astral-sh/uv>`_:

.. code:: bash

   git clone https://github.com/CoffeaTeam/awkward-zipper.git
   cd awkward-zipper

   uv venv
   source .venv/bin/activate
   uv pip install -e . --group dev

Now you can run the test with:

.. code:: bash

   pytest tests/
