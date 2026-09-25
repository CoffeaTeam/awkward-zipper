Quick start
--------

*NOTE: You can find this whole page as a jupyter notebook and try it out yourself •⩊• here:* ``notebooks/Usage_example_of_awkward_zipper.ipynb`` |br| |br|
Before even starting to work with awkward-zipper we have to use ``uproot`` library to load the data. First we load our root file:

.. code:: bash

    # Create a TTree from root
    tree = uproot.open("nano_dy.root")["Events"]

Then we use ``TBranch.arrays`` function to conver the TTree format we got
form the root file into an awkward array:

.. code:: bash

    # TTree -> awkward.Array[awkward.Record[str, awkward.Array]]
    array = tree.arrays(ak_add_doc=True)

Finally we can pass the resulting array to awkward-zipper.

.. code:: bash

    from awkward_zipper import NanoAOD
    restructure = NanoAOD(version="latest")
    result = restructure(array)

This whole process we can picture on a diagram:

.. graphviz::

   digraph {
      "Root file" -> "TTree" ->
        "awkward.Array[awkward.Record[str, awkward.Array]]"
        -> "awkward-zipper";
   }

How awkward-zipper works internally
--------

Now let's see how awkward-zipper restructures arrays internally,
using Nanoaod schema(layout) as an example.
The whole process can be broken down into two big steps.

Adding new fields
"""""""""

Any branches named ``n{name}`` are assumed to be counts branches
and can converted to offsets using this helper function:

+-------------------------------+
| counts2offsets(counts)        |
+===============================+
| Counts |br|                   |
| [5, 8, 5, 3, 5]               |
+-------------------------------+
| Result |br|                   |
| [0,  5, 13, 18, 21, 26]       |
+-------------------------------+

Any local index branches with names matching ``{source}_{target}Idx*`` are converted to global indexes for the event chunk (postfix ``G``).
All local indices and their correlating global indices are taken from :class:`awkward_zipper.NanoAOD.all_cross_references` dictionary.

+---------------------------------+
| local2globalindex(index, counts)|
+=================================+
| Index array |br|                |
| [[], [1], [], [2, 3]]           |
| |br| |br|                       |
| Counts array |br|               |
| [8, 7, 4, 7]                    |
+---------------------------------+
| Result |br|                     |
| [[], [9], [], [21, 22]]         |
+---------------------------------+

Any ``nested_items`` are constructed, if the necessary branches are available.
All the input indices are taken from the :class:`awkward_zipper.NanoAOD.nested_items` dictionary.

+---------------------------------+
| nestedindex(indices)            |
+=================================+
| First index array |br|          |
| [[0, 2, 4], [8, 6]]             |
| |br| |br|                       |
| Second index array |br|         |
| [[1, 3, 5], [-1, 7]]            |
+---------------------------------+
| Result |br|                     |
| [[[0, 1], [2, 3], [4, 5]], |br| |
| [[8, -1], [6, 7]]]              |
+---------------------------------+

In the same manner any :class:`awkward_zipper.NanoAOD.nested_index_items` and :class:`awkward_zipper.NanoAOD.special_items` are constructed,
if the necessary branches are available. You can find all these functions in the documentation for :class:`awkward_zipper.kernels`

Grouping(zipping) all the fields together
""""

From those arrays, NanoAOD collections are formed as collections of branches grouped by name, where:

* one branch exists named ``name`` and no branches start with ``name_``, interpreted as a single flat array;

.. table::
    :align: center

    +------------------------------------------------------------------------------------------------+
    | Example: Each event has only one Run Id. Interpreted flat array will look look like this: |br| |
    | <Array [1, 1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1] type='40 * uint32 |br|                           |
    | [parameters={"__doc__": "run/i"}]'> |br|                                                       |
    +------------------------------------------------------------------------------------------------+

* one branch exists named ``name``, one named ``n{name}``, and no branches start with ``name_``,
  interpreted as a single jagged array;

.. table::
    :align: center

    +--------------------------------------------------------------------------------------------------------------------+
    | Example: Each event has a flat array of PS Weights. Interpreted single jagged array will look look like this: |br| |
    | <Array                                                                                                        |br| |
    | [[1.01, 1.26, 0.99, 0.791],                                                                                   |br| |
    | [2.06, 0.872, 0.535, 0.962],                                                                                  |br| |
    | [1.07, 0.887, 0.933, 1.02],                                                                                   |br| |
    | [0.833, 0.827, 1.15, 1.1],                                                                                    |br| |
    | ...]                                                                                                          |br| |
    | type='40 * [var * float32, ...'>                                                                                   |
    +--------------------------------------------------------------------------------------------------------------------+

* no branch exists named ``{name}`` and many branches start with ``name_*``, interpreted as a flat table; or

.. table::
    :align: center

    +--------------------------------------------------------------------------------------------------------------------+
    | Example: Each event has a SINGLE Generator. Each Generator consists of a record of Generator parameters.           |
    | These parameters can be scalars or flat arrays. Interpreted flat table will look look like this:              |br| |
    | <Array                                                                                                        |br| |
    | [{id2: -1, x1: 0.214, xpdf2: 0, xpdf1: 0, weight: 2.63e+04, id1: 1, ...},                                     |br| |
    | {id2: 1, x1: 0.0142, xpdf2: 0, xpdf1: 0, weight: 2.63e+04, id1: -1, ...},                                     |br| |
    | ...]                                                                                                          |br| |
    | type='40 * NanoCollection[xpd...'>                                                                                 |
    +--------------------------------------------------------------------------------------------------------------------+

* one branch exists named ``n{name}`` and many branches start with ``name_*``, interpreted as a jagged table.

.. table::
    :align: center

    +--------------------------------------------------------------------------------------------------------------------+
    | Example: Each event has an array of Jets. Each Jet consists of a record of Jet parameters.                         |
    | These parameters can be scalars or flat arrays. Interpreted jagged table will look look like this:            |br| |
    | <Array                                                                                                        |br| |
    | [[Jet, ..., Jet],                                                                                             |br| |
    | [Jet, ...],                                                                                                   |br| |
    | [Jet, ...],                                                                                                   |br| |
    | ...]                                                                                                          |br| |
    | type='40 * var * Jet[area: f...'>                                                                                  |
    +--------------------------------------------------------------------------------------------------------------------+

Collections are assigned mixin types according to the ``mixins`` :class:`awkward_zipper.NanoAOD.mixins` mapping.
Finally, all collections are then zipped into one `NanoEvents` record and returned.

.. configures inserting |br| in the text to force a new line
.. |br| raw:: html

     <br>
