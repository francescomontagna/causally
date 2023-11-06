Welcome to Causally's documentation!
====================================

**Causally** is a Python library for the generation of synthetic benchmarks for causal discovery. 
Causally allows specifying several properties of the structural causal model generating the data, 
including violation of common assumptions such as absence of latent confounders, faithfulness of the data distribution,
absence of measurement error, as well as the *i.i.d.* assumption on the observed data.
``causally`` data generation suite enables evaluation of causal discovery algorithms in challenging environment,
beyond common assumptions in causal discovery that are hardly met in the real-world scenarios of interest.

Check out the :doc:`usage` section for further information, including
how to :ref:`install <installation>`.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 3

   usage
   api