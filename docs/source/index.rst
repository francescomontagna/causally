Welcome to Causally's documentation!
====================================

**Causally** is a Python library for the generation of synthetic benchmarks for causal discovery.
``causally`` shines when you need flexibility and control on the modelling assumptions of your data.
You can benchmark your brand new causal discovery algorithm on challenging environments: data 
can be generated under the presence of latent confounders, measurement errors, autoregressive effects,
unfaithful path cancelling. Nobody believes that *absence of latent confouders* holds in the real world,
yet this is commonly assumed by the most prominent methods for causal discovery out there. Algorithms'
evaluation under challenging and realistic scenarios is crucial to deploy safe and robust models.
``causally`` enables that: happy causality!

Check out the :doc:`usage` section for further information, including
how to :ref:`install <installation>`.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   api