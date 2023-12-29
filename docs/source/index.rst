Welcome to Causally's documentation!
====================================

**Causally** is a Python library for the generation of synthetic benchmarks for causal discovery.
``causally`` shines when you need flexibility and control on the modelling assumptions of your data.
You can benchmark your brand new causal discovery algorithm on challenging environments: data 
can be generated under the presence of latent confounders, measurement errors, autoregressive effects, and
unfaithful path cancelling. Nobody believes that *absence of latent confounders* holds in the real world,
yet this is commonly assumed by prominent methods for causal discovery out there. Evaluation
of algorithms under challenging and realistic scenarios is crucial to deploy safe and robust models.
``causally`` enables that: happy causality!

Check out the :doc:`usage` section for further information, including
how to :ref:`install <installation>`.

.. note::

   Notebooks with examples will be added soon. Stay tuned!

Cite
----
If you find ``causally`` useful, please consider citing the publication 
`Assumption violations in causal discovery and the robustness of score matching <https://arxiv.org/abs/2310.13387>`_.

.. code-block:: latex

    @inproceedings{montagna2023_assumptions,
        title {Assumption violations in causal discovery and the robustness of score matching},
        author {Francesco Montagna and Atalanti A. Mastakouri and Elias Eulig and Nicoletta Noceti and Lorenzo Rosasco and Dominik Janzing and Bryon Aragam and Francesco Locatello},
        booktitle {Advances in Neural Information Processing Systems 37 (NeurIPS)},
        year {2023}
    }

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   api