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
        author = {Montagna, Francesco and Mastakouri, Atalanti and Eulig, Elias and Noceti, Nicoletta and Rosasco, Lorenzo and Janzing, Dominik and Aragam, Bryon and Locatello, Francesco},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
        pages = {47339--47378},
        publisher = {Curran Associates, Inc.},
        title = {Assumption violations in causal discovery and the robustness of score matching},
        url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/93ed74938a54a73b5e4c52bbaf42ca8e-Paper-Conference.pdf},
        volume = {36},
        year = {2023}
    }

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   api