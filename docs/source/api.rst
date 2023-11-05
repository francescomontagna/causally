API
===

.. autosummary::
   :toctree: generated

   causally


Noise distributions
===================

.. currentmodule:: causally.datasets.random_noises
.. autosummary::
   :toctree: generated/

   Distribution
   RandomNoiseDistribution
   MLPNoise


Causal mechanisms
=================

.. currentmodule:: causally.datasets.causal_mechanisms
.. autosummary::
   :toctree: generated/

   PredictionModel
   LinearMechanism
   NeuralNetMechanism
   GaussianProcessMechanism
   InvertibleFunction


Graph generators
================

.. currentmodule:: causally.datasets.random_graphs
.. autosummary::
   :toctree: generated/

   GraphGenerator
   GaussianRandomPartition
   ErdosRenyi
   BarabasiAlbert
   

.. SCM properties
.. ===============

.. Causally allows modelling assumptions on the SCM such as presence of latent confounders, 
.. unfaithfulness of the data distribution, presence of measurement errors and time structure.

.. .. currentmodule:: causally.datasets.scm_properties
 .. autosummary::
..    :toctree: generated/

..    ConfoundedModel
..    MeasurementErrorModel
..    UnfaithfulModel
..    AutoregressiveModel


Structural causal models
========================
Causally implements linear, additive nonlinear, postnonliner structural causal models.
Additionally, it allows data generation from SCM with linear and nonlinear structural
equations.

.. currentmodule:: causally.datasets.scm
.. autosummary::
   :toctree: generated/

   BaseStructuralCausalModel
   AdditiveNoiseModel
   LinearModel
   PostNonlinearModel
