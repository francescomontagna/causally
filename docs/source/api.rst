API
===

Graph generators
----------------

.. currentmodule:: causally.graph.random_graphs
.. autosummary::
   :toctree: generated

   GraphGenerator
   GaussianRandomPartition
   ErdosRenyi
   BarabasiAlbert
   

Noise distributions
-------------------

.. currentmodule:: causally.scm.random_noises
.. autosummary::
   :toctree: generated

   Distribution
   RandomNoiseDistribution
   MLPNoise


Causal mechanisms
-----------------

.. currentmodule:: causally.scm.causal_mechanisms
.. autosummary::
   :toctree: generated

   PredictionModel
   LinearMechanism
   NeuralNetMechanism
   GaussianProcessMechanism
   InvertibleFunction
   

SCM properties
--------------

Causally allows modelling assumptions on the SCM such as presence of latent confounders, 
unfaithfulness of the data distribution, presence of measurement errors and time structure.

.. currentmodule:: causally.scm.scm_properties
.. autosummary::
   :toctree: generated

   ConfoundedModel
   MeasurementErrorModel
   UnfaithfulModel
   AutoregressiveModel


Structural causal models
------------------------
Causally implements linear, additive nonlinear, postnonliner structural causal models.
Additionally, it allows data generation from SCM with linear and nonlinear structural
equations.

.. currentmodule:: causally.scm.scm
.. autosummary::
   :toctree: generated

   BaseStructuralCausalModel
   AdditiveNoiseModel
   LinearModel
   PostNonlinearModel
