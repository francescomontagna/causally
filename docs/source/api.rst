API
===

Graph generators
----------------

Classes for generation of random graphs.

.. currentmodule:: causally.graph.random_graph
.. autosummary::
   :toctree: generated/
   
   GaussianRandomPartition
   ErdosRenyi
   BarabasiAlbert


Noise distributions
-------------------

Classes for random noise generation according to different parametric and nonparametric distributions.

.. currentmodule:: causally.scm.noise
.. autosummary::
   :toctree: generated/

   RandomNoiseDistribution
   MLPNoise
   Normal
   Exponential
   Uniform


Causal mechanisms
-----------------

Causally predefines linear and nonlinear causal mechanisms for the definition of structural equations.

.. currentmodule:: causally.scm.causal_mechanism
.. autosummary::
   :toctree: generated/

   LinearMechanism
   NeuralNetMechanism
   GaussianProcessMechanism
   InvertibleFunction
   

Challenging assumptions
-----------------------

Causally allows specifying challenging modelling assumptions on the SCM such as presence of
latent confounders, unfaithfulness of the data distribution, presence of measurement errors
and time structure.

.. currentmodule:: causally.scm.context
.. autosummary::
   :toctree: generated/

   ConfoundedModel
   MeasurementErrorModel
   UnfaithfulModel
   AutoregressiveModel


Structural causal models
------------------------
Causally implements linear, additive nonlinear, postnonliner structural causal models.
Additionally, it allows data generation from SCMs with linear and nonlinear structural
equations.

.. currentmodule:: causally.scm.scm
.. autosummary::
   :toctree: generated/

   BaseStructuralCausalModel
   AdditiveNoiseModel
   LinearModel
   PostNonlinearModel
   MixedLinearNonlinearModel
