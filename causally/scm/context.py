from abc import ABCMeta
from typing import Union, List


# * Base context class. Just for type hinting *
class Context(metaclass=ABCMeta):
    """Base class for specifying assumptions on the data generating process.
    
    Class inheriting from Context specifies stateful information on the assumptions
    required by the user for the data generating process. 
    """
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier
        self.parameters = dict()

# * Latent confounders assumption *
class ConfoundedModel(Context):
    """Class storing information for confounded data generation.

    Parameters
    ----------
    p_confounder: float, default 0.2
        The probability of adding a latent common cause between a pair of nodes,
        sampled as a Bernoulli random variable.
    """
    def __init__(self, p_confounder: float = 0.2):
        super().__init__(identifier="confounded")
        self.parameters["p_confounder"] = p_confounder

# * Measurement error assumption *
class MeasurementErrorModel(Context):
    r"""Class storing information for data generation under measurement errors.

    Parameters
    ----------
    gamma: Union[float, List[float]] 
        The inverse signal to noise ratio

        .. math::

        \\frac{\\operatorname{Var}(\\operatorname{error})}{\\operatorname{Var}(\\operatorname{signal})}

        parametrizing the variance of the measurement error proportionally to the
        variance of the signal. If a single float is provided, then gamma is the
        same for each column of the data matrix. Else, gamma is a vector of shape
        (num_nodes, ).
    """
    def __init__(self, gamma:Union[float, List[float]]) -> None:
        super().__init__(identifier="measurement error")
        if not gamma > 0 and gamma <= 1:
            raise ValueError("Signal to noise ratio outside of  (0, 1] interval")
        self.parameters["gamma"] = gamma


# * Time effects assumption * 
class AutoregressiveModel(Context):
    r"""Class storing information for data generation with time lags effects.

    Structural equations take the autoregressive form

    .. math::

    X_i(t):= f_i(\\operatorname{PA}_i(t)) + N_i + \\sum_{k=t-\\operatorname{order}} \\alpha(k)*X_i(k)

    where :math:`f_i` is the nonlinear causal mechanism,
    :math:`N_i` is the noise term of the structural equation,
    :math:`\alpha(k)` is a coefficient uniformly sampled between -1 and 1,
    :math:`t` is the sample step index, interpreted as the time step.

    Parameters
    ----------
    order: int
        The number of time lags
    """
    def __init__(self, order: int) -> None:
        super().__init__(identifier="autoregressive")
        self.parameters["order"] = order

# * Unfaithful assumption * 
class UnfaithfulModel(Context):
    """Class storing information for data generation with with unfaithful path cancelling.

    Class modelling unfaithful data cancelling in fully connected triplets
    ``X -> Y <- Z -> X``. 

    Parameters
    ----------
    p_unfaithful: float
        Probability of  unfaitfhul conditional independence in the presence of
        a fully connected triplet. 
    """
    def __init__(self, p_unfaithful: float) -> None:
        super().__init__(identifier="unfaithful")
        self.parameters["p_unfaithful"] = p_unfaithful