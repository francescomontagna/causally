class SCMContext:
    def __init__(self) -> None:
        self._p_confounder = None
        self._p_unfaithful = None
        self._autoregressive_order = None
        self._measure_error_gamma = None
        self.assumptions = list()

    def confounded_model(self, p_confounder: float):
        """Make the assumption of model with latent confounders.

        Parameters
        ----------
        p_confounder: float, default 0.2
            The probability of adding a latent common cause between a pair of nodes,
            sampled as a Bernoulli random variable. The value provided must be in
            the range (0, 1].
        """
        if p_confounder <= 0 or p_confounder > 1:
            raise ValueError(
                "The value of p_confounder must be in the range (0, 1]"
                + f" Instead, got p_confounder={p_confounder}."
            )
        self.assumptions.append("confounded")
        self._p_confounder = p_confounder

    def unfaithful_model(self, p_unfaithful: float):
        """Make the assumption of model with distribution unfaithful to the graph.

        Unfaithful path cancelling are modelled in fully connected triplets
        ``X -> Y <- Z -> X``. TODO: add model explaination.

        Parameters
        ----------
        p_unfaithful: float
            Probability of  unfaitfhul conditional independence in the presence of
            a fully connected triplet. The value provided must be in
            the range (0, 1].
        """
        if p_unfaithful <= 0 or p_unfaithful > 1:
            raise ValueError(
                "The value of p_unfaithful must be in the range (0, 1]"
                + f" Instead, got p_unfaithful={p_unfaithful}."
            )
        self.assumptions.append("unfaithful")
        self._p_unfaithful = p_unfaithful

    def autoregressive_model(self, order: int):
        """Make the assumption of model with time lagged autoregressive effects.

        Structural equations take the following autoregressive form:

        .. math:: X_i(t):= f_i(\operatorname{PA}_i(t)) + N_i + \sum_{k=t-\operatorname{order}} \alpha(k)*X_i(k)

        where :math:`f_i` is the nonlinear causal mechanism,
        :math:`N_i` is the noise term of the structural equation,
        :math:`\alpha(k)` is a coefficient uniformly sampled between -1 and 1,
        :math:`t` is the sample step index, interpreted as the time step.

        Parameters
        ----------
        order: int
            The number of time lags
        """
        if order <= 0:
            raise ValueError(
                "The value of order must be an integer larger than 0."
                + f" Instead, got order={order}."
            )
        self.assumptions.append("autoregressive")
        self._autoregressive_order = order

    def measure_err_model(self, gamma: float):
        """Make the assumption of model with measurement error.

        Parameters
        ----------
        gamma: Union[float, List[float]] 
            The inverse signal to noise ratio 
            :math:`\gamma := \frac{\operatorname{Var}(\operatorname{error})}{\operatorname{Var}(\operatorname{signal})}`\
            parametrizing the variance of the measurement error proportionally to the
            variance of the signal. If a single float is provided, then gamma is the
            same for each column of the data matrix. Else, gamma is a vector of shape
            (num_nodes, ).
        """
        if gamma <= 0:
            raise ValueError(
                "The value of gamma must be larger than 0."
                + f" Instead, got gamma={gamma}."
            )
        self.assumptions.append("measurement_error")
        self._measure_error_gamma = gamma

    @property
    def p_confounder(self):
        if self._p_confounder is None:
            raise AttributeError(
                "p_confounder is None. If you want a structural causal"
                + " model with latent confouders, call the method"
                + " ``confounded_model``."
            )
        return self._p_confounder

    @property
    def p_unfaithful(self):
        if self._p_unfaithful is None:
            raise AttributeError(
                "p_unfaithful is None. If you want a structural causal"
                + " model with unfaithful distribution, call the method"
                + " ``unfaithful_model``."
            )
        return self._p_unfaithful

    @property
    def measure_error_gamma(self):
        if self._measure_error_gamma is None:
            raise AttributeError(
                "measure_error_gamma is None. If you want a structural causal"
                + " model with measurement error on the data, call the method"
                + " ``measure_err_model``."
            )
        return self._p_confounder

    @property
    def autoregressive_order(self):
        if self._autoregressive_order is None:
            raise AttributeError(
                "autoregressive_order is None. If you want a structural causal"
                + " model with autoregressive effects, call the method"
                + " ``autoregressive_model``."
            )
        return self._autoregressive_order
