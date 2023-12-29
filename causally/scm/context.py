class SCMContext:
    def __init__(self) -> None:
        self._p_confounder = None
        self._p_unfaithful = None
        self._autoregressive_order = None
        self._measure_error_gamma = None
        self.assumptions = list()

        # Stateful information about violations
        self._confounded_adjacency = None
        self._unfaithful_adjacency = None

    def confounded_model(self, p_confounder: float):
        """Make the assumption of model with latent confounders.

        Parameters
        ----------
        p_confounder: float, default 0.2
            The probability of adding a latent common cause between a pair of nodes,
            parametrizing a Bernoulli random variable. The value provided must be in
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

        Unfaithful distribution are modeled via directed paths canceling.
        In practice, we identify the fully connected triplets of nodes 
        :math:`X_i \\rightarrow X_k \leftarrow X_j \leftarrow X_i` in the
        ground truth, and we adjust the causal mechanisms such that the
        direct effect of :math:`X_i` on :math:`X_k` cancels out.
        As an example, consider a graph :math:`\mathcal{G}` with vertices
        :math:`X_1, X_2, X_3`. We allow for mixed linear and nonlinear
        mechanisms, and define the set of structural equations as:

        .. math::

                & X_1 := U_1,\\\\
                & X_2 := f(X_1) + U_2,\\\\
                & X_3 := f(X_1) - X_2 + U_3,
        
        with :math:`f` nonlinear function. This definition of the mechanisms
        on :math:`X_3` cancels out :math:`f(X_1)` in the structural equation. 

        Parameters
        ----------
        p_unfaithful: float
            Probability of  unfaitfhul conditional independence in the presence of
            a fully connected triplet. The value provided must be in
            the range (0, 1].

        Notes
        -----
        Unfaithfulness of the distribution is not supported for the post-nonlinear model.
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

        .. math::

                X_i(t):= f_i(\operatorname{PA}_i(t)) + N_i + \sum_{k=t-\operatorname{order}} \\alpha(k) X_i(k)

        where :math:`f_i` is the nonlinear causal mechanism,
        :math:`N_i` is the noise term of the structural equation,
        :math:`\\alpha(k)` is a coefficient uniformly sampled between -1 and 1,
        :math:`t` the index of a sample, interpreted as the time step.

        Parameters
        ----------
        order: int
            The number of time lags

        Notes
        -----
        Time lagged autoregressive effects are not supported for the post-nonlinear model.
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

        Rather than observing perfectly measure random variables :math:`x_i`
        in the dataset (where :math:`i` is the index of a node), we observe
        :math:`\\tilde{x}_i := x_i + \epsilon_i`, where :math:`\epsilon_i` is
        a Gaussian random variable centered at zero, whose variance is parametrized
        by the inverse signal to noise ratio
        :math:`{\gamma} := \\frac{\operatorname{Var}(\operatorname{error})}{\operatorname{Var}(\operatorname{signal})}`.


        Parameters
        ----------
        gamma: Union[float, List[float]] 
            The inverse signal to noise ratio 
            :math:`{\gamma} := \\frac{\operatorname{Var}(\operatorname{error})}{\operatorname{Var}(\operatorname{signal})}`\
            parametrizing the variance of the measurement error proportionally to the
            variance of the signal. If a single float is provided, then gamma is the
            same for each node in the graph. Else, gamma is a vector of shape
            ``(num_nodes, )``.
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

    @property
    def confounded_adjacency(self):
        if self._confounded_adjacency is None:
            raise AttributeError(
                "confounded_adjacency is None. "
                + " The confounded adjacency matrix is created in the"
                + " `sample` method of the BaseStructuralCausalModel class"
            )
        return self._confounded_adjacency

    @confounded_adjacency.setter
    def confounded_adjacency(self, A):
        self._confounded_adjacency = A

    @property
    def unfaithful_adjacency(self):
        if self._unfaithful_adjacency is None:
            raise AttributeError(
                "unfaithful_adjacency is None. "
                + " The unfaithful adjacency matrix is created in the"
                + " `sample` method of the BaseStructuralCausalModel class"
            )
        return self._unfaithful_adjacency

    @unfaithful_adjacency.setter
    def unfaithful_adjacency(self, A):
        self._unfaithful_adjacency = A
