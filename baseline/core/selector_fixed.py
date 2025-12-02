"""Fixed aggregator selector for baseline experiments.

Always returns a pre-configured domain as the aggregation point,
used for FedAvg/FedProx baselines without dynamic selection.
"""


class FixedSelector:
    """Fixed aggregator selector.

    Always returns the same configured domain, regardless of metrics.
    """

    def __init__(self, fixed_domain: str = "real"):
        """Initialize fixed selector.

        Args:
            fixed_domain: Domain to always select as aggregator
        """
        self.fixed_domain = fixed_domain

    def select(self, *args, **kwargs) -> str:
        """Select aggregation point.

        Args:
            *args: Ignored (for API compatibility)
            **kwargs: Ignored (for API compatibility)

        Returns:
            The fixed domain name
        """
        return self.fixed_domain
