class InitializerInterface:
    """
    Interface for state initializers in the racing environment.
    This interface defines the methods that any state initializer must implement.
    """
    def __init__(self, env):
        self.env = env

    def initialize(self, mask=None, start_at_zero=False):
        """
        Initialize the state of the simulator.

        Args:
            mask: Optional mask to apply during initialization.
            start_at_zero: Whether to start the state at zero.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")