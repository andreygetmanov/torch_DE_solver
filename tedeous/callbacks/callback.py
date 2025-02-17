from abc import ABC, abstractmethod

class Callback(ABC):
    """Base class used to build new callbacks.
    """

    def __init__(self):
        """
    Initializes the object.

    Initializes the object with default values for print_every, verbose, validation_data, and _model.

    Args:
        self: The instance of the class.

    Returns:
        None
    """
        self.print_every = None
        self.verbose = 0
        self.validation_data = None
        self._model = None

    def set_params(self, params):
        """
    Sets the parameters for the object.

    Args:
        params (dict): A dictionary of parameters to be set.

    Returns:
        None
    """
        self.params = params

    def set_model(self, model):
        """
    Sets the model for the current instance.

    Args:
        model: The model to be set.

    Returns:
        None
    """
        self._model = model

    @property
    def model(self):
        """
    Returns the internal model instance.

    Args:
        self: The instance of the class.

    Returns:
        None: The internal model instance.
    """
        return self._model

    def on_epoch_begin(self, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_epoch_end(self, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        pass

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        pass

    def during_epoch(self, logs=None):
        """
    Called at the end of an epoch during training.

    Args:
        logs (dict, optional): Dictionary of logs. Defaults to None.

    Returns:
        None
    """
        pass