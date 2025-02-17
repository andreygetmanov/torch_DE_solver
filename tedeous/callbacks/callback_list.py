from tedeous.callbacks.callback import Callback

# import tree

class CallbackList(Callback):
    """Container abstracting a list of callbacks."""
    def __init__(
        self,
        callbacks=None,
        model=None,
        **params,
    ):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
            callbacks: List of `Callback` instances.
            model: The `Model` these callbacks are used with.
            **params: If provided, parameters will be passed to each `Callback`
                via `Callback.set_params`.
        """
        self.callbacks = callbacks if callbacks else []

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def set_model(self, model):
        """
    Sets the model for this object and all its callbacks.

    Args:
        model: The model to be set.

    Returns:
        None
    """
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def append(self, callback):
        """
    Appends a callback function to the list of callbacks.

    Args:
        callback (function): The callback function to be appended.

    Returns:
        None
    """
        self.callbacks.append(callback)

    def set_params(self, params):
        """
    Sets the parameters for this object and its callbacks.

    Args:
        params (dict): A dictionary of parameters to be set.

    Returns:
        None
    """
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, logs=None):
        """
    Called at the beginning of an epoch.

    Args:
        logs (dict, optional): Dictionary of logs. Defaults to None.

    Returns:
        None
    """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs=None):
        """
    Called at the end of an epoch.

    This method is a callback that is triggered after each epoch during training.
    It iterates over the list of callbacks and calls their on_epoch_end method,
    passing the logs dictionary.

    Args:
        logs (dict, optional): Dictionary of logs. Defaults to None.

    Returns:
        None
    """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(logs)

    def on_train_begin(self, logs=None):
        """
    Called at the beginning of training.

    Args:
        logs (dict, optional): Dictionary of logs. Defaults to None.

    Returns:
        None
    """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
    Called at the end of training.

    Args:
        logs (dict, optional): Dictionary of logs. Defaults to None.

    Returns:
        None
    """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
