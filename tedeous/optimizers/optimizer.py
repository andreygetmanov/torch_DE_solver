import torch
from abc import ABC
from typing import Union, Any
from tedeous.optimizers.pso import PSO
from tedeous.optimizers.ngd import NGD
from tedeous.optimizers.CSO import CSO
from tedeous.optimizers.nys_newton_cg import NysNewtonCG
from torch.optim.lr_scheduler import ExponentialLR


class Optimizer():
    """
    A class for managing and configuring optimizers for machine learning models.

    The Optimizer class provides a flexible way to select and configure different
    optimization algorithms for training machine learning models. It supports a range
    of built-in optimizers and allows for custom optimizer definitions.

    Methods:
        __init__(optimizer, params, gamma=None, decay_every=None):
            Initializes the object with the given optimizer, parameters, and learning rate decay settings.
        optimizer_choice(optimizer, learning_rate):
            Sets the optimizer and returns a ready-to-use optimizer object.

    Attributes:
        optimizer (str): The name of the optimizer to use.
        params (dict): A dictionary of parameters for the optimizer.
        gamma (float, optional): The learning rate decay factor. Defaults to None.
        decay_every (int, optional): The number of steps between learning rate decays. Defaults to None.
    """
    def __init__(
            self,
            optimizer: str,
            params: dict,
            gamma: Union[float, None]=None,
            decay_every: Union[int, None]=None):
        """
    Initializes the object with the given optimizer, parameters, and learning rate decay settings.

    Args:
        optimizer (str): The name of the optimizer to use.
        params (dict): A dictionary of parameters for the optimizer.
        gamma (Union[float, None], optional): The learning rate decay factor. Defaults to None.
        decay_every (Union[int, None], optional): The number of steps between learning rate decays. Defaults to None.

    Returns:
        None
    """
        self.optimizer = optimizer
        self.params = params
        self.gamma = gamma
        self.decay_every = decay_every

    def optimizer_choice(
        self,
        mode,
        model) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO, CSO, NysNewtonCG]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """

        if self.optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer == 'NNCG':
            torch_optim = NysNewtonCG
        elif self.optimizer == 'PSO':
            torch_optim = PSO
        elif self.optimizer == 'NGD':
            torch_optim = NGD
        elif self.optimizer == 'CSO':
            torch_optim = CSO


        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)
        
        if self.gamma is not None:
            self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return optimizer