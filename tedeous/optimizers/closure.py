import torch
from typing import Any
from tedeous.device import device_type

class Closure():
    """
    A class that provides a closure for various optimization algorithms.

    The Closure class is responsible for managing the model, optimizer, and other
    relevant attributes, and provides methods for performing optimization steps,
    computing losses and gradients, and updating model parameters.

    Methods:
        __init__(mixed_precision, model): Initializes the object with the given mixed precision and model.
        set_model(model): Sets the model for the current object.
        model: Returns the internal model instance.
        _amp_mixed(mixed_precision): Prepares for mixed precision operations.
        _closure(): Performs a single optimization step by computing the loss, applying backpropagation, and updating the model parameters.
        _closure_pso(): Evaluates the loss and gradients for the particle swarm optimization.
        _closure_ngd(): Performs a single iteration of the neural gradient descent algorithm.
        _closure_nncg(): Computes the loss and gradients for the neural network.
        get_closure(_type): Retrieves the closure based on the provided type.

    Attributes:
        model: The internal model instance.
    """
    def __init__(self,
        mixed_precision: bool,
        model):

        """
    Initializes the object with the given mixed precision and model.

    Args:
        mixed_precision (bool): Whether to use mixed precision training.
        model: The model to be used.

    Returns:
        None
    """

        self.mixed_precision = mixed_precision
        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop
        self.device = device_type()
        self.cuda_flag = True if self.device == 'cuda' and self.mixed_precision else False
        self.dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        if self.mixed_precision:
            self._amp_mixed()


    def set_model(self, model):
        """
    Sets the model for the current object.

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

    def _amp_mixed(self):
        """ Preparation for mixed precsion operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        

    def _closure(self):
        """
    Performs a single optimization step by computing the loss, applying backpropagation, and updating the model parameters.

    Args:
        self: The instance of the class, providing access to the model, optimizer, and other relevant attributes.

    Returns:
        torch.Tensor: The computed loss value.
    """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss

    def _closure_pso(self):
        """
    Evaluates the loss and gradients for the particle swarm optimization.

    This method is a closure that computes the loss and gradients for each particle
    in the swarm, and returns the losses and gradients as tensors.

    Args:
        self: The instance of the class, which contains the model, optimizer, and other necessary attributes.

    Returns:
        tuple: A tuple containing two tensors:
            losses (torch.Tensor): A tensor of shape (num_particles,) containing the losses for each particle.
            gradients (torch.Tensor): A tensor of shape (num_particles, num_params) containing the gradients for each particle.
    """
        def loss_grads():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device,
                                dtype=self.dtype,
                                enabled=self.mixed_precision):
                loss, loss_normalized = self.model.solution_cls.evaluate()

            if self.optimizer.use_grad:
                grads = self.optimizer.gradient(loss)
                grads = torch.where(grads != grads, torch.zeros_like(grads), grads)
            else:
                grads = torch.tensor([0.])

            return loss, grads

        loss_swarm = []
        grads_swarm = []
        for particle in self.optimizer.swarm:
            self.optimizer.vec_to_params(particle)
            loss_particle, grads = loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1, -1))

        losses = torch.stack(loss_swarm).reshape(-1)
        
        gradients = torch.vstack(grads_swarm)

        self.model.cur_loss = min(loss_swarm)

        return losses, gradients
    
    def _closure_ngd(self):
        """
    Performs a single iteration of the neural gradient descent algorithm.

    This method is responsible for computing the loss, applying backpropagation,
    updating the model parameters, and computing the current loss value.

    Args:
        self: The instance of the class, providing access to the model, optimizer,
            and other necessary attributes.

    Returns:
        A tuple containing:
            - int_res: The result of the PDE computation.
            - bval: The boundary value.
            - true_bval: The true boundary value.
            - loss: The current loss value.
            - evaluate: The evaluation function of the model's solution class.
    """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=True)

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        int_res = self.model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = self.model.solution_cls.boundary.apply_bcs()

        return int_res, bval, true_bval, loss, self.model.solution_cls.evaluate

    def _closure_nncg(self):
        """
    Computes the loss and gradients for the neural network.

    This method is used to evaluate the model's performance and compute the gradients
    of the loss with respect to the model's parameters. It also updates the model's
    current loss value.

    Args:
        self: The instance of the class, which contains the model, optimizer, and other
            relevant attributes.

    Returns:
        A tuple containing:
            loss (torch.Tensor): The computed loss value.
            grads (torch.Tensor): The gradients of the loss with respect to the model's
                parameters.
    """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()

        



        # if self.optimizer.use_grad:
        grads = self.optimizer.gradient(loss)
        grads = torch.where(grads != grads, torch.zeros_like(grads), grads)

        #this fellow moved to model.py since it called several times a row
        #if (self.model.t-1) % self.optimizer.precond_update_frequency == 0: 
        #        print('here t={} and freq={}'.format(self.model.t-1,self.optimizer.precond_update_frequency))
        #        self.optimizer.update_preconditioner(grads)


        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss, grads



    def get_closure(self, _type: str):
        """
    Retrieves the closure based on the provided type.

    Args:
        _type (str): The type of closure to retrieve. Can be 'PSO', 'CSO', 'NGD', 'NNCG', or any other type.

    Returns:
        str: The closure corresponding to the provided type. If the type is not recognized, returns the default closure.
    """
        if _type in ('PSO', 'CSO'):
            return self._closure_pso
        elif _type == 'NGD':
            return self._closure_ngd
        elif _type == 'NNCG':
            return self._closure_nncg 
        else:
            return self._closure
