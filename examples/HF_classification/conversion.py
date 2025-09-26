from typing import Dict, List

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import torch

from .MLP import MLPNet


class Converter:
    """
    A class that consist of convertion functions for 
    transitioning between a neural network (NN) model 
    and a dictionary of np.array models. 
    
    Note: Singleton Pattern
    
    Attributes:
        order_list (list): saves the order of models 
        (using the predefined model names).
    """

    _singleton_cvtr = None

    @classmethod
    def cvtr(cls):
        if not cls._singleton_cvtr:
            cls._singleton_cvtr = cls()
        return cls._singleton_cvtr

    def __init__(self):
        self.order_list = list()

    def convert_nn_to_dict_nparray(self, net) -> Dict[str, np.array]:
        """
        Convert a neural network model to a dictionary of np.array models
        and save the order of models in a list (using the names).

        Args:
            net (Net class): a neural network (NN).

        Returns:
            d (Dict[str, np.array]): A dictionary of np.array models.
        """

        d = dict()

        # get a dictionary of all layers in the NN
        layers = vars(net)['_modules']

        # for each layer
        for lname, model in layers.items():
            # convert it to numpy
            if not lname == 'pool':
                for i, ws in enumerate(model.parameters()):
                    mname = f'{lname}_{i}'
                    d[mname] = ws.data.numpy()
                    self.order_list.append(mname)
        return d

    def convert_dict_nparray_to_nn(self, models: Dict[str, np.array]) -> MLPNet:
        """
        Convert np.array models in a dictionary to a neural network (NN) model. 

        Args:
            models (Dict[str, np.array]): A dictionary of np.array models.
            
        Returns: 
            net (MLPNet): A PyTorch Multilayer Perceptron (MLP) neural network (NN) with weights loaded from the dictionary.)  
        """
        
        for key, value in models.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Value for key '{key}' must be a NumPy array.")

        net = MLPNet()
        layers = vars(net)['_modules']

        npa_iter = iter(_order_dict(models, self.order_list))
        total_weights = sum(1 for lname, model in layers.items() 
        if lname != "pool" for _ in model.parameters())
        # for each layer
        try: 
            for lname, model in layers.items():
                if not lname == 'pool':
                    # for loop to separatly update w and b
                    for ws in model.parameters():
                        # Since the order is kept in NN
                        # update it
                        ws.data = torch.from_numpy(next(npa_iter))
        except StopIteration:
            raise ValueError(f"Not enough weight arrays in model dict to populate {total_weights} parameters.")
        
        return net

    def get_model_names(self, net) -> List[str]:
        """
        Return a list of suggested model names for the config file.

        Args:
            net (Net class): a Multilayer Perceptron (MLP) neural network (NN).   
            
        Returns: 
            d (List[str]): a list of suggested model names.
        """

        logging.info('=== Model Names (for the config file) ===')
        d = self.convert_nn_to_dict_nparray(net)
        logging.info(d.keys())

        return d.keys()

def _order_dict(d: Dict, l: List) -> List:
    """
    If the order of the list is not found or exists, 
    it will just use dictionary order.

    Args:
        d (Dict)   
        
    Returns: 
        l (List)
    """

    if not l:  
        logging.info("[Warning] No order list found. Using dict key order.")
        return list(d.values())
    return [d[key] for key in l]
