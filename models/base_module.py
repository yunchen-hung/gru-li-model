from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self, name=None, to_numpy=True):
        super().__init__()
        # name of the module
        self.name = name if name is not None else self.__class__.__name__
        self.to_numpy = to_numpy    # if true, detach and convert pytorch tensors to numpy arrays when recording
        self.analyzing = False      # if true, record activities of the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def analyze(self, mode=True):
        """
        set up dict for recording activities
        """
        self.analyzing = mode
        if mode:
            self.readout_dict = defaultdict(list)
        for module in self.children():
            try:
                module.analyze(mode)
            except AttributeError:
                pass
        return self

    def write(self, tensor, name, append=True):
        """
        record a tensor to the readout_dict with a given name
        """
        if self.analyzing:
            if self.to_numpy and torch.is_tensor(tensor):
                tensor = tensor.detach().to(self.device).numpy()
            if append:
                # append
                self.readout_dict[name].append(tensor)
            else:
                # replace
                self.readout_dict[name] = tensor

    def readout(self):
        """
        process the readout_dict and return a dict of recorded data
        """
        if hasattr(self, 'readout_dict'):
            readout_dict = {}
            for key, value in self.readout_dict.items():
                if self.to_numpy:
                    readout_dict[key] = np.stack(value, axis=0)
                else:
                    readout_dict[key] = torch.stack(value, axis=0)
            
            for child in self.children():
                if isinstance(child, BasicModule):
                    readout_dict[child.name] = child.readout()

            return readout_dict
        else:
            return {}


class analyze:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.analyze(True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.analyze(False)
