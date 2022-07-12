from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self, config={}, name=None, regions=None, to_numpy=True):
        super().__init__()
        # name of the module
        self.name = name if name is not None else self.__class__.__name__
        self.to_numpy = to_numpy  # if true, detach and convert pytorch tensors to numpy arrays when recording
        self.analyzing = False

    def analyze(self, mode=True):
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
        if self.analyzing:
            if self.to_numpy:
                tensor = tensor.detach().clone().cpu().numpy()
            if append:
                # append
                self.readout_dict[name].append(tensor)
            else:
                # replace
                self.readout_dict[name] = tensor

    def readout(self):
        if hasattr(self, 'readout_dict'):
            readout_dict = {}
            for key, value in self.readout_dict.items():
                # print(key)
                # for v in value:
                #     print(v.shape)
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
