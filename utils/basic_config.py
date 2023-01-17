
class BasicConfig:
    def __init__(self):
        pass

    def update(self, config):
        self.__dict__.update(config.__dict__)

    def __str__(self):
        return str(self.__dict__)


class ModelConfig(BasicConfig):
    def __init__(self):
        # self.model_class = "ValueMemoryCTRNN"
        # self.hidden_dim = 64
        # self.input_dim = 21
        # self.output_dim = 21
        pass

