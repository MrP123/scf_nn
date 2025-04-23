import torch
from torch import Tensor
import torch.nn.functional as F

import torchviz
import torchinfo


class RenderableModule(torch.nn.Module):
    """
    This class is intended as a custom base class foor all models, just like torch.nn.Module is usually used.
    It adds functions for rendering of the model architecture using torchviz and for printing model info with torchinfo.
    """

    def __init__(self, input_params):
        super().__init__()
        self.input_params = input_params

    def render(self, filepath: str = "models/model_test", save_to_file: bool = False) -> torchviz.dot.Digraph:
        """
        Renders the model architecture as a torchviz graph and saves it to file if desired.
        Returning the graph object allows it to be automatically rendered in a Jupyter notebook in VS Code, even when not saved to file.

        A model for which render is called must be a proper model with a forward() method.
        """

        x = torch.randn(1, self.input_params).to("cpu")
        y = self.to("cpu")(x)
        dot = torchviz.make_dot(y, params=dict(self.named_parameters()))

        if save_to_file:
            dot.render(filepath, format="png")

        return dot

    def info(self, batch_size: int = 8) -> None:
        print(torchinfo.summary(self, (batch_size, self.input_params)))


class NNJungEtAl(RenderableModule):
    """
    Model identical to reference Jung et al. 2020, [3] in README.md
    """

    def __init__(self, input_params, hidden_size: int = 20):
        super().__init__(input_params)

        self.hidden1 = torch.nn.Linear(input_params, hidden_size)
        self.output1 = torch.nn.Linear(hidden_size, 2)  # x, y coord as output

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden1(x)
        x = self.output1(F.tanh(x))
        return x


class MyNN(RenderableModule):
    """
    Model that extends the one presented by Jung et al. by an additional hidden layer.
    Not used further at the moment.
    """

    def __init__(self, input_params):
        super().__init__(input_params)

        self.hidden1 = torch.nn.Linear(input_params, 20)
        self.hidden2 = torch.nn.Linear(20, 10)
        self.output1 = torch.nn.Linear(10, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden1(x)
        x = self.hidden2(F.tanh(x))
        x = self.output1(F.tanh(x))
        return x


class NNYuEtAl(RenderableModule):
    """
    Model based on reference Yu et al. 2024, [1] in README.md
    Does not feature the additional attention mechanism present in the paper.
    Sizes if input/output tensors of layers are at the moment chosen almost arbitrarily

    The input tensor for this model is of shape (batch_size, 1, input_params), this is already reflected in the training loop.
    """

    def __init__(self, input_params):
        super().__init__(input_params)

        self.conv1 = torch.nn.Conv1d(1, 20, kernel_size=3, stride=1)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.mp = torch.nn.MaxPool1d(2)

        self.conv2 = torch.nn.Conv1d(20, 10, kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm1d(10)

        self.flatten = torch.nn.Flatten()

        self.lstm1 = torch.nn.LSTM(10, 20)
        self.lstm2 = torch.nn.LSTM(20, 20)
        self.fc = torch.nn.Linear(20, 2)  # X, Y coords as output

    def info(self, batch_size: int = 8) -> None:
        print(torchinfo.summary(self, (batch_size, 1, self.input_params)))

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = self.conv1(x)
        # print(f"After 1st Conv1D: {x.shape}")
        x = self.bn1(x)
        # print(f"After Batch norm: {x.shape}")
        x = F.leaky_relu(x)
        # print(f"After LeakyRELU: {x.shape}")
        x = self.mp(x)
        # print(f"After Max pool: {x.shape}")
        x = self.conv2(x)
        # print(f"After 2nd Conv1D: {x.shape}")
        x = self.bn2(x)
        # print(f"After Batch norm: {x.shape}")
        x = F.leaky_relu(x)
        # print(f"After LeakyRELU: {x.shape}")
        x = self.mp(x)
        # print(f"After Max pool: {x.shape}")
        x = self.flatten(x)
        # print(f"After Flatten: {x.shape}")
        x, (hn, cn) = self.lstm1(x)
        # print(f"After 1st LSTM: {x.shape}")
        x, (hn2, cn2) = self.lstm2(x)
        # print(f"After 2nd LSTM: {x.shape}")
        x = self.fc(x)
        # print(f"After Linear: {x.shape}")
        return x
