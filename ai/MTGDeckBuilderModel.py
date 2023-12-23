import torch.nn as nn

class MTGDeckBuilderModel(nn.Module):
    def __init__(self, input_size, num_hidden_layer, output_size, device = "cpu"):
        super().__init__()
        # XXX include embedding layer?
        self.flatten = nn.Flatten()
        diff = int((input_size - output_size) / (num_hidden_layer+1))
        l = input_size - diff
        self.add_module("input_layer", nn.Linear(input_size, max(1,l), device=device))

        for k in range(num_hidden_layer):
            self.add_module(f"hidden_{k+1}",nn.Linear(max(1,l),max(1,l-diff), device=device))
            l -= diff
        self.add_module("output_layer",nn.Linear(max(1,l),output_size, device=device))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = self.layer(x)
            x = nn.ReLU(x)
        return x

