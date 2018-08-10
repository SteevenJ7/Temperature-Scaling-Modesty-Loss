import torch.nn as nn


class Ensemble(nn.Module):
    """Framework that simplify the deployment of deep ensemble"""
    def __init__(self, networks):
        """networks : list of the members of the deep ensemble"""
        super(Ensemble, self).__init__()
        self.model = networks

    def forward(self, x):
        """Output the average of outputs of each member"""
        out = self.model[0](x)
        for i in range(1, len(self.model)):
            out += self.model[i](x)
        return out/len(self.model)

    def eval(self):
        for model in self.model:
            model.eval()

    def train(self, **kwargs):
        for model in self.model:
            model.train()
