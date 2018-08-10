import torch.nn as nn


class CrossEntropyEnsemble(nn.Module):
    """ Implementation of CrossEntropy that make the training of
    Deep ensemble easier. Work only with Ensemble even if there is
    only one network in the ensemble """

    def __init__(self):
        super(CrossEntropyEnsemble, self).__init__()

    def forward(self, t, *args):
        ce = nn.CrossEntropyLoss().cuda()
        loss = 0
        for net in args:
            loss += ce(net, t)
        return loss