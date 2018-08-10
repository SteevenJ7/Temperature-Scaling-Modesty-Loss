import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from scipy.optimize import minimize
from math import pi, sqrt


class ModelWithTemperature(nn.Module):
    """Framework to quickly add and train Temperature scaling with modesty loss"""
    def __init__(self, model):
        """model : the network to perform TS on, can be an ensemble"""
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature.cuda()

    def modesty_loss(self, test_set):
        """Tune temperature parameter with modesty loss"""
        self.cuda()
        logits_list = []
        labels_list = []
        for index, (input, label) in enumerate(test_set):
            input_var = Variable(input).cuda()
            logits_var = self.model(input_var)
            logits_list.append(logits_var.data)
            labels_list.append(label)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        logits_var = Variable(logits)
        labels_var = Variable(labels)

        criterion = Modesty_loss()
        optimizer = torch.optim.Adam([self.temperature], lr=1e-2)

        def eval():
            loss = criterion(logits_var, labels_var, self.temperature)
            loss.backward()

            return loss

        l = 10
        iter = 0
        b = False
        b2 = False
        while l > 1e-10 and iter < 10000:
            iter += 1
            l = optimizer.step(eval)
            for param_group in optimizer.param_groups:
                if l<1e-4 and b==False:
                    param_group['lr'] = param_group['lr'] * 0.1
                    b = True
                if l<1e-6 and b2 == False:
                    param_group['lr'] = param_group['lr'] * 0.1
                    b2 = True
        print("Temperature = ", 1 / self.temperature)
        return self


class Modesty_loss(nn.Module):

    def __init__(self):
        super(Modesty_loss, self).__init__()

    def forward(self, inputs, label, T):
        (scores, prediction) = functional.softmax(inputs.data / T, dim=1).max(1)
        correct = prediction.eq(label.cuda()).sum()
        acc = (correct.float() / len(label))
        proba = torch.mean(scores)
        return (acc - proba) ** 2
