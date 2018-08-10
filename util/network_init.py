import torchvision.models as models
from Network.AlexNet import AlexNet
from Network.VGGNet import VGGNet
from Network.ResNet import ResNet
from Network.ensemble import Ensemble
from Network.temperature_scaling import ModelWithTemperature
import torch


def init_network(name, d, nb_net):
    """ Return network that correspond to the desired configuration """
    if d == "ImageNet":
        if nb_net != 1:
            raise TypeError("ImageNet et Ensemble de "+str(nb_net)+" incompatibles")
        else :
            if name == "AlexNet":
                return models.alexnet(pretrained=True).cuda()
            elif name == "VGGNet":
                return models.vgg11_bn(pretrained=True).cuda()
            elif name == "ResNet":
                return models.resnet101(pretrained=True).cuda()
            else:
                raise TypeError("Nom du réseau inconnu : " + name)

    elif d == "CIFAR10" or d == "SVHN":
        if name == "AlexNet":
            return Ensemble([AlexNet(num_classes=10).cuda() for k in range(nb_net)])
        elif name == "VGGNet":
            return Ensemble([VGGNet(num_classes=10).cuda() for k in range(nb_net)])
        elif name == "ResNet":
            return Ensemble([ResNet(num_classes=10).cuda() for k in range(nb_net)])
        else:
            raise TypeError("Nom du réseau inconnu : " + name)
    else:
        if name == "AlexNet":
            return Ensemble([AlexNet().cuda() for k in range(nb_net)])
        elif name == "VGGNet":
            return Ensemble([VGGNet().cuda() for k in range(nb_net)])
        elif name == "ResNet":
            return Ensemble([ResNet().cuda() for k in range(nb_net)])
        else:
            raise TypeError("Nom du réseau inconnu : " + name)


def load_network(name, dataset, training, network):
    """ Load network's weight from saved network"""

    print("\nChargement du réseau...\n")
    
    if dataset == "ImageNet":
        print("Chargement Terminé !")
        return network
    else:
        chemin = "Trained_Networks/" + dataset + "/" + name + "/" + training
        for i, model in enumerate(network.model):
            model.load_state_dict(torch.load(chemin+"/Net_"+str(i)))
        print("Chargement Terminé !")
        return network


def post_process(network, test_data, post_processing):
    """ Apply post-processing to the network"""
    if post_processing:
        net = ModelWithTemperature(network)
        if post_processing == "TS":
            net.modesty_loss(test_data)
        else:
            raise TypeError("Post Processing inconnnu : ", post_processing)
        return net
    else:
        return network


def save_network(network, chemin):
    """ Save Network weights"""
    for i, model in enumerate(network.model):
        torch.save(model.state_dict(), chemin+"/Net_" + str(i))
