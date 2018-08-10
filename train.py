from util.dataset_loader import load_data
from util.network_init import init_network, save_network
from util.loss_function import CrossEntropyEnsemble
import torch
from torch.autograd import Variable
from util.metrics import get_log, write_log

####################################################
# Train a neural network using selected parameters #
####################################################


network_name = "VGGNet"  # AlexNet, VGGNet, ResNet
dataset = "CIFAR10"  # CIFAR10, CIFAR100, ImageNet, SVHN
batch_size = 10
nb_network = 1  # >1 to train a deep Ensemble

gamma = 1
lr = 0.001
epochs = 120

save_log = False  # Compute the evolution of Accuracy, average confidence and ECE a each epoch
save_net = False  # Save the trained network

if __name__ == '__main__':

    print("Network :", network_name,
          "\nDataset : ", dataset,
          "\nGamma : ", gamma,
          "\nNb Network : ", nb_network,
          "\nNb Epochs : ", epochs)

    train_data = load_data(dataset, train=True, batch_size=batch_size)
    if save_log:
        test_data = load_data(dataset, train=False, batch_size=batch_size)
        ece = []
        confidence = []
        accuracy = []
        losses = []

    # Initialize the network
    network = init_network(network_name, dataset, nb_network)
    network.train()

    # Set up the training process
    criterion = CrossEntropyEnsemble().cuda()
    optimizers = [torch.optim.Adam(net.parameter(), lr=lr) for net in network.model]

    # TRAINING
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            for optim in optimizers:
                optim.zero_grad()

            outputs = [netw(data).cuda() for netw in network.model]
            loss = criterion(target, *outputs)
            loss.backward()

            for i in range(0, nb_network):
                optimizers[i].step()  # Optimize

            # Monitor training
            if batch_idx == 0:
                print('Step ', epoch+1, ', Loss : ', loss.data.cpu().numpy())
            elif batch_idx % 50 == 0:
                print("\t\tLoss : ", loss.data.cpu().numpy())

        # Compute metrics
        if save_log:
            e, c, a = get_log(network, test_data)
            ece.append(e)
            confidence.append(c)
            accuracy.append(a)
            losses.append(loss.data.cpu().numpy())
            print("\tSave e=", e, ", c=", c, ", a=", a)

    # Save Network weights
    if save_log:
        name = network_name+"_"+dataset
        write_log(ece, confidence, accuracy, losses, name)

    if save_network:
        if nb_network > 1:
            path = "Trained_networks/" + dataset + "/" + network_name + "/Ensemble"
        else:
            path = "Trained_networks/" + dataset + "/" + network_name + "/Single"
        save_network(network, path)
