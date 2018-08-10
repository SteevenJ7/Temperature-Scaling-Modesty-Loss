from util.dataset_loader import load_data
from util.network_init import init_network, load_network, post_process
from util.metrics import plot_diagrams
import torch
from torch.nn.functional import softmax


network_name = "AlexNet"  # AlexNet, VGGNet, ResNet
dataset = "CIFAR10"  # CIFAR10, CIFAR100, ImageNet, SVHN
batch_size = 10

training = "Ensemble"  # "Single", "Ensemble"
post_processing = "TS"  # None or "TS"
nb_network = 5

create_diagrams = True  # Compute RD
delta = 0.05

if __name__ == '__main__':

    print("Network :", network_name,
          "\nDataset : ", dataset,
          "\nTraining : ", training,
          "\nPost Processing : ", post_processing,
          "\nNb Network : ", nb_network)

    test_data = load_data(dataset, train=False, batch_size=batch_size)

    print("Initialisation du réseau ...")
    network = init_network(network_name, dataset, nb_network)
    network.eval()
    print("Initialisation terminé !")
    network = load_network(network_name, dataset, training, network)
    print("\nPost-Processing ...")
    network = post_process(network, test_data, post_processing)
    print("Post-Processing terminé !")

    print("\nCalcul des performances : \n")
    correct = 0
    confidence = 0
    bins = [0 for i in range((int(1 // delta) + 1))]
    acc_hist = [0 for i in range((int(1 // delta) + 1))]
    conf_hist = [0 for i in range((int(1 // delta) + 1))]
    for d, t in test_data:
        d, t = torch.autograd.Variable(d).cuda(), torch.autograd.Variable(t).cuda()
        net_out = network(d)

        prediction = net_out.data.max(1)[1]
        correct += prediction.eq(t.data).sum()
        proba = softmax(net_out, dim=1).data.max(1)[0].cpu().numpy()
        confidence += proba.sum()

        for idx in range(0, len(proba)):
            bins[int(proba[idx] // delta)] += 1
            acc_hist[int(proba[idx] // delta)] += (prediction.data[idx] == t.data[idx]).cpu().numpy()
            conf_hist[int(proba[idx] // delta)] += proba.data[idx]

    s = sum(bins)
    ece_value = 0

    accuracy = 100 * correct.cpu().numpy() / len(test_data.dataset)
    confidence = 100 * confidence / len(test_data.dataset)

    for index in range(0, len(bins)):
        acc_hist[index] /= (bins[index] + (bins[index] == 0) * 1)
        conf_hist[index] /= (bins[index] + (bins[index] == 0) * 1)
        bins[index] /= s
        ece_value += bins[index] * abs(acc_hist[index] - conf_hist[index])

    print("Acurracy = ", accuracy,
          "\nConfidence : ", confidence,
          "\nECE : ", ece_value)

    if create_diagrams:
        name = network_name+"_"+dataset+"_"+training+"_"+post_processing
        plot_diagrams(name, acc_hist, ece_value)
