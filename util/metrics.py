from torch.autograd import Variable
from torch.nn.functional import softmax
from numpy import arange
import matplotlib.pyplot as plt


def get_log(network, test_data, delta = 0.05):
    """ Compute ECE, Average confidence and accuracy for the given
    network on the given dataset """
    network.eval()

    bins = [0] * (int(1 // delta) + 1)
    acc_hist = [0] * (int(1 // delta) + 1)
    conf_hist = [0] * (int(1 // delta) + 1)

    correct = 0
    confidence = 0

    for d, t in test_data:
        d, t = Variable(d).cuda(), Variable(t).cuda()
        net_out = network(d)
        prediction = net_out.data.max(1)[1]
        proba = softmax(net_out, dim=1).data.max(1)[0].cpu().numpy()
        for idx in range(0, len(proba)):
            bins[int(proba[idx] // delta)] += 1
            acc_hist[int(proba[idx] // delta)] += (prediction.data[idx] == t.data[idx]).cpu().numpy()
            conf_hist[int(proba[idx] // delta)] += proba.data[idx]

        confidence += proba.sum()
        correct += prediction.eq(t.data).sum()
    accuracy = 100 * correct.cpu().numpy() / len(test_data.dataset)
    confidence = 100 * confidence / len(test_data.dataset)

    s = sum(bins)
    ece = 0

    # Normalizations
    for index in range(0, len(bins)):
        acc_hist[index] /= (bins[index] + (bins[index] == 0))
        conf_hist[index] /= (bins[index] + (bins[index] == 0))
        bins[index] /= s
        ece += bins[index] * abs(acc_hist[index] - conf_hist[index])
    network.train()
    return ece, confidence, accuracy


def write_log(ece, confidence, accuracy, loss, name):
    """ Save log from training into text file"""
    with open(name+'.txt', 'w') as file:
        for i in range(ece):
            ligne = str(ece[i])+", "+str(confidence[i])+", "+str(accuracy[i])+", "+str(loss[i])+"\n"
            file.write(ligne)


def plot_diagrams(name, accuracy, ece,  delta=0.05):
    """ Plot Reliability diagram"""
    ind = arange(1 / delta) * delta
    plt.bar(ind, accuracy, delta - delta / 10)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title("RD for "+name)
    plt.text(0, 0.9, "ECE = " + str(round(ece, 5)))
    plt.plot([0, 1], [0, 1], c="red")
    plt.savefig("Graphics/RD_"+name+".png")

