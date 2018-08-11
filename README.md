# Temperature-Scaling-Modesty-Loss
Simple framework in pytorch using Temperature Scaling and Modesty Loss to improve calibration of deep neural networks

# Content
- Network/ :
    - AlexNet.py : Pytorch class for AlexNet Network
    - ResNet.py : Pytorch class for ResNet110 Network
    - VGGNet.py : Pytorch class for VGGNet11 with batch normalization
    - Ensemble.py : Framework for deep ensemble
    - temperature_scaling.py : Framework for Temperature Scaling
- util/:
    - dataset_loader.py : create dataloader for the desired dataset. Note : ImageNet is the ILSVRC 2012 validation dataset
    - loss_function.py : Pytorch class for an adaptation of Cross Entropy for deep ensemble
    - metrics.py : functions to compute ECE, Accuracy, save logs, etc...
    - network_init.py : functions to create, load, save and modify a network
- train.py : To train a network, simply modify parameters at the beginning of the file and run the code
- test.py : Compute Accuracy, ECE, and Average Confidence. Possible to plot Reliability Diagram. Simply modify parameters at the beginning of the file and run the code.

# Note
Modesty Loss is an improvment of Temperature Scaling proposed by :https://github.com/gpleiss/temperature_scaling
Pretrained AlexNet networks are available as PoC. ImageNet option in test.py load the pretrained networks given by Pytorch.
