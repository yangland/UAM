# UAM - Federated Learning Attack with Nelder_Mead method
Code for the ECCV paper (May 2024):
* Exploiting the Properties of Adaptive Black-box Attack Against Federated Learning

### Datasets:
* MNIST
* EMNIST
* CIFAR-10
* Fashion-MNIST
* CIFAR-100
* Imagenette (10 class subset for ImageNet), require download dataset files from https://github.com/fastai/imagenette  into "./data"

### Single Experiment
```
python main.py --params ./utils/mnist_3l_single.yaml
```
### Series Experiments
The parameters file in the utils/series folder
```
python main.py --params ./utils/series/mnist_3l_series.yaml
python main.py --params ./utils/series/imagenette_series.yaml
```
Editing yaml file for the needed configuration.