# Attack Federated Learning from its aggregation rule (Sep 2023)
Code for the ICRL paper:
* Attack Federated Learning from its aggregation rule

### Datasets:
* MNIST
* EMNIST
* Cifar10
* Fashion-MNIST

### Series Experiments
The parameters file in the utils/series folder
```
python main.py --params .\utils\series\mnist_3l_params_rl_rev_grad_series.yaml
python main.py --params .\utils\series\fashion_mnist_params_rl_rev_grad_series.yaml
python main.py --params .\utils\series\emnist_params_rl_rev_grad_series.yaml
python main.py --params .\utils\series\cifar10_params_rl_rev_grad_series.yaml
```
Normal parameter file to run one FL experiment in the utils folder, edit yaml file for needed setup