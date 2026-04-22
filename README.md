# UAM: Adaptive Black-Box Attack Against Federated Learning

This repository contains the public implementation for the ICASSP 2026 paper:

**Exploiting the Properties of an Adaptive Black-box Attack Against Federated Learning**

UAM is an AGR-agnostic untargeted federated-learning attack. Malicious clients send diverse sign-reversed local-gradient updates, while a one-dimensional Nelder-Mead controller adapts the shared attack magnitude.

## Paper Setup

The public release is scoped to the experiments reported in the paper:

- Datasets: MNIST and CIFAR-10
- Clients: 100 total clients, 30 malicious clients
- Round sampling: 10 clients per round with a fixed 30% malicious ratio
- Partitioning: Dirichlet non-IID splits
- Models: 3-layer MLP for MNIST and ResNet-20 for CIFAR-10
- Local training: SGD, momentum 0.9, 2 local epochs
- UAM controller: logistic conversion with `(z_a, z_b, z_c) = (10, 0.7, 3.1)`, empirical controls `[-4, 0, 4]`, Nelder-Mead step size `1.0`, restart threshold `R=3`

Note: internally the code records Nelder-Mead feedback as `acc_t - acc_{t-1}` and minimizes it. This is equivalent to maximizing the paper reward `acc_{t-1} - acc_t`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation can depend on your CUDA version. If the generic requirement does not match your machine, install the appropriate PyTorch build from https://pytorch.org/get-started/locally/ and then install the remaining requirements.

## Running Experiments

Each YAML file is a series config for one aggregation rule and attack/baseline set. Results are written under `logs/`.

MNIST, alpha = 0.1:

```bash
python main.py --params utils/mnist/mnist_fedavg.yaml
python main.py --params utils/mnist/mnist_foolsgold.yaml
python main.py --params utils/mnist/mnist_contra.yaml
python main.py --params utils/mnist/mnist_med.yaml
python main.py --params utils/mnist/mnist_geo.yaml
python main.py --params utils/mnist/mnist_krum.yaml
python main.py --params utils/mnist/mnist_flame.yaml
```

CIFAR-10, alpha = 1.0:

```bash
python main.py --params utils/cifar10_alpha1/cifar10_fedavg.yaml
python main.py --params utils/cifar10_alpha1/cifar10_foolsgold.yaml
python main.py --params utils/cifar10_alpha1/cifar10_contra.yaml
python main.py --params utils/cifar10_alpha1/cifar10_med.yaml
python main.py --params utils/cifar10_alpha1/cifar10_geo.yaml
python main.py --params utils/cifar10_alpha1/cifar10_krum.yaml
python main.py --params utils/cifar10_alpha1/cifar10_flame.yaml
```

CIFAR-10, alpha = 5.0:

```bash
python main.py --params utils/cifar10_alpha5/cifar10_fedavg.yaml
python main.py --params utils/cifar10_alpha5/cifar10_foolsgold.yaml
python main.py --params utils/cifar10_alpha5/cifar10_contra.yaml
python main.py --params utils/cifar10_alpha5/cifar10_med.yaml
python main.py --params utils/cifar10_alpha5/cifar10_geo.yaml
python main.py --params utils/cifar10_alpha5/cifar10_krum.yaml
python main.py --params utils/cifar10_alpha5/cifar10_flame.yaml
```

MNIST ablation configs are under `utils/ablation_mnist/`.

## Config Names

The implementation uses these internal names:

- UAM: `UAM`
- No attack: `None`
- LIE: `little`
- EB: `explicit boosting`
- MinMax: `min-max`
- MinSum: `min-sum`
- RFA/geometric median: `geom_median`
- Krum: `single-krum`
- Median: `coordinate_wise_median`

## Citation

The citation will be added after the ICASSP 2026 proceedings entry is available.
