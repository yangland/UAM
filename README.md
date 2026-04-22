# UAM: Adaptive Black-Box Attack Against Federated Learning

Code for the ICASSP 2026 paper:

**Exploiting the Properties of an Adaptive Black-box Attack Against Federated Learning**

## Overview

UAM (Unified Adaptive Magnitude) is an aggregation-rule-agnostic (AGR-agnostic) untargeted attack in federated learning. Malicious clients generate diverse sign-reversed local gradient updates, while a lightweight one-dimensional controller dynamically adjusts the shared attack magnitude.

The controller uses a Nelder–Mead optimization process to adaptively search for effective scaling, enabling strong attack performance without knowledge of the server aggregation rule.

## Paper Setup

The public release is scoped to the experimental settings reported in the paper:

* **Datasets:** MNIST, CIFAR-10
* **Clients:** 100 total, 30 malicious
* **Sampling:** 10 clients per round with fixed 30% malicious ratio
* **Data split:** Dirichlet non-IID partitioning
* **Models:**

  * MNIST: 3-layer MLP
  * CIFAR-10: ResNet-20
* **Local training:** SGD, momentum = 0.9, 2 local epochs
* **UAM controller:**

  * Logistic conversion: `(z_a, z_b, z_c) = (10, 0.7, 3.1)`
  * Empirical controls: `[-4, 0, 4]`
  * Optimization: Nelder–Mead, step size = 1.0
  * Restart threshold: `R = 3`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation depends on the CUDA version. If the default requirement does not match your system, install the appropriate build from:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then install the remaining dependencies.

## Run Experiments

Each YAML file defines a full experiment configuration (aggregation rule + attack/baselines). Results are written to `logs/`.

### MNIST (Dirichlet α = 0.1)

```bash
python main.py --params utils/mnist/mnist_fedavg.yaml
python main.py --params utils/mnist/mnist_foolsgold.yaml
python main.py --params utils/mnist/mnist_contra.yaml
python main.py --params utils/mnist/mnist_med.yaml
python main.py --params utils/mnist/mnist_geo.yaml
python main.py --params utils/mnist/mnist_krum.yaml
python main.py --params utils/mnist/mnist_flame.yaml
```

### CIFAR-10 (Dirichlet α = 1.0)

```bash
python main.py --params utils/cifar10_alpha1/cifar10_fedavg.yaml
python main.py --params utils/cifar10_alpha1/cifar10_foolsgold.yaml
python main.py --params utils/cifar10_alpha1/cifar10_contra.yaml
python main.py --params utils/cifar10_alpha1/cifar10_med.yaml
python main.py --params utils/cifar10_alpha1/cifar10_geo.yaml
python main.py --params utils/cifar10_alpha1/cifar10_krum.yaml
python main.py --params utils/cifar10_alpha1/cifar10_flame.yaml
```

### CIFAR-10 (Dirichlet α = 5.0)

```bash
python main.py --params utils/cifar10_alpha5/cifar10_fedavg.yaml
python main.py --params utils/cifar10_alpha5/cifar10_foolsgold.yaml
python main.py --params utils/cifar10_alpha5/cifar10_contra.yaml
python main.py --params utils/cifar10_alpha5/cifar10_med.yaml
python main.py --params utils/cifar10_alpha5/cifar10_geo.yaml
python main.py --params utils/cifar10_alpha5/cifar10_krum.yaml
python main.py --params utils/cifar10_alpha5/cifar10_flame.yaml
```

### Ablation (MNIST)

Ablation configurations are provided under:

```
utils/ablation_mnist/
```

## Config Naming

Internal identifiers used in the implementation:

| Method                 | Name                     |
| ---------------------- | ------------------------ |
| UAM                    | `UAM`                    |
| No attack              | `None`                   |
| LIE                    | `little`                 |
| Explicit Boosting      | `explicit boosting`      |
| Min-Max                | `min-max`                |
| Min-Sum                | `min-sum`                |
| Geometric Median (RFA) | `geom_median`            |
| Krum                   | `single-krum`            |
| Coordinate-wise Median | `coordinate_wise_median` |

## Notes

* The implementation focuses on reproducibility of reported results rather than providing a generalized FL framework.
* Attack behavior depends on interaction with the aggregation rule; results may vary under modified settings.
* Logs include accuracy and attack effectiveness metrics for each run.

## Citation

Citation information will be added after the ICASSP 2026 proceedings entry is available.