# from Paper: Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning
# https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning
import torch
import torch.nn.functional as F


def our_attack_dist(all_updates, model_re, n_attackers, cuda_device, dev_type='unit_vec'):
    # ---- dtype/device hygiene ----
    dtype = torch.float32
    all_updates = all_updates.to(device=cuda_device, dtype=dtype)   # [N, D]
    model_re    = model_re.to(device=cuda_device,    dtype=dtype)   # [D]

    # ---- deviation direction ----
    if dev_type == 'unit_vec':
        deviation = F.normalize(model_re, dim=0, eps=1e-12)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        dev = torch.std(all_updates, dim=0, unbiased=False)
        deviation = F.normalize(dev, dim=0, eps=1e-12)
    else:
        deviation = F.normalize(model_re, dim=0, eps=1e-12)

    # Scalars as tensors on the same device/dtype
    lamda         = torch.tensor(50.0, device=cuda_device, dtype=dtype)
    threshold_diff= 1e-5
    lamda_fail    = lamda.clone()
    lamda_succ    = torch.tensor(0.0, device=cuda_device, dtype=dtype)

    if model_re.ndim <= 1:
        return model_re

    # ---- max pairwise distance among benign updates (vectorized) ----
    # distances[i,j] = ||all_updates[i] - all_updates[j]||^2
    diffs = all_updates.unsqueeze(1) - all_updates.unsqueeze(0)      # [N, N, D]
    distances = torch.linalg.vector_norm(diffs, dim=2).pow(2)        # [N, N]
    max_distance = distances.max()
    del diffs, distances

    # ---- binary search for lamda ----
    while (lamda - lamda_succ).abs().item() > threshold_diff:
        mal_update = model_re - lamda * deviation                     # [D]
        # distance[i] = ||all_updates[i] - mal_update||^2
        distance = torch.linalg.vector_norm(all_updates - mal_update, dim=1).pow(2)  # [N]
        max_d = distance.max()

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    return model_re - lamda_succ * deviation


def our_attack_score(all_updates, model_re, n_attackers, cuda_device, dev_type='unit_vec'):
    # --- dtype/device hygiene ---
    dtype = torch.float32
    all_updates = all_updates.to(device=cuda_device, dtype=dtype)  # [N, D]
    model_re    = model_re.to(device=cuda_device,    dtype=dtype)  # [D]

    # --- deviation direction ---
    if dev_type == 'unit_vec':
        deviation = F.normalize(model_re, dim=0, eps=1e-12)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        dev = torch.std(all_updates, dim=0, unbiased=False)
        deviation = F.normalize(dev, dim=0, eps=1e-12)
    else:
        deviation = F.normalize(model_re, dim=0, eps=1e-12)

    lamda          = torch.tensor(50.0, device=cuda_device, dtype=dtype)
    threshold_diff = 1e-5
    lamda_fail     = lamda.clone()
    lamda_succ     = torch.tensor(0.0, device=cuda_device, dtype=dtype)

    if model_re.ndim <= 1:
        return model_re

    # --- compute per-client Krum-like scores (vectorized) ---
    diffs = all_updates.unsqueeze(1) - all_updates.unsqueeze(0)             # [N, N, D]
    distances = torch.linalg.vector_norm(diffs, dim=2).pow(2)               # [N, N]
    scores = distances.sum(dim=1)                                           # [N]
    min_score = scores.min()
    del diffs, distances, scores

    # --- binary search on lambda to keep score <= min_score ---
    while (lamda - lamda_succ).abs().item() > threshold_diff:
        mal_update = model_re - lamda * deviation                            # [D]
        d = torch.linalg.vector_norm(all_updates - mal_update, dim=1).pow(2) # [N]
        score = d.sum()

        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    return model_re - lamda_succ * deviation
