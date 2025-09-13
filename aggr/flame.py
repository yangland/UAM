import torch
import numpy as np
import hdbscan
from copy import deepcopy
import torch.nn.functional as F
import sys
import os


# from visualization import matrix_plot
# Manually specify the path to your project folder
project_path = os.path.abspath('..')  # Or provide the full path if needed
sys.path.append(project_path)
import logging
logger = logging.getLogger("individual_logging")
logger.addHandler(logging.StreamHandler())


def flame_aggr(server_model_state_dict, client_updates_dict, noise=0.001, exp_dir="", iter=0, **kwargs):
    client_ids = list(client_updates_dict.keys())
    reveled_model_dict = {}
    
    # Convert gradients to model states (current approach)
    for client_id in client_ids:
        reveled_model_dict[client_id] = get_model_merged(client_updates_dict[client_id], server_model_state_dict)
        
    # Get updated server state
    new_server_sd, flame_clients = flame(server_model_state_dict, reveled_model_dict, noise, client_ids, exp_dir, iter)
    
    # Return the gradient update (difference between new and old server state)
    server_grad = get_model_update(new_server_sd, server_model_state_dict)
    
    client_weights = {
        cid: (1.0 / len(flame_clients) if cid in flame_clients else 0.0)
        for cid in client_updates_dict
    }

    return server_grad, client_weights


def flame(server_sd, model_dict, noise=0.001, client_ids=None, exp_dir="", iter=0):

    # --- normalize server_sd to a plain state_dict ---
    if isinstance(server_sd, tuple):              # e.g., (state_dict, ...)
        server_sd = server_sd[0]
    if hasattr(server_sd, "state_dict"):          # nn.Module
        server_sd = server_sd.state_dict()
    # FIX: keep a detached, cloned copy (don't overwrite later)
    server_sd_copy = {k: v.detach().clone() for k, v in server_sd.items()}

    cos_list = []
    local_model_vector = []
    update_params = []
    local_client_ids = [] if client_ids is None else []  # we rebuild from model_dict
    num_clients = len(model_dict)

    # Build flattened models & updates
    for cid in sorted(model_dict.keys()):
        param_sd = model_dict[cid]     # dict of tensors (client model or delta dict)
        local_model_vector.append(modelsd2flat(param_sd))
        update_params.append(get_model_update(param_sd, server_sd))  # DIFF = client - server
        local_client_ids.append(cid)

    # Sanitize local_model_vector
    for i in range(len(local_model_vector)):
        if torch.isnan(local_model_vector[i]).any() or torch.isinf(local_model_vector[i]).any():
            logger.info(f"FLAME: NaNs/Infs in local_model_vector[{i}], replacing with zeros")
            local_model_vector[i] = torch.nan_to_num(local_model_vector[i], nan=0.0, posinf=0.0, neginf=0.0)

    # Pairwise cosine distance matrix for HDBSCAN (NumPy array)
    for i in range(len(local_model_vector)):
        row = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - F.cosine_similarity(local_model_vector[i], local_model_vector[j], dim=0)
            row.append(cos_ij.item())
        cos_list.append(row)
    cos_mat = np.asarray(cos_list, dtype=np.float64)
    if np.isnan(cos_mat).any():
        logger.info("FLAME: NaN detected in cos matrix; replacing with zeros")
        cos_mat = np.nan_to_num(cos_mat, nan=0.0)

    # Cluster to pick benign subset
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=num_clients // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(cos_mat)

    logger.info(f"FLAME: clusterer.labels_ {str(clusterer.labels_)}")

    benign_idx = []
    if clusterer.labels_.max() < 0:
        benign_idx = list(range(num_clients))
    else:
        # largest cluster
        labels = clusterer.labels_
        max_cluster = None
        max_size = -1
        for c in range(labels.max() + 1):
            size = (labels == c).sum()
            if size > max_size:
                max_size = size
                max_cluster = c
        benign_idx = [i for i in range(num_clients) if labels[i] == max_cluster]

    # Norm list & clipping
    norm_list = []
    for i in range(num_clients):
        flat = modelsd2flat(update_params[i])
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            logger.warning("FLAME: NaN/Inf in update; replacing with zeros for norm calc")
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norm_list.append(flat.norm(p=2).item())
    norm_list = np.array(norm_list, dtype=np.float64)

    if np.isnan(norm_list).any():
        norm_list = norm_list[~np.isnan(norm_list)]
    if norm_list.size == 0:
        raise ValueError("FLAME: norm_list is empty after NaN removal")

    clip_value = np.median(norm_list)
    for idx in benign_idx:
        gamma = clip_value / (norm_list[idx] + 1e-15)
        if gamma < 1.0:
            for k, t in update_params[idx].items():
                if k.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[idx][k] = t * gamma

    logger.info(f"[FLAME DEBUG][Iter {iter}] Norm list: {norm_list}")
    logger.info(f"[FLAME DEBUG][Iter {iter}] Clip value: {clip_value}")

    # Average deltas of benign subset
    avg_update, _ = fedavg(
        server_model_state_dict=server_sd_copy,
        client_updates_dict={local_client_ids[i]: update_params[i] for i in benign_idx},
        average_bn_buffers=True
    )

    # FIX: avg_update is a DELTA dict -> merge as delta
    new_server_sd = get_model_merged(avg_update, server_sd_copy, assume_delta=True)

    # NOTE: never corrupt BN buffers with noise; only apply noise to weights/biases
    with torch.no_grad():
        for key, var in new_server_sd.items():
            suffix = key.split('.')[-1]
            if suffix in ('num_batches_tracked',):
                continue
            if ('running_mean' in key) or ('running_var' in key):
                continue  # <-- DO NOT NOISE BN buffers
            # small noise
            var.add_(torch.normal(mean=0.0, std=noise, size=var.shape, device=var.device))

    # Extra safety: ensure BN running_var stays positive
    with torch.no_grad():
        for key, var in new_server_sd.items():
            if 'running_var' in key:
                var.clamp_(min=1e-6)

    benign_client_ids = [local_client_ids[i] for i in benign_idx]
    return new_server_sd, benign_client_ids


def fedavg(server_model_state_dict, client_updates_dict, average_bn_buffers=True, **kwargs):
    """
    Federated averaging of model parameters.
    """
    # print("\n[DEBUG] Entered fedavg()")
    
    # Print types of inputs
    # print(f"[DEBUG] server_model_state_dict type: {type(server_model_state_dict)}")
    # print(f"[DEBUG] client_updates_dict type: {type(client_updates_dict)}")
    
    # Print a few keys of the server model
    if isinstance(server_model_state_dict, dict):
        # print(f"[DEBUG] server_model_state_dict keys (sample): {list(server_model_state_dict.keys())[:3]}")
        pass
    
    # Print client_updates_dict summary
    if isinstance(client_updates_dict, dict):
        # print(f"[DEBUG] Number of clients: {len(client_updates_dict)}")
        first_client = next(iter(client_updates_dict))
        # print(f"[DEBUG] First client ID: {first_client}")
        # print(f"[DEBUG] First client update type: {type(client_updates_dict[first_client])}")
        # print(f"[DEBUG] First client update keys (sample): {list(client_updates_dict[first_client].keys())[:3]}")
    else:
        # print("[ERROR] client_updates_dict is not a dict!")
        pass

    # Turn updates into a list
    model_list = list(client_updates_dict.values())
    
    # Check what's inside model_list
    if not model_list:
        raise ValueError("[ERROR] model_list is empty!")

    if not isinstance(model_list[0], dict):
        raise TypeError(f"[ERROR] Expected model_list[0] to be a dict, got {type(model_list[0])}")

    # Start aggregation
    avg_model = {}

    for key in model_list[0].keys():
        if 'num_batches_tracked' in key:
            avg_model[key] = model_list[0][key].clone()
        elif any(buf in key for buf in ['running_mean', 'running_var']):
            if average_bn_buffers:
                avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)
        else:
            avg_model[key] = torch.stack([m[key] for m in model_list]).mean(0)

    # print("[DEBUG] Aggregation complete. Returning averaged model.")
    
    # model weights for clients
    n_clients = len(client_updates_dict)
    client_weights = {cid: 1.0 / n_clients for cid in client_updates_dict}
    return avg_model, client_weights



def get_model_update(updated_model, model):
    """
    Returns the difference between updated_model and model (server),
    while preserving 'num_batches_tracked' and other non-learnable buffers.
    """
    update = {}
    for key in updated_model:
        if key.endswith('num_batches_tracked'):
            # Preserve the original value from the server model
            update[key] = model[key].detach().clone()
        else:
            update[key] = updated_model[key] - model[key].detach()
    return update


def get_model_merged(gradient_update, base_model, assume_delta=True):
    # normalize base_model to a state_dict
    if isinstance(base_model, tuple):
        base_model = base_model[0]
    if hasattr(base_model, "state_dict"):
        base_sd = base_model.state_dict()
    elif isinstance(base_model, dict):
        base_sd = base_model
    else:
        raise TypeError(f"Unsupported base_model type: {type(base_model)}")

    merged = {}
    for key, base_tensor in base_sd.items():
        if key.endswith('num_batches_tracked'):
            merged[key] = base_tensor.detach().clone()
        else:
            if assume_delta:
                merged[key] = base_tensor.detach() + gradient_update[key]
            else:
                merged[key] = gradient_update[key]  # already absolute
    return merged


def modelsd2flat(model_dict):
    """
    Flattens a model's state dictionary into a single 1D tensor.
    Skips non-trainable buffers like 'num_batches_tracked' for consistency.
    """
    ravel_list = []
    for layer_name, parms in model_dict.items():
        if isinstance(parms, torch.Tensor) and not layer_name.endswith('num_batches_tracked'):
            ravel_list.append(parms.detach().reshape(-1))
    
    if not ravel_list:
        return torch.tensor([])
    
    flat_tensor = torch.cat(ravel_list)
    return flat_tensor