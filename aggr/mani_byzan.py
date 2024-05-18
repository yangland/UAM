# from Paper: Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning
# https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning
import torch

def our_attack_dist(all_updates, model_re, n_attackers, cuda_device, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([50.0]).float().to(cuda_device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    if model_re.dim() <= 1:
        return model_re

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update).double(), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation).double()
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update


def our_attack_score(all_updates, model_re, n_attackers, cuda_device, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([50.0]).float().to(cuda_device)
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    if model_re.dim() <= 1:
        return model_re

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation).double()
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print(lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)

    return mal_update