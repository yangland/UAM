import numpy as np
import torch
import heapq
import itertools

def get_layer_weight(model, name_list):
    splitted = name_list.split('.')
    result = model
    for p in splitted:
        result = getattr(result, p)
    return result


def coordinate_wise_median(model_dict):
    new_weights = dict()
    name_of_models = list(model_dict.keys())

    with torch.no_grad():
        for name, param in model_dict[name_of_models[0]].state_dict().items():
            layer_weights = []
            for client_name in name_of_models:
                layer_weights.append(get_layer_weight(model_dict[client_name], name))

            t = torch.stack(layer_weights)
            meds = torch.median(t, 0).values
            new_weights[name] = meds
    return new_weights


def trimmed_mean(model_dict, m):
    new_weights = dict()
    name_of_models = list(model_dict.keys())

    with torch.no_grad():
        for name, param in model_dict[name_of_models[0]].state_dict().items():
            size = get_layer_weight(model_dict[name_of_models[0]], name).size()
            layer_weights = []
            for client_name in name_of_models:
                layer_weight = get_layer_weight(model_dict[client_name], name)
                layer_weights.append(torch.flatten(layer_weight))

            t = torch.stack(layer_weights)
            meds = torch.median(t, 0).values
            # dist = -(t - meds.unsqueeze(0).repeat(t.size()[0])) ** 2
            dist = -(t - meds.repeat(t.size()[0], 1))**2
            m_closet_idx = torch.topk(dist, m, dim=0).indices
            y_axis = torch.range(0, t.size()[1]-1).unsqueeze(0).repeat(m,1)
            m_closet = t[m_closet_idx.flatten(start_dim=0), y_axis.flatten(start_dim=0).long(),...].view(m,-1)
            mean_m_closet = torch.mean(m_closet.float(), 0)
            new_weights[name] = mean_m_closet.view(size)
    return new_weights


def closest(lst, K, m):
    # using heapq.nsmallest() to find the element in the list with the smallest absolute difference with K
    return heapq.nsmallest(m, lst, key=lambda x: abs(x - K))


def trimmed_mean_old(model_dict, m, cuda_device):
    n = len(model_dict)
    name_of_models = list(model_dict.keys())
    weight_accumulator = {}
    # top (n-m) values nearest to median

    for name, param in model_dict[name_of_models[0]].state_dict().items():
        struct = torch.zeros(size=param.size(), device=cuda_device)
        weight_accumulator[name] = struct

    # layer size can be [256, 256, 3, 3], n clients
    with torch.no_grad():
        for name, param in model_dict[name_of_models[0]].state_dict().items():
            layer_shape = param.size()
            for idx in itertools.product(*[range(s) for s in layer_shape]):
                one_dimension_values = []
                for client_name in model_dict:
                    value = get_layer_weight(model_dict[client_name], name)[idx] #.cpu()
                    one_dimension_values.append(value)
                    # print(f"value: {idx}, {value}")
                # med = np.median(one_dimension_values)
                one_dimension_values = torch.stack(one_dimension_values, dim=0)
                med = 1 # todo
                trimmed_mean_single = np.mean(closest(one_dimension_values, med, m))
                # print(f"trimmed_mean_single: {idx}, {trimmed_mean_single}")
                weight_accumulator[idx] = trimmed_mean_single
    return weight_accumulator
