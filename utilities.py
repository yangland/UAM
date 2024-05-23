import random
import numpy as np
import copy
# other utility functions

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def domain_without_edge(domain, step):
    new_domain = None
    if len(domain) == 2:
        x_max = max(domain[0])
        x_min = min(domain[0])
        y_max = max(domain[1])
        y_min = min(domain[1])
        new_domain = [[x_min + step, x_max - step], [y_min + step, y_max - step]]
    elif len(domain) == 1:
        x_max = max(domain[0])
        x_min = min(domain[0])
        new_domain = [[x_min + step, x_max - step]]
    return new_domain


def get_furthest_point(node, points):
    dists = []
    for i in range(len(points)):
        dists.append(np.linalg.norm((points[i], node)))
    ind = dists.index(max(dists))
    return points[ind]


def find_corners(domain, step):
    x_min = min(domain[0])
    x_max = max(domain[0])
    y_min = min(domain[1])
    y_max = max(domain[1])
    return ([[x_max - step, y_min + step], [x_min + step, y_min + step], [x_max - step, y_max - step],
             [x_max - step, y_max - step]])


def gen_random_from_domain(domain):
    x_min = min(domain[0])
    y_min = min(domain[1])
    x_len = max(domain[0]) - x_min
    y_len = max(domain[1]) - y_min

    x = x_min + random.random() * x_len
    y = y_min + random.random() * y_len

    return [x, y]


def create_mismatch_mapping(to_map):
    target_map = to_map.copy()
    full_mismatch = False

    while full_mismatch is False:
        random.shuffle(target_map)
        for i in range(len(to_map)):
            if target_map[i] != to_map[i]:
                if i == len(to_map) - 1:
                    return list(map(list, zip(to_map, target_map)))
                else:
                    continue
            else:
                # print("break")
                break


def get_layer_weight(model, name_list):
    splitted = name_list.split('.')
    result = model
    for p in splitted:
        result = getattr(result, p)
    return result


def action_to_params(action, sigmoid_c1, action_conv_func, rl_mode):
    """
    Map goal_conc to Sigmoid [0, 1]
    Map attack_magnitude to a**x a>0 [0, inf] (old)
    Map attack_magnitude to Sigmoid [0, 2]
    """

    if action_conv_func == '0-8':
        attack_magnitude = 8 / (1 + np.exp(-(action[0] - 2.75) * sigmoid_c1))
    elif action_conv_func == '0-10':
        attack_magnitude = 10 / (1 + np.exp(-(action[0] - 3.1) * sigmoid_c1))
    elif action_conv_func == '0-2':
        attack_magnitude = 2 / (1 + np.exp(-action[0] * sigmoid_c1))
    elif action_conv_func == 'set2':
        attack_magnitude = 10 / (1 + np.exp(-(action[0] - 1.1) * 2))
    elif action_conv_func == 'set3':
        attack_magnitude = 10 / (1 + np.exp(-(action[0] - 2.0) * 0.4))
    else:
        warnings.warn("unknown action_conv_func")
        return -1, -1

    goal_conc = None
    if len(action) == 1:
        pass
    elif len(action) == 2:
        if rl_mode == 'reverse_grad':
            goal_conc = 10 / (1 + np.exp(-action[1] * sigmoid_c1)) - 1
        elif rl_mode == 'untargeted_label_flip':
            goal_conc = 1 / (1 + np.exp(-action[1] * sigmoid_c1))
    else:
        warnings.warn("action to params only support action in 1, 2 dimension(s)")
        return None
    return goal_conc, attack_magnitude


def check_action_domain(action, domain):
    # check if the rl_action is within domain
    for j in range(len(action)):
        if action[j] > domain[j][-1] or action[j] < domain[j][0]:
            return False
    return True


def get_weights_from_model_dict(model_dict):
    model_dict_copy = copy.deepcopy(model_dict)
    return model_dict_copy

def client_random_selection(model_dict, num_clients, m):
    if m != None:
        idxs_clients = np.random.choice(range(num_clients), m, replace=False)

        model_names = []
        for i in idxs_clients:
            model_names.append("model" + str(i))
        
        selected_model_dict = {k: model_dict[k] for k in model_names}
        return selected_model_dict, idxs_clients
    else:
        return model_dict, list(range(num_clients))

class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0.5):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.kpi = None

    def __call__(self, kpi):
        if self.kpi is None:
            self.kpi = kpi
        else:
            if abs(kpi - self.kpi) <= self.min_delta:
                self.counter += 1
                if self.counter >= self.tolerance:
                    self.counter = 0
                    self.early_stop = True
            else:
                self.counter = 0
                self.kpi = kpi
