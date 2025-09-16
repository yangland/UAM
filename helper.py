import os
import numpy as np
import math
import random
import logging
import os
import warnings
import copy

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from models.mnist_3l import Mnist_3L
from models.mnist_1l import Mnist_1L
from models.cifar10_6l import Cifar10_6L
from models.resnet18 import ResNet18
from models.cifar10_9l import Cifar10_9L
from models.fashion_cnn import FashionCNN
from models.emnist_cnn import EmnistCNN
from models.cifar10_googlenet import GoogLeNet
from models.cifar10_8l import Cifar10_8L
from models.cifar10_18l import Cifar10_18L
from models.squeezenet11_cifar import SqueezeNet11_CIFAR
from models.cifar10_vgg import vgg11, vgg13, vgg16
from models.kmnist_12l import Kmnist_12L
from models.cifar100_mobilenetv2 import mobilenetv2
from models.cifar100_mobilenet import mobilenet
from models.cifar100_resnet import resnet18
from models.mobilenetv2_cifar import MobileNetV2_CIFAR
from models.resnet20_cifar import ResNet20_CIFAR

from csv_logging import Csv_logging
from distance_matrix import Distance
from collections import Counter
from neldermead import *
from utilities import *
from collections import OrderedDict

from aggr.geometric_median import geometric_median
from aggr.krum import krum
from aggr.foolsgold import foolsgold, foolsgold_new, pendulum, foolsgold_contra
from aggr.coordinate_wise_median import coordinate_wise_median, trimmed_mean
from aggr.mani_byzan import our_attack_dist, our_attack_score
from aggr.flame import flame_aggr, get_model_merged, get_model_update, fedavg

# helper.py (top of file)
import logging, sys

logger = logging.getLogger("uam")
logger.propagate = False  # don't bubble to root (prevents double printing)
if not logger.handlers:   # guard so we don't add multiple handlers
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

class Helper:
    def __init__(self, current_time, series_time, params, name, cuda_device):
        self.pre_aggregation_method = None
        self.cuda_device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.test_dataset = None
        self.train_dataset = None
        self.weights_csv_logger = None
        self.mali_valid_pca_csv_logger = None
        self.mali_valid_al_csv_logger = None
        self.god_valid_pca_csv_logger = None
        self.attack_method = None
        self.contra_sorted_dict = None
        self.reputation_dict = None
        self.honest_sample = None
        self.malicious_sample = None
        self.criterion_dict_m = None
        self.optimizer_dict_m = None
        self.malicious_names = None
        self.malicious_dict = None
        self.model_dict = None
        self.name_of_criterions = None
        self.name_of_optimizers = None
        self.criterion_dict = None
        self.optimizer_dict = None
        self.name_of_models = None
        self.main_criterion = None
        self.sample_nodes = None
        self.god_valid_al_csv_logger = None
        self.params = params
        self.neldermead = None
        self.g_main = None
        self.model_dict_m = None
        self.model_dict_before = None
        self.mali_mcts = None
        self.pool_dl = None
        self.search_dimension = len(self.params['x_start'])
        self.step = self.params['step']
        self.current_time = current_time
        self.name = name
        self.aggregation_method = self.params['aggregation_method']
        self.class_names = self.params['class_names']
        self.sampling_layer = self.params['sampling_layer']
        self.memory_dict = dict()
        self.history_len = self.params['history_len']
        self.number_of_clients = self.params['number_of_clients']
        self.server_lr = self.params['server_lr']
        self.server_lr_decay = self.params['server_lr_decay']
        self.server_lr_floor = self.params['server_lr_floor']
        self.num_classes = self.params['num_classes']
        self.contra_f = self.params['contra_f']
        self.if_series = self.params['if_series']
        self.sigmoid_c1 = self.params['sigmoid_c1']
        self.effective_action_domain = self.params['effective_action_domain']
        self.empirical_points = self.params['empirical_points'].copy()
        self.rounds_before_restart = self.params['rounds_before_restart']
        self.rl_mode = self.params['rl_mode']
        self.pre_aggregation_method = self.params['pre_aggregation_method']
        self.num_clients = int(self.params["number_of_clients"])
        self._logged_once = set()
        
        # Default to 100% if not present
        sp = self.params.get("select_precentage", 100)  # supports 10 or 0.10
        sp = float(sp)
        frac = sp / 100.0 if sp > 1.0 else sp
        self.k_per_round = max(1, min(self.num_clients, int(round(self.num_clients * frac))))

        # Two modes only
        self.selection_mode = str(self.params.get("selection_mode", "random")).lower()
        if self.selection_mode not in ("random", "fixed_pct"):
            self.selection_mode = "random"

        # Malicious pool (assumes you already create it; else fall back to params)
        if hasattr(self, "malicious_ids"):
            self.malicious_ids = set(int(x) for x in self.malicious_ids)
        else:
            if "number_of_malicious" in self.params:
                nm = int(self.params["number_of_malicious"])
            elif "number_of_malicious_s" in self.params and isinstance(self.params["number_of_malicious_s"], list):
                # interpret as percentage, e.g. [30] -> 30%
                perc = float(self.params["number_of_malicious_s"][0]) / self.params["number_of_clients"]
                nm = int(round(self.num_clients * perc))
            else:
                nm = 0
            nm = max(0, min(self.num_clients, nm))
            # deterministic: malicious are lowest IDs 0..nm-1
            self.malicious_ids = set(range(nm))

        # normalized forms used throughout the code
        self.malicious_ids = set(int(x) for x in self.malicious_ids)
        self.malicious_names = sorted(self.malicious_ids)

        # set selection ratio default if not provided (used by fixed_pct)
        if "malicious_selection_ratio" not in self.params:
            # ratio of total malicious to total clients (global)
            self.malicious_selection_ratio = len(self.malicious_ids) / float(self.num_clients) if self.num_clients > 0 else 0.0
        else:
            self.malicious_selection_ratio = float(self.params["malicious_selection_ratio"])
        
        # RNG
        self._rng = np.random.default_rng(self.params.get("random_seed", None))


        if self.params.get('select_precentage') is None or not self.params['select_precentage']:
            self.clients_selection = None
        else:
            self.clients_selection = int(self.params['select_precentage'] * self.number_of_clients / 100)

        if self.params['rl_manual']:
            self.rl_action = np.asarray([float(i) for i in self.params['manual_coordinates']])
        else:
            self.rl_action = np.asarray([float(i) for i in self.params['x_start']])

        if self.if_series:
            self.master_path = f'./logs/series_{self.name}_{series_time}'
            self.folder_path = f'{self.master_path}/{self.name}_{current_time}'
        else:
            self.folder_path = f'./logs/{self.name}_{current_time}'
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
            
        logfile = f'{self.folder_path}/log_{self.name}_{current_time}.txt'
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(logfile)
                for h in logger.handlers):
            logger.addHandler(logging.FileHandler(filename=logfile))

        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.global_results = []
        self.y_train_dict = None
        self.x_train_dict = None
        self.x_train_m_dict = dict()
        self.y_train_m_dict = dict()
        self.name_of_x_train_sets = None
        self.name_of_y_train_sets = None
        self.main_model = None


    def ensure_malicious_pool(self):
        """Create deterministic malicious pool from YAML before sampling rounds."""
        N = int(self.params["number_of_clients"])

        if "number_of_malicious" in self.params:
            nm = int(self.params["number_of_malicious"])
        elif "number_of_malicious_s" in self.params and isinstance(self.params["number_of_malicious_s"], list):
            perc = float(self.params["number_of_malicious_s"][0]) / 100.0
            nm = int(round(N * perc))
        else:
            nm = 0

        nm = max(0, min(N, nm))
        # deterministic: 0..nm-1
        self.malicious_ids = set(range(nm))
        self.malicious_names = sorted(self.malicious_ids)
        print( f"ensure_malicious_pool: {len(self.malicious_ids)} malicious out of {N} clients: {self.malicious_names}" )
        # ratio used by fixed_pct
        if "malicious_selection_ratio" in self.params:
            self.malicious_selection_ratio = float(self.params["malicious_selection_ratio"])
        else:
            self.malicious_selection_ratio = (len(self.malicious_ids) / float(N)) if N > 0 else 0.0


    # Create the main model and initialization
    def fl_initialization(self):
        if self.cuda_device.type != 'cpu':
            logger.info(
                f"device: {self.cuda_device} - {self.cuda_device.type} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            logger.info(f"device: {self.cuda_device.type}")
        nn_model = eval(self.params['nn_model'])
        self.main_model = nn_model()
        self.main_model.to(self.cuda_device)

        self.main_criterion = nn.CrossEntropyLoss()
        self.model_dict, self.optimizer_dict, self.criterion_dict = self.create_model_optimizer_criterion_dict(
            list(range(self.params['number_of_clients'])))
        self.sample_nodes = self.random_sampling_nodes(self.main_model, self.params['each_layer_list'])

        self.name_of_models = list(self.model_dict.keys())
        self.name_of_optimizers = list(self.optimizer_dict.keys())
        self.name_of_criterions = list(self.criterion_dict.keys())
        self.model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict)
        self.attack_method = self.params['attack_method']
        
        self.ensure_malicious_pool()  # create {0..nm-1} from YAML (e.g., 0..29 for 30%)
        logger.info(f"GLOBAL malicious_ids ({len(self.malicious_ids)}): {sorted(self.malicious_ids)[:10]}...")
        self.malicious_names, self.malicious_dict, self.honest_names = self.get_malicious_name()

        if len(self.malicious_names) == 0:
            self.params['malicious_attack'] = False
        else:
            self.params['malicious_attack'] = True

        self.malicious_sample = random.sample(self.malicious_names, min(4, self.params['number_of_malicious']))
        self.honest_sample = random.sample([i for i in list(range(self.number_of_clients))
                                            if i not in self.malicious_names], 4)

        self.reputation_dict = {"model" + str(i): 1 for i in range(self.number_of_clients)}
        self.contra_sorted_dict = dict()

        # create the object of Naive_Mcts class
        if self.attack_method == 'UAM':
            self.g_main = [[-1, -1], [-1, -1]]

            self.neldermead = NelderMead(name='NelderMead1', x_start=self.rl_action, step=self.step, progress=True,
                                         rl_manual=self.params['rl_manual'],
                                         manual_coordinates=self.params['manual_coordinates'],
                                         empirical_points=self.empirical_points)

        return None

    def create_honest_malicious_data(self):
        sample_dict_train = self.load_data()
        self.x_train_dict, self.y_train_dict = self.create_subsamples(sample_dict=sample_dict_train,
                                                                      x_data=self.x_train,
                                                                      y_data=self.y_train, x_name="x_train",
                                                                      y_name="y_train")

        self.name_of_x_train_sets = list(self.x_train_dict.keys())  # length is the # of clients
        self.name_of_y_train_sets = list(self.y_train_dict.keys())

        # spot check self.y_train_dict
        logger.info("spot check on y_train_dict:")
        keys = np.random.choice(list(self.y_train_dict), size=2, replace=False)
        for key in keys:
            c = Counter(self.y_train_dict[key].tolist())
            logger.info(f"{key} total {str(len(self.y_train_dict[key]))}: {str(sorted(c.items()))}")

        if self.params['malicious_attack'] is True:
            if self.attack_method != 'UAM':
                logger.info(
                    f"with {self.attack_method} malicious attack, {len(self.malicious_dict)} group(s), malicious clients id:")
            else:
                logger.info(f"with UAM malicious attack, {len(self.malicious_dict)} group, malicious clients id:")
            logger.info(f"malicious names: {self.malicious_names}")
            logger.info(f"malicious proportion: {self.malicious_dict}")

            x_pool = []
            y_pool = []

            for i in self.malicious_names:
                x_pool.append(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device))
                y_pool.append(self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))

            x_pool = torch.cat(x_pool, dim=0)
            y_pool = torch.cat(y_pool, dim=0)

            logger.info(
                f"Classes distribution on the pooled data (one malicious group, {len(self.malicious_names)} members):")
            c = Counter(y_pool.tolist())
            logger.info(f"total {len(y_pool)}: {sorted(c.items())}")

            pool_ds = TensorDataset(x_pool, y_pool)
            self.pool_dl = DataLoader(pool_ds, batch_size=self.params['batch_size'], shuffle=True)

            if self.attack_method == 'explicit boosting':
                # in EB attack the malicious dataset is fixed, created before the train_iteration starts
                for i, param in self.malicious_dict.items():
                    malicious_mode = "untargeted1"
                    malicious_group_list = param[1]
                    from_label = -1
                    to_label = -1

                    # Create the malicious datasets for the group
                    x_train_m_dict, y_train_m_dict = self.create_malicious_datasets(self.x_train_dict,
                                                                                    self.y_train_dict,
                                                                                    malicious_group_list,
                                                                                    malicious_mode, from_label,
                                                                                    to_label)
                    self.x_train_m_dict.update(x_train_m_dict)
                    self.y_train_m_dict.update(y_train_m_dict)

                self.name_of_x_train_m_sets = list(self.x_train_m_dict.keys())
                self.name_of_y_train_m_sets = list(self.y_train_m_dict.keys())
            elif self.attack_method == 'UAM':
                # in UAM attack, the malicious dataset in changing in each iteration
                assert len(self.malicious_dict) == 1, "UAM attack only support 1 group of malicious"
                # UAM attack need different malicious dataset every iteration
        else:
            self.malicious_names = []

        return None

    def train_iterations(self):
        # preparation before iterations
        god_loss_records, god_accuracy_records, god_asr_records = [], [], []

        # God's viewpoint
        self.god_valid_al_csv_logger = Csv_logging('god_valid_al', self.class_names, self.number_of_clients,
                                                   self.malicious_names)
        self.god_valid_pca_csv_logger = Csv_logging('god_valid_pca', self.class_names, self.number_of_clients,
                                                    self.malicious_names)
        # Malicious' viewpoint
        self.mali_valid_al_csv_logger = Csv_logging('mali_valid_al', self.class_names, self.number_of_clients,
                                                    self.malicious_names)
        self.mali_valid_pca_csv_logger = Csv_logging('mali_valid_pca', self.class_names, self.number_of_clients,
                                                     self.malicious_names)
        # Aggregation function's weights between all clients
        self.weights_csv_logger = Csv_logging('Fagg_weights', self.class_names, self.number_of_clients,
                                              self.malicious_names)
        # Early stopping when training matures from the God view
        if self.params['early_stopping']:
            early_stopping = EarlyStopping(tolerance=self.params['early_stopping_tolerance'],
                                           min_delta=self.params['early_stopping_min_delta'])

        logger.info("start training iterations:")

        dataset = self.params.get('type', getattr(self, 'environment_name', 'unknown'))
        attack  = self.params.get('attack_method', getattr(self, 'attack_method', 'None'))
        agg     = self.aggregation_method
        
        for i in range(self.params['iterations']):
            self.current_iter = i + 1
            logger.info(f"\n Iteration {self.current_iter}: dataset={dataset} | attack={attack} | agg={agg}")
            mali_loss_record, mali_accuracy_record, mali_asr_record, model_output_na = self.recording_validation(i,
                                                                                                                 "mali")
            if model_output_na:
                logger.info(f"model output nan, stop at epoch: {i}")
                break

            self.model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict)

            if self.params['malicious_attack']:
                if self.attack_method == 'UAM':
                    # 1. Record the reward from the previous round with the current feedback
                    # self.mali_mcts.record(mali_accuracy_record, mali_asr_record)
                    self.neldermead.record(mali_accuracy_record, self.rl_action, progress=True)

                    # 2. NelderMead
                    if not self.neldermead.rl_manual:
                        # first check the points in the empirical_points
                        if len(self.neldermead.empirical_points) != 0:
                            empirical_action = random.choice(self.neldermead.empirical_points)
                            self.neldermead.empirical_points.remove(empirical_action)
                            self.rl_action = empirical_action
                        else:
                            if self.neldermead.init_spot_search:
                                # find the best among the empirical_points
                                rewards_empirical = [row[2] for row in self.neldermead.history[1:]]
                                best_empirical = \
                                    self.neldermead.history[rewards_empirical.index(min(rewards_empirical)) + 1][0]
                                self.neldermead.restart(new_start=best_empirical)

                                # Step NM until it proposes a real action
                                self.rl_action = self.neldermead.nelder_mead()
                                spins = 0
                                while self.rl_action is False:
                                    spins += 1
                                    if spins > 100:
                                        raise RuntimeError("NelderMead: no actionable point after 100 internal steps (init spot search).")
                                    self.rl_action = self.neldermead.nelder_mead()
                                self.neldermead.init_spot_search = False

                            else:
                                if self.neldermead.reach_restart_threshold(self.rounds_before_restart,
                                                                           eps=0.01):
                                    new_start = None
                                    if self.rl_mode == "untargeted_label_flip":
                                        new_start = gen_random_from_domain(
                                            domain_without_edge(self.effective_action_domain, self.step))
                                    elif self.rl_mode == "reverse_grad":
                                        new_start = []
                                        for k in range(self.search_dimension):
                                            new_start.append(
                                                random.uniform(min(self.effective_action_domain[k]) + self.step,
                                                               max(self.effective_action_domain[k]) - self.step))
                                    self.neldermead.restart(new_start)
                                else:
                                    pass
                                # Step NM until it proposes a real action
                                self.rl_action = self.neldermead.nelder_mead()
                                spins = 0
                                while self.rl_action is False:
                                    spins += 1
                                    if spins > 100:
                                        raise RuntimeError(
                                            f"NelderMead: no actionable point after 100 internal steps "
                                            f"(mode={self.neldermead.mode}, submode={self.neldermead.submode})."
                                        )
                                    self.rl_action = self.neldermead.nelder_mead()
                    else:
                        self.rl_action = self.neldermead.manual_coordinates

                    logger.info(f"NelderMead's action: {self.rl_action}")
                else:
                    warnings.warn("Please use rl_manual to set up manual attack, set rl_based_sttack = True")
            else:
                pass

            # 2. clients local training
            self.process_with_local_training(i)

            # check if any model has nan value after the local training, if true stop the FL training
            for model_name, local_model in self.model_dict.items():
                is_nan = torch.stack([torch.isnan(p).any() for p in local_model.parameters()]).any()
                if is_nan:
                    logger.info(f"Found 'nan' in {model_name}, stop at epoch: {i}")
                    return self.global_results, god_accuracy_records, god_loss_records, god_asr_records

            # 3. aggregation and get the new main model
            self.main_model = self.update_main_model_use_aggregated_weights(self.main_model, self.subset_dict, i,
                                                                            self.server_lr)


            god_loss_record, god_accuracy_record, god_asr_record, model_output_na = self.recording_validation(i, "god")
            god_loss_records.append(god_loss_record)
            god_accuracy_records.append(god_accuracy_record)
            god_asr_records.append(god_asr_record)

            # add early_stopper in the god view
            if self.params['early_stopping']:
                early_stopping(god_accuracy_record)
                if early_stopping.early_stop:
                    logger.info(f"Early stop at epoch: {i}")
                    break

            # if main model contains nan, stop early
            is_nan = torch.stack([torch.isnan(p).any() for p in self.main_model.parameters()]).any()
            if is_nan:
                logger.info(f"Found 'nan' in main model, stop at epoch: {i}")
                break

        self.god_valid_al_csv_logger.save_csv(self.folder_path, self.current_time)
        self.god_valid_pca_csv_logger.save_csv(self.folder_path, self.current_time)
        self.mali_valid_al_csv_logger.save_csv(self.folder_path, self.current_time)
        self.mali_valid_pca_csv_logger.save_csv(self.folder_path, self.current_time)
        self.weights_csv_logger.save_csv(self.folder_path, self.current_time)
        return self.global_results, god_accuracy_records, god_loss_records, god_asr_records

    def process_with_local_training(self, iteration):
        if len(self.malicious_names) != 0:
            goal_conc, attack_magnitude = action_to_params(self.rl_action,
                                                           self.sigmoid_c1,
                                                           self.params['action_conv_func'], self.params['rl_mode'])

            if self.attack_method == 'UAM':
                if self.rl_mode == "untargeted_label_flip":
                    g_locals = self.malicious_goals_on_goal_conc(goal_conc)
                    logger.info("g_locals: ")
                    for key, value in g_locals.items():
                        logger.info(f"{key}: {dict(value)}")

                    x_train_m_dict_rl, y_train_m_dict_rl = self.create_rl_malicious_datasets(self.x_train_dict,
                                                                                             self.y_train_dict,
                                                                                             self.malicious_names,
                                                                                             g_locals)
                    # replace the dictionary every iteration
                    self.x_train_m_dict.update(x_train_m_dict_rl)
                    self.y_train_m_dict.update(y_train_m_dict_rl)
                elif self.rl_mode == "reverse_grad":
                    pass
            else:
                pass
        else:
            attack_magnitude = 1.0
            goal_conc = 0

        # Minimal, 2-mode selection
        idx_clients = self.sample_clients_for_round()
        idx_clients_int = [int(i) for i in idx_clients] 
        logger.info(f"selected clients: {idx_clients_int} (total {len(idx_clients_int)})")
        self.last_selected_client_ids = [int(i) for i in idx_clients]

        # build the subset from the normalized ids
        self.subset_dict = {f"model{i}": self.model_dict[f"model{i}"] for i in self.last_selected_client_ids}

        # did we sample any malicious clients?
        self.iteration_mali = any(i in self.malicious_names for i in self.last_selected_client_ids)
        if self.iteration_mali:
            print("This iteration has malicious client(s)")

        logger.info("training on the end nodes...")
        grads_regular = self.start_train_end_node_process(
            self.params['malicious_attack'],
            self.malicious_names,
            self.sample_nodes, iteration + 1,
            self.params['plot_individual'],
            self.params['plot_global'],
            self.params['weights_layer'],
            self.params['distance_matrix'],
            self.rl_mode,
            self.last_selected_client_ids,  # pass normalized ids
        )

        # Only attack if (a) there are malicious clients in the config AND (b) some were sampled this round
        if len(self.malicious_names) != 0 and self.iteration_mali:
            self.attacker_modify_trained_weights(grads_regular, attack_magnitude, goal_conc, self.rl_mode)

        return None


    def attacker_modify_trained_weights(self, grads_regular, attack_magnitude, goal_conc, rl_mode):
        with torch.no_grad():
            # --- Scope targets to THIS ROUND ONLY ---
            selected_keys = set(self.subset_dict.keys())                 # e.g., {"model3","model7",...}
            mal_keys      = {f"model{i}" for i in self.malicious_names}  # all malicious model names
            target_keys   = selected_keys & mal_keys                      # selected ∩ malicious
            assert len(target_keys) > 0, "No malicious client selected this round, but attacker was invoked."

            if self.attack_method == 'explicit boosting':
                K = 5
                # apply ONLY to selected malicious (target_keys)
                for model_name in target_keys:
                    for layer_name, _ in self.main_model.state_dict().items():
                        get_layer_weight(self.model_dict[model_name], layer_name).data = (
                            get_layer_weight(self.model_dict_before[model_name], layer_name).clone()
                            + K * get_layer_weight(grads_regular[model_name], layer_name).clone()
                        )

            elif self.attack_method == 'little':
                # stats computed on all malicious, applied to selected malicious
                malicious_models = {k: self.model_dict[k] for k in mal_keys}
                mean = self.get_aggregated_weights(malicious_models, 'fedavg')
                Z_max = 0.08
                std_dict = {}
                for layer_name, _ in self.main_model.state_dict().items():
                    layer_list = []
                    model_dict_after = get_weights_from_model_dict(self.model_dict)  # full snapshot for std
                    for k in mal_keys:
                        layer_list.append(get_layer_weight(model_dict_after[k], layer_name).clone())
                    stacked = torch.stack(layer_list)
                    std_dict[layer_name] = torch.std(stacked.float(), dim=0, correction=0)

                for model_name in target_keys:
                    for layer_name, _ in self.main_model.state_dict().items():
                        get_layer_weight(self.model_dict[model_name], layer_name).data = (
                            mean[layer_name].float() + Z_max * std_dict[layer_name].float()
                        )

            elif self.attack_method in ['min-max', 'min-sum']:
                # updates computed on all malicious, applied to selected malicious
                n_attacker = 5
                malicious_models = {k: self.model_dict[k] for k in mal_keys}
                mean = self.get_aggregated_weights(malicious_models, 'fedavg')
                mal_updates = {}
                model_dict_after = get_weights_from_model_dict(self.model_dict)  # full snapshot for stacking
                for layer_name, _ in self.main_model.state_dict().items():
                    layer_list = [get_layer_weight(model_dict_after[k], layer_name).clone() for k in mal_keys]
                    stacked = torch.stack(layer_list)
                    if self.attack_method == 'min-max':
                        mal_updates[layer_name] = our_attack_dist(stacked, mean[layer_name], n_attacker,
                                                                self.cuda_device, dev_type='unit_vec')
                    else:
                        mal_updates[layer_name] = our_attack_score(stacked, mean[layer_name], n_attacker,
                                                                self.cuda_device, dev_type='unit_vec')
                for model_name in target_keys:
                    for layer_name, _ in self.main_model.state_dict().items():
                        get_layer_weight(self.model_dict[model_name], layer_name).data = mal_updates[layer_name]

            elif self.attack_method == 'UAM':
                if rl_mode == "untargeted_label_flip":
                    # apply ONLY to selected malicious (target_keys)
                    for model_name in target_keys:
                        for layer_name, _ in self.main_model.state_dict().items():
                            get_layer_weight(self.model_dict[model_name], layer_name).data = (
                                get_layer_weight(self.model_dict_before[model_name], layer_name).clone()
                                + attack_magnitude * get_layer_weight(grads_regular[model_name], layer_name).clone()
                            )

                elif rl_mode == "reverse_grad":
                    if goal_conc is None:
                        # standard reverse-grad on selected malicious
                        for model_name in target_keys:
                            for layer_name, _ in self.main_model.state_dict().items():
                                if self.params['Median_att'] is False:
                                    get_layer_weight(self.model_dict[model_name], layer_name).data = (
                                        get_layer_weight(self.model_dict_before[model_name], layer_name).clone()
                                        - attack_magnitude * get_layer_weight(grads_regular[model_name], layer_name).clone()
                                    )
                                else:
                                    # median-variant base pool: selected malicious (black-box) or selected honest (white-box)
                                    if self.params['White_box'] is False:
                                        base_pool = target_keys                  # selected malicious this round
                                    else:
                                        base_pool = selected_keys - target_keys  # selected honest this round
                                    assert len(base_pool) > 0, "Median attack base_pool empty for this round."

                                    model_list = {k: self.model_dict[k] for k in base_pool}
                                    model_median = self.get_aggregated_weights(model_list, 'fedavg')
                                    get_layer_weight(self.model_dict[model_name], layer_name).data = (
                                        - model_median[layer_name].clone() * attack_magnitude
                                    )

                    else:
                        # diversity-control path; apply to selected malicious
                        for layer_name, _ in self.main_model.state_dict().items():
                            # mean over all malicious (global)
                            layers = []
                            for i in self.malicious_names:
                                k = f"model{i}"
                                layers.append(torch.flatten(get_layer_weight(self.model_dict[k], layer_name).data.clone()))
                            t = torch.stack(layers)
                            mean_t = torch.mean(t.float(), 0)

                            for i in [int(n[5:]) for n in target_keys]:
                                model_name = f"model{i}"
                                shape = get_layer_weight(self.model_dict[model_name], layer_name).data.size()
                                single_layer = torch.flatten(get_layer_weight(self.model_dict[model_name], layer_name).data.clone())
                                v_i_prime = (1 - goal_conc) * single_layer + goal_conc * mean_t
                                v_i_prime = v_i_prime / torch.norm(v_i_prime) * torch.norm(mean_t)
                                v_i_prime = v_i_prime.view(shape)

                                get_layer_weight(self.model_dict[model_name], layer_name).data = (
                                    get_layer_weight(self.model_dict_before[model_name], layer_name).clone()
                                    - attack_magnitude * v_i_prime
                                )

                elif rl_mode == "zeros":
                    for model_name in target_keys:
                        for layer_name, _ in self.main_model.state_dict().items():
                            layer_size = get_layer_weight(self.model_dict[model_name], layer_name).data.size()
                            get_layer_weight(self.model_dict[model_name], layer_name).data = torch.zeros(layer_size).to(self.cuda_device)
                else:
                    raise ValueError(f"Unrecognized rl_mode: {rl_mode}")

            else:
                raise ValueError(f"Unrecognized attack_method: {self.attack_method}")
        return None


    def malicious_goals_on_goal_conc(self, goal_conc):
        g_main = self.g_main
        g_locals = {}

        for i in self.malicious_names:
            g_locals[i] = self.goal_randomizer(g_main, goal_conc)
        return g_locals

    def goal_randomizer(self, goal, goal_conc):
        all_classes = list(range(0, self.num_classes))
        new_goal = OrderedDict()

        for i in range(len(all_classes)):
            p_0 = np.random.random()
            if p_0 < goal_conc:
                new_goal[all_classes[i]] = goal[i][-1]
            else:
                new_goal[all_classes[i]] = random.choice(all_classes)
        return new_goal

    def recording_validation(self, iteration, viewpoint):
        valid_data = None
        assert viewpoint in ['god', 'mali'], "viewpoint has to be in ['god', 'mali']"

        if viewpoint == 'god':
            # validation from God's viewpoint
            valid_data = self.god_valid_dl
        elif viewpoint == 'mali':
            if self.pool_dl is not None:
                valid_data = self.pool_dl
            else:
                return None, None, None, None

        loss_record, accuracy_record, attack_success_rate_dict, per_class_accuracy_dict, model_output_na = (
            self.validation(self.main_model, valid_data, self.main_criterion))

        if viewpoint == 'god':
            # csv logger train, accuracy loss
            if self.attack_method == 'UAM':
                act1 = self.rl_action[0]
                if len(self.rl_action) == 2:
                    act2 = self.rl_action[1]
                else:
                    act2 = 'no action'
            else:
                act1 = 'no action'
                act2 = 'no action'
            self.god_valid_al_csv_logger.append_csv([iteration + 1, accuracy_record, loss_record, act1, act2])

            # csv logger, per class accuracy, attack success rate
            god_valid_pca_csv_logger_line = [iteration + 1]
            for classname in self.class_names:
                god_valid_pca_csv_logger_line.append(per_class_accuracy_dict[classname])
            god_valid_pca_csv_logger_line.append(attack_success_rate_dict)

            self.god_valid_pca_csv_logger.append_csv(god_valid_pca_csv_logger_line)

            logger.info("(god) main_model accuracy % on all test data: {:.2f}".format(100 * float(accuracy_record)))

            # multi attack objects
            if self.params['malicious_attack']:
                if self.params['Goal'] == 'label_flipping':
                    attack_success_rates = []
                    for target_name, param in attack_success_rate_dict.items():
                        line = [self.class_names[param[0]], self.class_names[param[1]], param[-1]]
                        attack_success_rates.append(line)
                        logger.info(
                            "(god) main_model attack success rate % (from {} to {}): {:.2f}".format(
                                self.class_names[param[0]] \
                                ,
                                self.class_names[param[1]],
                                param[-1]))

            # asr_records.append(attack_success_rates)
        elif viewpoint == 'mali':
            # csv logger train, accuracy loss
            self.mali_valid_al_csv_logger.append_csv([iteration + 1, accuracy_record, loss_record])

            # csv logger, per class accuracy, attack success rate
            mali_valid_pca_csv_logger_line = [iteration + 1]
            for classname in self.class_names:
                mali_valid_pca_csv_logger_line.append(per_class_accuracy_dict[classname])
            mali_valid_pca_csv_logger_line.append(attack_success_rate_dict)

            self.mali_valid_pca_csv_logger.append_csv(mali_valid_pca_csv_logger_line)

            logger.info("(mali) main_model accuracy % on all test data: {:.2f}".format(100 * float(accuracy_record)))

            if self.params['Goal'] == 'label_flipping':
                # multi attack objects
                attack_success_rates = []
                # print("attack_success_rate_dict", attack_success_rate_dict)
                for target_name, param in attack_success_rate_dict.items():
                    line = [self.class_names[param[0]], self.class_names[param[1]], param[-1]]
                    attack_success_rates.append(line)
                    logger.info(
                        "(mali) main_model attack success rate % (from {} to {}): {:.2f}".format(
                            self.class_names[param[0]] \
                            ,
                            self.class_names[param[1]],
                            param[-1]))

            # asr_records.append(attack_success_rates)
        return loss_record, accuracy_record, attack_success_rate_dict, model_output_na

    # def get_malicious_name(self):
    #     # new self.malicious_name as a dictionary
    #     malicious_dict = dict()
    #     malicious_selected = np.random.choice(range(self.params['number_of_clients']), \
    #                                           self.params['number_of_malicious'], replace=False)
    #     malicious_selected.sort()
    #     honest_selected = [i for i in list(range(self.params['number_of_clients'])) if i not in malicious_selected]
    #     malicious_preprocessing = list(malicious_selected)

    #     if self.attack_method in ['explicit boosting', 'traditional']:
    #         malic_prop = self.params['malicious_proportion']

    #         sum_prop = 0
    #         for i in range(len(malic_prop)):
    #             # ensure the num of malicious and the malic_prop % are match
    #             assert (self.params['number_of_malicious'] * malic_prop[i][1]) % 1 <= 0.001, \
    #                 "num of malicious * each malic_prop % has to be int"
    #             sum_prop += malic_prop[i][1]
    #             group_selected = np.random.choice(malicious_preprocessing, \
    #                                               int(self.params['number_of_malicious'] * malic_prop[i][1]),
    #                                               replace=False)

    #             malicious_mode = malic_prop[i][0]
    #             if malic_prop[i][0] in ['untargeted1', 'untargeted2']:
    #                 # -1 due to untargeted
    #                 from_label = -1
    #                 to_label = -1
    #             else:
    #                 from_label = malic_prop[i][2]
    #                 to_label = malic_prop[i][3]
    #             malicious_dict[i] = [malicious_mode, list(group_selected), from_label, to_label]
    #             malicious_preprocessing = [x for x in malicious_preprocessing if x not in group_selected]

    #         assert sum_prop - 1 <= 0.001, "malic_prop % have to sum to 1"
    #     elif self.attack_method == 'little':
    #         malicious_mode = 'None'
    #         goal = [-1, -1]
    #         malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
    #     elif self.attack_method == 'UAM':
    #         malicious_mode = 'None'

    #         if self.params['Goal'] == 'untargeted':
    #             malicious_mode = 'untargeted'
    #             to_map = list(range(self.num_classes))
    #             goal = create_mismatch_mapping(to_map)
    #         elif self.params['Goal'] == 'random_pair':
    #             [from_label, to_label] = np.random.choice(self.params['num_classes'], 2, replace=False)
    #             goal = [[from_label, to_label], [to_label, from_label]]
    #             malicious_mode = 'label_flipping'
    #         elif type(self.params['Goal']) == list:
    #             malicious_mode = 'label_flipping'
    #             goal = self.params['Goal']
    #         else:
    #             warnings.warn('Unknown attack_method!')
    #         malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
    #         # logger.info("helper.get_malicious_name debugging 4: malicious_dict created")
    #         print("malicious_dict", malicious_dict)
    #     return list(malicious_selected), malicious_dict, honest_selected


    def get_malicious_name(self):
        # Allow zero-malicious case – no assert here.
        N = int(self.params['number_of_clients'])

        # Use the pool created by ensure_malicious_pool()
        malicious_selected = sorted(int(x) for x in getattr(self, "malicious_ids", set()))
        honest_selected = [i for i in range(N) if i not in malicious_selected]
        malicious_preprocessing = list(malicious_selected)

        malicious_dict = {}

        # --- If no malicious clients, short-circuit cleanly ---
        if len(malicious_selected) == 0:
            # No groups; everything is benign
            # For 'little'/'UAM' or others, returning an empty malicious group list is fine.
            return [], malicious_dict, honest_selected

        # --- From here on, we have >=1 malicious ---
        if self.attack_method in ['explicit boosting', 'traditional']:
            malic_prop = self.params['malicious_proportion']
            m_total = len(malicious_selected)
            sum_prop = 0.0
            rng = np.random.default_rng(self.params.get("random_seed", None))

            for i, spec in enumerate(malic_prop):
                mode = spec[0]
                p = float(spec[1])
                # ensure integer split
                assert abs(m_total * p - round(m_total * p)) <= 1e-3, \
                    "num of malicious * each malic_prop % has to be int"
                sum_prop += p

                take = int(round(m_total * p))
                group_selected = rng.choice(malicious_preprocessing, size=take, replace=False).tolist()

                if mode in ['untargeted1', 'untargeted2']:
                    from_label, to_label = -1, -1
                else:
                    from_label, to_label = int(spec[2]), int(spec[3])

                malicious_dict[i] = [mode, list(group_selected), from_label, to_label]
                # remove chosen from pool for next groups
                malicious_preprocessing = [x for x in malicious_preprocessing if x not in group_selected]

            assert abs(sum_prop - 1.0) <= 1e-3, "malic_prop % have to sum to 1"

        elif self.attack_method == 'little':
            malicious_mode = 'None'
            goal = [-1, -1]
            malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]

        elif self.attack_method == 'UAM':
            malicious_mode = 'None'
            if self.params['Goal'] == 'untargeted':
                malicious_mode = 'untargeted'
                to_map = list(range(self.num_classes))
                goal = create_mismatch_mapping(to_map)
            elif self.params['Goal'] == 'random_pair':
                rng = np.random.default_rng(self.params.get("random_seed", None))
                from_label, to_label = rng.choice(self.params['num_classes'], 2, replace=False)
                goal = [[int(from_label), int(to_label)], [int(to_label), int(from_label)]]
                malicious_mode = 'label_flipping'
            elif isinstance(self.params['Goal'], list):
                malicious_mode = 'label_flipping'
                goal = self.params['Goal']
            else:
                warnings.warn('Unknown attack_method!')
                goal = [-1, -1]
            malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
            print("malicious_dict", malicious_dict)  # keep or remove as you like

        else:
            # default: one group with all malicious
            malicious_dict[0] = ['None', malicious_preprocessing, [-1, -1]]

        return list(malicious_selected), malicious_dict, honest_selected


    def create_rl_malicious_datasets(self, x_train_dict, y_train_dict, malicious_names, g_locals):
        x_train_m_dict = {}
        y_train_m_dict = {}

        for i in malicious_names:
            x_train_m_dict["x_train_m_" + str(i)] = x_train_dict["x_train" + str(i)]
            # modify y labels based on the action dictionary
            y_modified = []
            for j in range(len(y_train_dict['y_train' + str(i)])):
                if y_train_dict['y_train' + str(i)][j].cpu().item() not in g_locals[i].keys():
                    # for classes not in the action, keep the same
                    y_modified.append(y_train_dict['y_train' + str(i)][j].cpu())
                else:
                    # for classes included in the action, modify based on the g_local dict
                    y_modified.append(torch.tensor(g_locals[i][y_train_dict['y_train' + str(i)][j].cpu().item()]))
            y_train_m_dict["y_train_m_" + str(i)] = torch.from_numpy(np.asarray(y_modified)).to(self.cuda_device)
        return x_train_m_dict, y_train_m_dict

    def create_malicious_datasets(self, x_train_dict, y_train_dict, malicious_names, mode, from_label, to_label):
        x_train_m_dict = {}
        y_train_m_dict = {}

        if mode == 'untargeted1':
            for i in malicious_names:
                # keep x train unchanged
                x_train_m_dict["x_train_m_" + str(i)] = x_train_dict["x_train" + str(i)]
                y_untargeted = []
                for j in range(len(y_train_dict['y_train' + str(i)])):
                    # y = existed y label + 1, 1->2, ... , 9 -> 1
                    y_untargeted.append((y_train_dict['y_train' + str(i)][j].cpu() + 1) % 10)
                y_train_m_dict["y_train_m_" + str(i)] = torch.from_numpy(np.asarray(y_untargeted)).to(self.cuda_device)

        if mode == 'untargeted2':
            classes = list(range(self.params['num_classes']))
            for i in malicious_names:
                # keep x train unchanged
                x_train_m_dict["x_train_m_" + str(i)] = x_train_dict["x_train" + str(i)]
                y_untargeted = []
                for j in range(len(y_train_dict['y_train' + str(i)])):
                    # y = existed y label -> random label of all class with same chance
                    y_untargeted.append(random.choice(classes))
                y_train_m_dict["y_train_m_" + str(i)] = torch.from_numpy(
                    np.asarray(y_untargeted).astype(np.int64)).to(self.cuda_device)

        if mode == 'label_flipping':
            for i in malicious_names:
                # keep x train unchanged
                x_train_m_dict["x_train_m_" + str(i)] = x_train_dict["x_train" + str(i)]
                y_flipped = []
                for j in range(len(y_train_dict['y_train' + str(i)])):
                    # if y is from_label, update it to to_label, vice versa
                    current_label = y_train_dict['y_train' + str(i)][j].cpu()
                    if current_label not in [from_label, to_label]:
                        y_flipped.append(current_label)
                    else:
                        if current_label == from_label:
                            y_flipped.append(to_label)
                        elif current_label == to_label:
                            y_flipped.append(from_label)
                y_train_m_dict["y_train_m_" + str(i)] = torch.from_numpy(np.asarray(y_flipped)).to(self.cuda_device)

        if mode == 'backdoor':
            for i in malicious_names:
                client_x = []
                client_y = []
                for j in range(len(x_train_dict['x_train' + str(i)])):
                    current_label = y_train_dict['y_train' + str(i)][j]
                    current_x = x_train_dict['x_train' + str(i)][j]
                    if current_label == from_label:
                        if self.params['type'] == 'mnist':
                            # Set pixel #783 to white (the brightest white in picture)
                            current_x[783] = torch.max(current_x)
                        elif self.params['type'] == 'cifar10':
                            for layer in range(len(current_x)):
                                current_x[layer][31][31] = torch.max(current_x[layer])
                        else:
                            print("This dataset's backdoor has not been setup")
                            break
                        client_x.append(current_x)
                        client_y.append(to_label)
                    else:
                        client_x.append(current_x)
                        client_y.append(current_label.item())
                client_x = torch.stack(client_x)
                x_train_m_dict["x_train_m_" + str(i)] = client_x.to(self.cuda_device)
                y_train_m_dict["y_train_m_" + str(i)] = torch.from_numpy(
                    np.asarray(client_y).astype(np.int64)).to(self.cuda_device)

        return x_train_m_dict, y_train_m_dict

    def build_classes_dict(self):
        classes_dict = {}
        class_ids = set([])
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000
            _, label = x
            class_ids.add(label)
            key = str("label") + str(label)
            if key in classes_dict:
                classes_dict[str("label") + str(label)].append(int(ind))
            else:
                classes_dict[str("label") + str(label)] = [int(ind)]
        return classes_dict, list(class_ids)

    def get_x_y_datasets(self, dataset):
        x_data = []
        y_data = []
        for ind, data in enumerate(dataset):  # for cifar: 50000
            x, y = data
            if self.params['type'] == 'mnist':
                x_data.append(torch.flatten(x[0]))
            elif self.params['type'] in ['cifar10', 'fashion_mnist', 'emnist', 'kmnist', 'cifar100', 'imagenette']:
                x_data.append(x)
            y_data.append(y)
        x_data = torch.stack(x_data)
        y_data = torch.as_tensor(y_data)
        return (x_data, y_data)

    # Data Loading and Processing Functions
    def load_data(self):
        transform_vanilla = transforms.Compose([transforms.ToTensor(), ])

        # ---------- CHANGE: add CIFAR normalizations ----------
        transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616)),
        ])
        transform_cifar100 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761)),
        ])
        # ------------------------------------------------------

        if self.params['type'] == 'mnist':
            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset  = datasets.MNIST('./data', train=False, transform=transform_vanilla)

        elif self.params['type'] == 'cifar10':
            # ---------- CHANGE: use transform_cifar10 ----------
            self.train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=transform_cifar10)
            self.test_dataset  = datasets.CIFAR10('./data', train=False, transform=transform_cifar10)
            # ---------------------------------------------------

        elif self.params['type'] == 'fashion_mnist':
            self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset  = datasets.FashionMNIST('./data', train=False, transform=transform_vanilla)

        elif self.params['type'] == 'emnist':
            self.train_dataset = datasets.EMNIST('./data', train=True,  download=True, transform=transform_vanilla, split="letters")
            self.test_dataset  = datasets.EMNIST('./data', train=False, transform=transform_vanilla, split="letters")

        elif self.params['type'] == 'kmnist':
            self.train_dataset = datasets.KMNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset  = datasets.KMNIST('./data', train=False, transform=transform_vanilla)

        elif self.params['type'] == 'cifar100':
            # ---------- CHANGE: use transform_cifar100 ----------
            self.train_dataset = datasets.CIFAR100('./data', train=True,  download=True, transform=transform_cifar100)
            self.test_dataset  = datasets.CIFAR100('./data', train=False, transform=transform_cifar100)
            # ----------------------------------------------------

        elif self.params['type'] == 'imagenette':
            transform = transforms.Compose([
                transforms.CenterCrop(160),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
            ])
            # Please add downloaded file to "./data/"
            # https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
            self.train_dataset = datasets.ImageFolder("./data/imagenette2-160/train", transform)
            self.test_dataset  = datasets.ImageFolder("./data/imagenette2-160/val", transform)

        else:
            warnings.warn("Data type not recognized in load_data")

        logger.info('reading data done')

        self.classes_dict, self.class_ids = self.build_classes_dict()

        # order for class ids when the id in y data not start with 0
        self.label_classname_dict = dict(zip(self.class_ids, self.class_names))

        self.x_train, self.y_train = self.get_x_y_datasets(self.train_dataset)

        self.god_valid_dl = DataLoader(self.test_dataset, batch_size=self.params['batch_size'] * 2, shuffle=False)

        logger.info('build_classes_dict done')

        sample_dict_train = self.get_subsamples_indices(label_dict=self.classes_dict,
                                                        number_of_clients=self.params['number_of_clients'],
                                                        amount=self.params['train_amount'],
                                                        num_classes=self.params['num_classes'],
                                                        class_ids=self.class_ids,
                                                        NonIID=self.params['if_noniid'],
                                                        alpha=self.params['dirichlet_alpha'])

        return sample_dict_train

    # def get_subsamples_indices(self, label_dict, number_of_clients, amount, num_classes, class_ids, NonIID, alpha):
    #     sample_dict = dict()
    #     batch_size = int(math.floor(amount / number_of_clients))

    #     if NonIID == False:
    #         for i in range(number_of_clients):  # create 100 number_of_clients
    #             sample_name = "sample" + str(i)
    #             sample_list = []
    #             # IID distribution, each number class gets same samples
    #             for j in range(num_classes):  # for each class
    #                 label_name = str("label") + str(j)
    #                 a = label_dict[label_name][i * batch_size:(i + 1) * batch_size]
    #                 sample_list.extend(a)
    #             sample_dict.update({sample_name: np.asarray(sample_list)})  # each sample 450, 100 samples
    #     else:
    #         image_nums = []
    #         sample_dict = {}

    #         # created the keys for sample_dict
    #         for i in range(number_of_clients):
    #             sample_name = "sample" + str(i)
    #             sample_dict.update({sample_name: []})

    #         if self.params.get("if_quantity_based_noniid") is None or self.params["if_quantity_based_noniid"] is False:
    #             # distribution based label skew based of Dirichlet distribution
    #             for label in class_ids:
    #                 image_num = []
    #                 sampled_probabilities = amount * np.random.dirichlet(
    #                     np.array(number_of_clients * [alpha]))
    #                 class_label_len = len(label_dict[str("label") + str(label)])

    #                 for sample in range(number_of_clients):
    #                     # at least 1 sample image per client
    #                     no_imgs = math.ceil(sampled_probabilities[sample])
    #                     label_name = str("label") + str(label)

    #                     cut_off = min(class_label_len, no_imgs)
    #                     sampled_list = label_dict[label_name][:cut_off]
    #                     label_dict[label_name] = label_dict[label_name][cut_off:]
    #                     image_num.append(len(sampled_list))
    #                     dumb2 = sampled_list

    #                     class_label_len = class_label_len - len(sampled_list)
    #                     sample_name = "sample" + str(sample)
    #                     image_nums.append(image_num)
    #                     sample_dict[sample_name] = sample_dict[sample_name] + dumb2
    #         else:
    #             # quantity-based label skew, limit the class number (theta) for client
    #             theta = self.params['quantity_based_theta']
    #             sets_per_client = math.floor(self.num_classes * theta)
    #             sets_per_class = {}

    #             for label, indices in label_dict.items():
    #                 sets_per_class[label] = math.floor(len(indices) / (number_of_clients * theta)) - 1

    #             for sample in range(number_of_clients):
    #                 if len(class_ids) > sets_per_client:
    #                     selected_class_ids = random.sample(class_ids, sets_per_client)
    #                 else:
    #                     selected_class_ids = class_ids

    #                 for label in selected_class_ids:
    #                     label_name = str("label") + str(label)
    #                     class_label_len = len(label_dict[label_name])
    #                     cut_off = min(class_label_len, sets_per_class[label_name])
    #                     sampled_list = label_dict[label_name][:cut_off]
    #                     label_dict[label_name] = label_dict[label_name][cut_off:]

    #                     sample_name = "sample" + str(sample)
    #                     sample_dict[sample_name] = sample_dict[sample_name] + sampled_list

    #                     if len(label_dict[label_name]) < sets_per_class[label_name] * 0.1:
    #                         class_ids.remove(label)

    #     return sample_dict

    # def create_subsamples(self, sample_dict, x_data, y_data, x_name, y_name):
    #     x_data_dict = dict()
    #     y_data_dict = dict()

    #     for i in range(len(sample_dict)):  ### len(sample_dict)= number of clients
    #         xname = x_name + str(i)
    #         yname = y_name + str(i)
    #         sample_name = "sample" + str(i)

    #         # use the global index to reconnect x and y
    #         sample_dict[sample_name].sort()
    #         x_info = [x_data[j] for j in sample_dict[sample_name]]
    #         if x_info == []:
    #             print("i", i)
    #         x_info = torch.stack(x_info)
    #         x_info.to(self.cuda_device)
    #         x_data_dict.update({xname: x_info})
    #         y_info = torch.as_tensor([y_data[j] for j in sample_dict[sample_name]])
    #         y_info.to(self.cuda_device)
    #         y_data_dict.update({yname: y_info})

    #     return (x_data_dict, y_data_dict)


    def get_subsamples_indices(self, label_dict, number_of_clients, amount, num_classes, class_ids, NonIID, alpha):
        """
        Returns {'sample{i}': [global_indices, ...]}.

        - If 'amount' is None/<=0/'NA'/'none' -> use full dataset.
        - NonIID=True: for each class c, sample Dirichlet([alpha]*C) -> probs, then Multinomial(class_quota, probs).
        - Totals exactly match the chosen target (full dataset or 'amount').
        - Enforces min-1 sample per client by borrowing from largest holder if needed.
        """
        import numpy as np

        # ----- helpers -----
        def _parse_amount(amnt):
            # accept None, 0, negative, 'NA', 'none', 'full' to mean "use full dataset"
            if amnt is None:
                return None
            if isinstance(amnt, (np.integer, int)) and amnt > 0:
                return int(amnt)
            if isinstance(amnt, str) and amnt.strip().lower() in {"na", "none", "full", "all"}:
                return None
            try:
                v = int(amnt)
                return v if v > 0 else None
            except Exception:
                return None

        rng = np.random.default_rng(self.params.get("random_seed", None))

        # available per class and total
        avail_per_class = {c: len(label_dict[f"label{c}"]) for c in class_ids}
        total_avail = sum(avail_per_class.values())

        target_total = _parse_amount(amount)
        if target_total is None or target_total >= total_avail:
            target_total = total_avail  # full dataset

        # ---- per-class quotas that sum to target_total ----
        # base proportional allocation (largest remainder), capped by availability
        if target_total == total_avail:
            per_class_quota = {c: avail_per_class[c] for c in class_ids}
        else:
            # proportional floor
            props = {c: (avail_per_class[c] / total_avail) * target_total for c in class_ids}
            floors = {c: int(np.floor(props[c])) for c in class_ids}
            per_class_quota = floors.copy()
            remainder = target_total - sum(floors.values())
            # fractional parts for largest remainder
            fracs = sorted(class_ids, key=lambda c: (props[c] - floors[c]), reverse=True)
            for c in fracs:
                if remainder == 0:
                    break
                if per_class_quota[c] < avail_per_class[c]:
                    per_class_quota[c] += 1
                    remainder -= 1
            # if still remainder (due to caps), spread to any class with spare
            if remainder > 0:
                for c in class_ids:
                    if remainder == 0:
                        break
                    spare = avail_per_class[c] - per_class_quota[c]
                    if spare > 0:
                        take = min(spare, remainder)
                        per_class_quota[c] += take
                        remainder -= take

        # build output dict
        sample_dict = {f"sample{i}": [] for i in range(number_of_clients)}

        # concrete indices per class (copy so label_dict remains untouched)
        class_indices = {c: list(label_dict[f"label{c}"]) for c in class_ids}
        for c in class_ids:
            rng.shuffle(class_indices[c])

        # ---- distribute each class quota across clients ----
        for c in class_ids:
            quota_c = int(per_class_quota[c])
            if quota_c <= 0:
                continue
            idxs = class_indices[c][:quota_c]  # exact count
            # NonIID vs IID client counts for this class
            if not NonIID:
                q, r = divmod(quota_c, number_of_clients)
                counts = np.full(number_of_clients, q, dtype=int)
                counts[:r] += 1
            else:
                probs = rng.dirichlet([alpha] * number_of_clients)
                counts = rng.multinomial(quota_c, probs)

            # assign indices to clients according to counts
            p = 0
            for i in range(number_of_clients):
                k = int(counts[i])
                if k > 0:
                    take = idxs[p:p + k]
                    p += k
                    sample_dict[f"sample{i}"].extend(take)

        # ---- enforce min-1 per client (borrow from largest holder) ----
        empties = [i for i in range(number_of_clients) if len(sample_dict[f"sample{i}"]) == 0]
        if empties:
            if target_total < number_of_clients:
                raise ValueError("Cannot enforce min-1: target_total < number_of_clients.")
            for i in empties:
                donor = max(range(number_of_clients), key=lambda j: len(sample_dict[f"sample{j}"]))
                if len(sample_dict[f"sample{donor}"]) <= 1:
                    raise ValueError("No donor with >1 samples to borrow from.")
                sample_dict[f"sample{i}"].append(sample_dict[f"sample{donor}"].pop())

        # tidy
        for i in range(number_of_clients):
            sample_dict[f"sample{i}"].sort()

        # consistency check (optional)
        assigned_total = sum(len(v) for v in sample_dict.values())
        if assigned_total != target_total:
            raise RuntimeError(f"Assigned total {assigned_total} != target_total {target_total}")

        return sample_dict


    def create_subsamples(self, sample_dict, x_data, y_data, x_name, y_name):
        """
        Build per-client tensors from global x/y and assigned indices.
        Guarantees at least 1 sample per client by borrowing one index
        from the client holding the largest shard (with >1 samples).
        """
        x_data_dict, y_data_dict = {}, {}
        num_clients = len(sample_dict)

        for i in range(num_clients):
            xname = f"{x_name}{i}"
            yname = f"{y_name}{i}"
            sample_name = f"sample{i}"

            # Ensure indices exist and are sorted
            indices = sample_dict.get(sample_name, [])
            indices.sort()

            # --- enforce min-1 per client (borrow 1 if empty) ---
            if not indices:
                donor = max(range(num_clients), key=lambda j: len(sample_dict.get(f"sample{j}", [])))
                donor_list = sample_dict.get(f"sample{donor}", [])
                if len(donor_list) <= 1:
                    raise ValueError(
                        "Cannot enforce min-1 per client; no donor with >1 samples. "
                        "Increase dirichlet_alpha, train_amount, or reduce number_of_clients."
                    )
                indices.append(donor_list.pop())
                indices.sort()
                sample_dict[sample_name] = indices  # keep dict consistent
            # ----------------------------------------------------

            x_info = torch.stack([x_data[j] for j in indices], dim=0).to(self.cuda_device)
            y_info = torch.as_tensor([y_data[j] for j in indices], device=self.cuda_device)

            x_data_dict[xname] = x_info
            y_data_dict[yname] = y_info

        return x_data_dict, y_data_dict


    # Training and Validation Functions
    @staticmethod
    def train_local_model(model, train_loader, criterion, optimizer, device=None):
        if device is None:
            device = next(model.parameters()).device
        model.train()
        train_loss = 0.0
        correct = 0

        for data, target in train_loader:
            # H2D here (fast path with pin_memory + non_blocking)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()

        return train_loss / len(train_loader), correct / len(train_loader.dataset)


    def validation(self, model, test_loader, criterion):
        model.eval()
        test_loss = float(0.0)
        correct = float(0.0)
        attack_success_rate_dict = {}

        targeted_goals = dict()
        if self.attack_method == 'UAM':
            if self.g_main not in ['untargeted']:
                for i in self.g_main:
                    targeted_goals[str(i[0]) + "," + str(i[1])] = [[0, 'label_flipping']]
        elif self.attack_method == 'explicit boosting':
            for i, param in self.malicious_dict.items():
                if param[0] not in ['untargeted1', 'untargeted2']:
                    # [['label_flipping', 1, 1, 7]]
                    key = str(param[2]) + "," + str(param[3])
                    if key not in targeted_goals.keys():
                        targeted_goals[key] = [[i, param[0]]]
                    else:
                        targeted_goals[key].append([i, param[0]])
        # from_label, to_label, Ns=0, Nt=0
        attack_success_rate_dict = {
            targeted_goal: [int(targeted_goal.split(',')[0]), int(targeted_goal.split(',')[1]), 0, 0] \
            for targeted_goal in targeted_goals.keys()}

        correct_pred = {classname: 0 for classname in self.class_names}
        total_pred = {classname: 0 for classname in self.class_names}
        per_class_accuracy_dict = {classname: 0 for classname in self.class_names}

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.cuda_device)
                target = target.to(self.cuda_device)

                # adjust data dimension
                if self.params['type'] == 'mnist':
                    # convert data from [64, 1, 28, 28] to [64, 784]
                    data = torch.flatten(data, 1)
                elif self.params['type'] in ['cifar10', 'fashion_mnist', 'emnist', 'kmnist', 'cifar100', 'imagenette']:
                    # print("data.shape", data.shape) # [64, 3, 32, 32] # 64 as 2 * batch size
                    data = data
                else:
                    warnings.warn("data type not in helper.validation")

                is_nan = torch.stack([torch.isnan(p).any() for p in self.main_model.parameters()]).any()
                if is_nan:
                    logger.info(f"Found 'nan' in the main model")

                output = model(data)

                model_output_na = False
                if torch.isnan(output).any():
                    model_output_na = True
                test_loss += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                prediction = prediction.flatten()
                target = target.flatten()

                for label, prediction in zip(target, prediction):
                    output_classname = self.label_classname_dict[int(label)]
                    if label == prediction:
                        correct_pred[output_classname] += 1
                    total_pred[output_classname] += 1

                    # Attack Success Rate with multi attack targets
                    for target_name, param in attack_success_rate_dict.items():
                        if label == int(param[0]):
                            attack_success_rate_dict[target_name][2] += 1
                            if prediction == int(param[1]):
                                attack_success_rate_dict[target_name][3] += 1

        test_loss /= len(test_loader)
        correct /= len(test_loader.dataset)

        for classname, correct_count in correct_pred.items():
            if total_pred[classname] != 0:
                per_class_accuracy = 100 * float(correct_count) / total_pred[classname]
            else:
                per_class_accuracy = -1
            per_class_accuracy_dict[classname] = per_class_accuracy

        for target_name, param in attack_success_rate_dict.items():
            # Nt / Ns
            if param[2] != 0:
                attack_success_rate = 100 * float(param[3]) / param[2]
            else:
                attack_success_rate = -1
            attack_success_rate_dict[target_name].append(attack_success_rate)
        return test_loss, correct, attack_success_rate_dict, per_class_accuracy_dict, model_output_na

    def get_gradients_bet_models(self, model_dict_before, model_dict_after):
        with torch.no_grad():
            for model_name in model_dict_after.keys():
                for layer_name in self.main_model.state_dict().keys():
                    get_layer_weight(model_dict_after[model_name], layer_name).data = \
                        get_layer_weight(model_dict_after[model_name], layer_name).clone() \
                        - get_layer_weight(model_dict_before[model_name], layer_name).clone()
        return model_dict_after

    # Functions for Federated Aggregation
    def start_train_end_node_process(self, malicious_attack, malicious_names, sample_list, iteration,
                                     plot_indiv, plot_glob, layer, distance_matrix, rl_mode, idx_clients):

        initial_result = self.create_similarity_matrix(self.model_dict, layer, distance_matrix)
        self.global_results.append([iteration, 0, layer, distance_matrix, initial_result])

        self.subset_dict = {key: value for key, value in self.model_dict.items() if int(key[5:]) in idx_clients}
        self.model_dict_before = get_weights_from_model_dict(self.subset_dict)

        for epoch in range(1, self.params['numEpoch'] + 1):
            for i in idx_clients:
                if malicious_attack is True and self.iteration_mali is True: 
                    train_ds = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                             self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                    train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)

                    if i in malicious_names:
                        if rl_mode == "untargeted_label_flip" or self.attack_method == 'explicit boosting':
                            train_ds = TensorDataset(self.x_train_m_dict["x_train_m_" + str(i)].to(self.cuda_device),
                                                     self.y_train_m_dict["y_train_m_" + str(i)].to(
                                                         self.cuda_device))  # malicious dataset
                            train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True )

                    model = self.model_dict[self.name_of_models[i]]
                    criterion = self.criterion_dict[self.name_of_criterions[i]]
                    optimizer = self.optimizer_dict[self.name_of_optimizers[i]]

                    # the local training process
                    train_loss, train_accuracy = self.train_local_model(model, train_dl, criterion, optimizer, device=self.cuda_device)
                else:
                    train_ds = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                             self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                    train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True )

                    model = self.model_dict[self.name_of_models[i]]
                    criterion = self.criterion_dict[self.name_of_criterions[i]]
                    optimizer = self.optimizer_dict[self.name_of_optimizers[i]]

                    train_loss, train_accuracy = self.train_local_model(model, train_dl, criterion, optimizer, device=self.cuda_device)
            if plot_indiv is True:
                for sample in sample_list:
                    self.plot_node_weights(self.model_dict, sample[0], sample[1], iteration)

            if plot_glob is True:
                iteration_result = self.create_similarity_matrix(self.model_dict, layer, distance_matrix)
                self.global_results.append([iteration, epoch, layer, distance_matrix, iteration_result])

        model_dict_after = get_weights_from_model_dict(self.subset_dict)

        honest_model_names = ["model" + str(x) for x in list(range(self.number_of_clients)) if
                              x not in self.malicious_names]
        selected_honest_model = honest_model_names[-1]

        grads_regular = self.get_gradients_bet_models(self.model_dict_before, model_dict_after)

        return grads_regular

    def create_model_optimizer_criterion_dict(self, list_of_clients):
        model_dict = dict()
        optimizer_dict = dict()
        criterion_dict = dict()

        for i in list_of_clients:
            model_name = "model" + str(i)
            # create based on the type of main model
            nn_model = eval(self.params['nn_model'])

            model_info = nn_model()
            model_info.to(self.cuda_device)
            model_dict.update({model_name: model_info})

            optimizer_name = "optimizer" + str(i)
            optimizer_info = torch.optim.SGD(model_info.parameters(),
                                             lr=self.params['learning_rate'], 
                                             momentum=self.params['momentum'],
                                             weight_decay=5e-4,)
            optimizer_dict.update({optimizer_name: optimizer_info})

            criterion_name = "criterion" + str(i)
            criterion_info = nn.CrossEntropyLoss()
            criterion_dict.update({criterion_name: criterion_info})

        return model_dict, optimizer_dict, criterion_dict

    def pre_aggregation_process(self, model_dict, pre_aggregation_method):
        name_of_models = list(model_dict.keys())
        number_of_clients = len(model_dict)
        per_agg_model_dict = {}

        if pre_aggregation_method == 'bucketing':
            # randomly partitions into n/s bucket, s = 2
            s = 2
            num_buckets = math.ceil(number_of_clients / s)
            model_list = name_of_models.copy()
            buckets_group = []

            while len(model_list) != 0:
                bucket = random.sample(model_list, s)
                model_list = list(set(model_list) - set(bucket))
                buckets_group.append(bucket)

            # averaging in the buckets
            for b in range(len(buckets_group)):
                bucket_dict = {key: model_dict[key] for key in buckets_group[b]}
                # averaging
                bucket_avg = self.get_aggregated_weights(bucket_dict, 'fedavg')

                nn_model = eval(self.params['nn_model'])
                bucket_model = nn_model()

                with torch.no_grad():
                    for name, param in bucket_model.named_parameters():
                        # print("bucket", name)
                        param.copy_(bucket_avg[name])

                bucket_model.to(self.cuda_device)

                # b_model_name = "model" + "_b_" + str(b)
                b_model_name = "model" + str(b)
                per_agg_model_dict[b_model_name] = bucket_model

        elif pre_aggregation_method == 'NNM':
            # f = 8 for n = 20
            f = 8
            num_neigh = number_of_clients - f

            # find neighbors for each client
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
                l2_dist = {k: {} for k in model_sample_weight_dict.keys()}

                for name1, value1 in model_sample_weight_dict.items():
                    for name2, value2 in model_sample_weight_dict.items():
                        if name1 != name2:
                            value1 = torch.flatten(value1)
                            value2 = torch.flatten(value2)
                            value1 = value1.unsqueeze_(0)
                            value2 = value2.unsqueeze_(0)
                            dist = torch.cdist(value1, value2, p=2)
                            # print("dist", dist)
                            # l2_dist[name1].append(dist.item())
                            l2_dist[name1][name2] = dist.item()

            # print("l2_dist", l2_dist)

            for client_name, dists in l2_dist.items():
                sorted_dists = sorted(dists.items(), key=lambda x: x[1])
                neighbors = sorted_dists[:num_neigh - 1]
                select_clients = [i[0] for i in neighbors] + [client_name]
                select_models = {key: model_dict[key] for key in select_clients}

                # averaging
                nnm_avg = self.get_aggregated_weights(select_models, 'fedavg')

                nn_model = eval(self.params['nn_model'])
                nnm_model = nn_model()

                with torch.no_grad():
                    for name, param in nnm_model.named_parameters():
                        param.copy_(nnm_avg[name])

                nnm_model.to(self.cuda_device)

                per_agg_model_dict[client_name] = nnm_model

        return per_agg_model_dict

    # def get_aggregated_weights(self, model_dict, aggregation_method):
    #     name_of_models = list(model_dict.keys())
    #     number_of_clients = len(model_dict)
    #     weight_accumulator = {}
    #     malicious_weight_sel = None

    #     # print("name_of_models[0]", name_of_models[0])
    #     for name, param in model_dict[name_of_models[0]].state_dict().items():
    #         struct = torch.zeros(size=param.size(), device=self.cuda_device)
    #         weight_accumulator[name] = struct

    #     # FedAvg
    #     if aggregation_method == 'fedavg':
    #         # print('agg method is fedavg')
    #         with torch.no_grad():
    #             for i in range(number_of_clients):
    #                 for key, value in weight_accumulator.items():
    #                     weight_accumulator[key] += get_layer_weight(model_dict[name_of_models[i]],
    #                                                                 key).data.clone()

    #             for key, value in weight_accumulator.items():
    #                 weight_accumulator[key] = value / number_of_clients
    #     elif aggregation_method == 'flame':
    #         # --- Build client updates relative to the previous server model ---
    #         with torch.no_grad():
    #             # You likely already have a handle to the server model.
    #             # If its name differs, replace `self.server_model` accordingly.
    #             server_sd = self.main_model.state_dict()

    #             # Build client_updates_dict: { "modelX": delta_dict }
    #             client_updates_dict = {}
    #             for cid, client_model in model_dict.items():
    #                 # client_model is an nn.Module in your codebase
    #                 client_sd = client_model.state_dict()
    #                 client_updates_dict[cid] = get_model_update(client_sd, server_sd)

    #             # FLAME hyperparams (safely fallback if missing)
    #             flame_noise = self.params.get('flame_noise', 1e-3)
    #             exp_dir     = self.params.get('exp_dir', "")
    #             cur_iter    = getattr(self, "current_iter", 0)

    #             # Run FLAME: returns (server_grad_delta, client_weights)
    #             server_grad, client_weights = flame_aggr(
    #                 server_model_state_dict=server_sd,
    #                 client_updates_dict=client_updates_dict,
    #                 noise=flame_noise,
    #                 exp_dir=exp_dir,
    #                 iter=cur_iter
    #             )

    #             # Merge delta into absolute new server state (absolute weights)
    #             new_server_sd = get_model_merged(server_grad, server_sd, assume_delta=True)

    #             # --- Logging in your house style ---
    #             malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
    #             ordered_client_w_values = self.ordered_client_weight_values(client_weights)
    #             self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

    #             logger_line_nonzero = []
    #             logger_line_zero    = []
    #             for name, w in client_weights.items():
    #                 tag = f"{name}_M" if (int(str(name)[5:]) in self.malicious_names) else name
    #                 pair = f"{tag}: {w:.4f}"
    #                 if w != 0:
    #                     logger_line_nonzero.append(pair)
    #                 else:
    #                     logger_line_zero.append(pair)

    #             logger.info(f"FLAME non-zero weights: {logger_line_nonzero}")
    #             logger.info(f"FLAME zero weights: {logger_line_zero}")

    #             # Convert state_dict to your expected return format (absolute)
    #             weight_accumulator = {}
    #             for key, param in model_dict[next(iter(model_dict))].state_dict().items():
    #                 # ensure same device/dtype as your pipeline expects
    #                 weight_accumulator[key] = new_server_sd[key].to(self.cuda_device).detach().clone()

    #     elif aggregation_method == 'geom_median':
    #         with torch.no_grad():
    #             model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
    #             geo_med, client_weights = geometric_median(model_sample_weight_dict, number_of_clients,
    #                                                        method='weiszfeld', options={})
    #             malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
    #             ordered_client_w_values = self.ordered_client_weight_values(client_weights)
    #             self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

    #             logger_line = []

    #             for i in self.malicious_sample:
    #                 key = "model" + str(i) + "_M"
    #                 pair = "{}: {:.1f}".format(key, client_weights["model" + str(i)])
    #                 logger_line.append(pair)

    #             for i in self.honest_sample:
    #                 key = "model" + str(i)
    #                 pair = "{}: {:.1f}".format(key, client_weights[key])
    #                 logger_line.append(pair)

    #             logger.info(f"client_weights in aggr: {logger_line}")
    #         weight_accumulator = self.weighted_average(model_dict, client_weights)
    #     elif aggregation_method == 'single-krum':
    #         weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=1)
    #     elif aggregation_method == 'multi-krum':
    #         weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=self.params['krum_m'])
    #     elif aggregation_method == 'foolsgold':
    #         with torch.no_grad():
    #             model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)

    #             for name, weight in model_sample_weight_dict.items():
    #                 if name in self.memory_dict.keys():
    #                     self.memory_dict[name].append(weight)
    #                     if len(self.memory_dict[name]) == self.history_len + 1:
    #                         # keep the history length not increasing
    #                         self.memory_dict[name].pop(0)
    #                 else:
    #                     self.memory_dict[name] = [weight]

    #             historical_weights = {}
    #             for name, weights in self.memory_dict.items():
    #                 for w in weights:
    #                     if name in historical_weights.keys():
    #                         historical_weights[name] += w
    #                     else:
    #                         historical_weights[name] = w

    #             client_weights = foolsgold(historical_weights)
    #             malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
    #             ordered_client_w_values = self.ordered_client_weight_values(client_weights)
    #             self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

    #             logger_line1 = []
    #             logger_line2 = []
    #             # logger unzero weights
    #             for name, weight in client_weights.items():
    #                 if weight != 0:
    #                     if int(name[5:]) in self.malicious_sample:
    #                         key = name + "_M"
    #                         pair = "{}: {:.4f}".format(key, client_weights[name])
    #                         logger_line1.append(pair)
    #                     else:
    #                         pair = "{}: {:.4f}".format(name, client_weights[name])
    #                         logger_line1.append(pair)
    #                 else:
    #                     if int(name[5:]) in self.malicious_sample:
    #                         key = name + "_M"
    #                         pair = "{}: {:.4f}".format(key, client_weights[name])
    #                         logger_line2.append(pair)
    #                     else:
    #                         pair = "{}: {:.4f}".format(name, client_weights[name])
    #                         logger_line2.append(pair)

    #             logger.info(f"foolsgold non zero weights: {logger_line1}")
    #             logger.info(f"foolsgold zero weights: {logger_line2}")

    #         weight_accumulator = self.weighted_average(model_dict, client_weights)
    #     elif aggregation_method == 'contra':
    #         logger.info(f"reputation_dict: {self.reputation_dict}")

    #         # clients selection based on reputation
    #         total = sum(self.reputation_dict.values(), 0.0)
    #         if total == 0:
    #             total = 1e-7
    #         self.reputation_dict = {k: v / total for k, v in self.reputation_dict.items()}
    #         sorted_keys = sorted(self.reputation_dict, key=self.reputation_dict.get, reverse=True)

    #         for w in sorted_keys:
    #             self.contra_sorted_dict[w] = (self.reputation_dict[w] * self.contra_f * (
    #                     1 - self.contra_f)) + self.contra_f
    #         key_list = list(self.contra_sorted_dict.keys())
    #         value_list = list(self.contra_sorted_dict.values())
    #         # in case value_list has negative values
    #         if min(value_list) < 0:
    #             value_list = [x - min(value_list) for x in value_list]
    #         agent_name_keys = np.random.choice(key_list, self.params["contra_clients_selection"],
    #                                            p=value_list / np.sum(value_list), replace=False)

    #         logger_line = []
    #         for name in agent_name_keys:
    #             if int(name[5:]) in self.malicious_names:
    #                 logger_line.append(name + "_M")
    #             else:
    #                 logger_line.append(name)
    #         logger.info(f"contra selected clients: {logger_line}")

    #         # Keep the history
    #         model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
    #         with torch.no_grad():
    #             for name, weight in model_sample_weight_dict.items():
    #                 if name in self.memory_dict.keys():
    #                     self.memory_dict[name].append(weight)
    #                     if len(self.memory_dict[name]) == self.history_len + 1:
    #                         # keep the history length not increasing
    #                         self.memory_dict[name].pop(0)
    #                 else:
    #                     self.memory_dict[name] = [weight]

    #             historical_weights = {}
    #             for name, weights in self.memory_dict.items():
    #                 for w in weights:
    #                     if name in historical_weights.keys():
    #                         historical_weights[name] += w
    #                     else:
    #                         historical_weights[name] = w

    #         # use foolsgold
    #         selected_model_dict = dict()

    #         with torch.no_grad():
    #             for name, weight in self.model_dict.items():
    #                 if name in agent_name_keys:
    #                     selected_model_dict[name] = weight
    #             contra_sample_weight_dict = self.sampling_weight_layer(selected_model_dict, self.sampling_layer)

    #         # update the reputation dict
    #         client_weights, self.reputation_dict = foolsgold_new(contra_sample_weight_dict, self.reputation_dict)

    #         # for not selected clients, their weights are 0
    #         for name in model_dict.keys():
    #             if name not in client_weights.keys():
    #                 client_weights[name] = 0

    #         malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
    #         ordered_client_w_values = self.ordered_client_weight_values(client_weights)
    #         self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

    #         # normalize the reputation
    #         maxr = max(self.reputation_dict.values())
    #         for k in self.reputation_dict:
    #             self.reputation_dict[k] = self.reputation_dict[k] / maxr

    #         weight_accumulator = self.weighted_average(model_dict, client_weights)
    #     elif aggregation_method == 'coordinate_wise_median':
    #         weight_accumulator = coordinate_wise_median(model_dict)
    #     elif aggregation_method == 'trimmed_mean':
    #         weight_accumulator = trimmed_mean(model_dict, self.params['trimmed_mean_m'])
    #     elif aggregation_method == 'bulyan':
    #         # Use the clients selected from multi-krum
    #         with torch.no_grad():
    #             model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
    #             muti_krum_m, krum_clients, client_weights = krum(model_sample_weight_dict, self.params['krum_f'],
    #                                                              number_of_clients - 2 * self.params['bulyan_m'])
    #             krum_selected_model_dict = {key: model_dict[key] for key in krum_clients}

    #             # trimmed_mean
    #             weight_accumulator = trimmed_mean(krum_selected_model_dict, 2 * self.params['bulyan_m'])
    #     else:
    #         warnings.warn("unknown aggregation method")

    #     if malicious_weight_sel is not None:
    #         logger.info(f"malicious weight selected %: {malicious_weight_sel * 100 :.2f}")

    #     return weight_accumulator


    def get_aggregated_weights(self, model_dict, aggregation_method):
        """
        Aggregates client models into an absolute server state_dict (not deltas).
        - Canonicalizes model names: "model3_M" -> "model3" (keys inside stay clean).
        - Uses self.malicious_ids (or self.malicious_names if that's what you have) only as metadata.
        - Keeps BN buffers out of FedAvg accumulator (server update already skips BN).
        """
        import numpy as np
        # ---- helpers ----
        def _norm_name(name: str) -> str:
            # "model3_M" -> "model3"; "model3" -> "model3"
            return name.split("_", 1)[0]

        def _cid(name: str) -> int:
            # assumes canonical "model<id>"
            n = _norm_name(name)
            return int(n[5:])

        mal_id_set = set(getattr(self, "malicious_ids", getattr(self, "malicious_names", [])))

        def _is_mal(name: str) -> bool:
            try:
                return _cid(name) in mal_id_set
            except Exception:
                return False

        def _log_client_weights(title: str, client_weights: dict):
            nz, z = [], []
            for name, w in client_weights.items():
                label = f"{_norm_name(name)}{'_M' if _is_mal(name) else ''}: {w:.4f}"
                (nz if w != 0 else z).append(label)
            logger.info(f"{title} non-zero weights: {nz}")
            logger.info(f"{title} zero weights: {z}")

        # ---- canonicalize input dict keys once ----
        model_dict = { _norm_name(k): v for k, v in model_dict.items() }
        name_of_models = list(model_dict.keys())
        number_of_clients = len(model_dict)

        weight_accumulator = {}
        malicious_weight_sel = None

        # initialize accumulator shape/device
        first_sd = model_dict[name_of_models[0]].state_dict()
        for name, param in first_sd.items():
            weight_accumulator[name] = torch.zeros_like(param, device=self.cuda_device)

        # ---------------- FedAvg ----------------
        if aggregation_method == 'fedavg':
            with torch.no_grad():
                server_sd = self.main_model.state_dict()
                client_updates = {nm: model_dict[nm].state_dict() for nm in name_of_models}

                # use our helper; do NOT average BN buffers
                avg_sd, _ = fedavg(server_sd, client_updates, average_bn_buffers=True)

                # build the return dict as absolute weights (keep server BN buffers)
                # use averaged tensors for ALL keys (fedavg() already handles BN/int counters)
                weight_accumulator = {
                    k: avg_sd[k].to(self.cuda_device).clone()
                    for k in server_sd.keys()
                }


        # ---------------- FLAME ----------------
        elif aggregation_method == 'flame':
            with torch.no_grad():
                server_sd = self.main_model.state_dict()

                # deltas per client
                client_updates_dict = {}
                for nm in name_of_models:
                    client_updates_dict[nm] = get_model_update(model_dict[nm].state_dict(), server_sd)

                flame_noise = self.params.get('flame_noise', 1e-3)
                exp_dir     = self.params.get('exp_dir', "")
                cur_iter    = getattr(self, "current_iter", 0)

                server_grad, client_weights = flame_aggr(
                    server_model_state_dict=server_sd,
                    client_updates_dict=client_updates_dict,
                    noise=flame_noise,
                    exp_dir=exp_dir,
                    iter=cur_iter
                )

                # absolute new server weights
                new_server_sd = get_model_merged(server_grad, server_sd, assume_delta=True)

                # logging / CSV
                malicious_weight_sel = self.malicious_weight_percentage(client_weights, list(mal_id_set)) \
                    if hasattr(self, "malicious_weight_percentage") else None
                if hasattr(self, "ordered_client_weight_values"):
                    ordered = self.ordered_client_weight_values(client_weights)
                    if hasattr(self, "weights_csv_logger"):
                        self.weights_csv_logger.append_csv(ordered + ([malicious_weight_sel] if malicious_weight_sel is not None else []))
                _log_client_weights("FLAME", client_weights)

                # fill accumulator with absolute weights
                weight_accumulator = {k: new_server_sd[k].to(self.cuda_device).detach().clone()
                                    for k in first_sd.keys()}

        # ---------------- Geometric Median ----------------
        elif aggregation_method == 'geom_median':
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(model_dict, self.sampling_layer)
                geo_med, client_weights = geometric_median(model_sample_weight_dict, number_of_clients,
                                                        method='weiszfeld', options={})
                malicious_weight_sel = self.malicious_weight_percentage(client_weights, list(mal_id_set)) \
                    if hasattr(self, "malicious_weight_percentage") else None
                if hasattr(self, "ordered_client_weight_values"):
                    ordered = self.ordered_client_weight_values(client_weights)
                    if hasattr(self, "weights_csv_logger"):
                        self.weights_csv_logger.append_csv(ordered + ([malicious_weight_sel] if malicious_weight_sel is not None else []))
                _log_client_weights("geom_median", client_weights)

            weight_accumulator = self.weighted_average(model_dict, client_weights)

        # ---------------- Krum variants ----------------
        elif aggregation_method == 'single-krum':
            weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=1)

        elif aggregation_method == 'multi-krum':
            weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=self.params['krum_m'])

        # ---------------- Foolsgold ----------------
        elif aggregation_method == 'foolsgold':
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(model_dict, self.sampling_layer)

                # build history
                for name, weight in model_sample_weight_dict.items():
                    self.memory_dict.setdefault(name, [])
                    self.memory_dict[name].append(weight)
                    if len(self.memory_dict[name]) > self.history_len:
                        self.memory_dict[name].pop(0)

                historical_weights = {}
                for name, weights in self.memory_dict.items():
                    for w in weights:
                        historical_weights[name] = historical_weights.get(name, 0) + w

                if len(historical_weights) == 0:
                    client_weights = {k: 1.0 for k in model_dict.keys()}
                elif len(historical_weights) == 1:
                    only = next(iter(historical_weights.keys()))
                    client_weights = {k: (1.0 if k == only else 0.0) for k in model_dict.keys()}
                else:
                    client_weights = foolsgold(historical_weights)

                malicious_weight_sel = self.malicious_weight_percentage(client_weights, list(mal_id_set)) \
                    if hasattr(self, "malicious_weight_percentage") else None
                if hasattr(self, "ordered_client_weight_values"):
                    ordered = self.ordered_client_weight_values(client_weights)
                    if hasattr(self, "weights_csv_logger"):
                        self.weights_csv_logger.append_csv(ordered + ([malicious_weight_sel] if malicious_weight_sel is not None else []))
                _log_client_weights("foolsgold", client_weights)

            weight_accumulator = self.weighted_average(model_dict, client_weights)

        # ---------------- Contra (FG on a subset with reputation) ----------------
        elif aggregation_method == 'contra':
            logger.info(f"reputation_dict: {self.reputation_dict}")

            # normalize reputation
            total = sum(self.reputation_dict.values()) or 1e-7
            self.reputation_dict = {k: v / total for k, v in self.reputation_dict.items()}
            sorted_keys = sorted(self.reputation_dict, key=self.reputation_dict.get, reverse=True)

            # contra score
            for w in sorted_keys:
                self.contra_sorted_dict[w] = (self.reputation_dict[w] * self.contra_f * (1 - self.contra_f)) + self.contra_f

            # 1) restrict to this round's active clients
            active_keys = set(model_dict.keys())
            key_list = [k for k in self.contra_sorted_dict.keys() if k in active_keys]

            # 2) build probs aligned to key_list
            value_list = np.array([self.contra_sorted_dict[k] for k in key_list], dtype=float)
            if value_list.size == 0:
                # no active keys? fall back to using all model_dict keys uniformly
                key_list = list(model_dict.keys())
                value_list = np.ones(len(key_list), dtype=float)

            # 3) sanitize scores and ensure enough positive mass
            if not np.all(np.isfinite(value_list)):
                value_list = np.ones_like(value_list, dtype=float)

            mn = np.min(value_list) if value_list.size else 0.0
            if mn < 0:
                value_list = value_list - mn  # shift negatives to zero

            # 4) clamp how many to draw
            sel_n = min(int(self.params["contra_clients_selection"]), len(key_list))
            sel_n = max(1, sel_n)

            # build a probability vector 'p' that has at least sel_n positive entries
            pos_idx = np.where(value_list > 0)[0]

            if pos_idx.size < sel_n:
                # fallback: pick the top-sel_n by score and sample uniformly over them
                if len(key_list) == 0:
                    agent_name_keys = []
                else:
                    take = min(sel_n, len(key_list))
                    top_idx = np.argsort(value_list)[-take:]
                    p = np.zeros_like(value_list, dtype=float)
                    p[top_idx] = 1.0 / take
                    agent_name_keys = np.random.choice(key_list, take, p=p, replace=False)
            else:
                # normalize over positive entries only
                p = np.zeros_like(value_list, dtype=float)
                p[pos_idx] = value_list[pos_idx] / np.sum(value_list[pos_idx])
                agent_name_keys = np.random.choice(key_list, sel_n, p=p, replace=False)

            logger.info(f"contra selected clients: "
                        f"{[n + ('_M' if int(n[5:]) in self.malicious_names else '') for n in agent_name_keys]}")

            # strip the cosmetic "_M" marker before filtering
            agent_name_keys_norm = {k.replace("_M", "") for k in agent_name_keys}

            # history (only for selected; normalized keys)
            model_sample_weight_dict = self.sampling_weight_layer(
                {k: v for k, v in model_dict.items() if k in agent_name_keys_norm}, self.sampling_layer
            )
            for name, weight in model_sample_weight_dict.items():
                self.memory_dict.setdefault(name, [])
                self.memory_dict[name].append(weight)
                if len(self.memory_dict[name]) > self.history_len:
                    self.memory_dict[name].pop(0)

            # foolsgold on selected (guard tiny pools)
            if len(model_sample_weight_dict) == 0:
                client_weights = {k: 1.0 for k in model_dict.keys()}
            elif len(model_sample_weight_dict) == 1:
                only = next(iter(model_sample_weight_dict.keys()))
                client_weights = {k: (1.0 if k == only else 0.0) for k in model_dict.keys()}
            else:
                client_weights, self.reputation_dict = foolsgold_new(model_sample_weight_dict, self.reputation_dict)
                # give 0 to non-selected
                for name in model_dict.keys():
                    client_weights.setdefault(name, 0.0)

            # renormalize reputation [0,1]
            maxr = max(self.reputation_dict.values()) or 1.0
            for k in self.reputation_dict:
                self.reputation_dict[k] = self.reputation_dict[k] / maxr

            weight_accumulator = self.weighted_average(model_dict, client_weights)


        # ---------------- Coord-median / Trimmed mean / Bulyan ----------------
        elif aggregation_method == 'coordinate_wise_median':
            weight_accumulator = coordinate_wise_median(model_dict)

        elif aggregation_method == 'trimmed_mean':
            weight_accumulator = trimmed_mean(model_dict, self.params['trimmed_mean_m'])

        elif aggregation_method == 'bulyan':
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(model_dict, self.sampling_layer)
                muti_krum_m, krum_clients, client_weights = krum(model_sample_weight_dict, self.params['krum_f'],
                                                                number_of_clients - 2 * self.params['bulyan_m'])
                krum_selected_model_dict = {key: model_dict[key] for key in krum_clients}
                weight_accumulator = trimmed_mean(krum_selected_model_dict, 2 * self.params['bulyan_m'])
        else:
            warnings.warn("unknown aggregation method")

        if malicious_weight_sel is not None:
            logger.info(f"malicious weight selected %: {malicious_weight_sel * 100 :.2f}")

        return weight_accumulator


    def krum_get_weight_accumulator(self, model_dict, krum_m):
        with torch.no_grad():
            model_sample_weight_dict = self.sampling_weight_layer(model_dict, self.sampling_layer)

            # Returns: m_used, krum_clients, client_weights, client_scores
            m_used, krum_clients, client_weights, client_scores = krum(
                model_sample_weight_dict,
                int(self.params['krum_f']),
                int(krum_m)
            )

            # --- log scores (lower is better) ---
            logger_line = [f"{nm}: {score:.1f}" for nm, score in client_scores.items()]
            logger.info(f"krum score: {logger_line}")

            # NEW: log which clients were chosen
            logger.info(f"krum selected: {krum_clients}")

            # warn if malicious chosen
            mistakes = [nm for nm in krum_clients if any(str(mid) in nm for mid in self.malicious_names)]
            if mistakes:
                logger.info(f"mistakenly chosen: {mistakes}")

            # Optional: log mixing weights
            if hasattr(self, "ordered_client_weight_values"):
                ordered = self.ordered_client_weight_values(client_weights)
                mal_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
                if hasattr(self, "weights_csv_logger"):
                    self.weights_csv_logger.append_csv(ordered + [mal_sel])

        # Mix with weights returned from krum()
        weight_accumulator = self.weighted_average(model_dict, client_weights)
        return weight_accumulator


    def malicious_weight_percentage(self, client_weights, malicious_names):
        malicious_weight = 0
        for key, value in client_weights.items():
            if int(key[5:]) in malicious_names:
                # print("key", key, value)
                malicious_weight += value
        total_weights = sum(client_weights.values())
        return malicious_weight / total_weights

    def ordered_client_weight_values(self, client_weights):
        sortedDict = sorted(client_weights.items(), key=lambda x: int(x[0][5:]))
        values = [t[1] for t in sortedDict]
        return values

    def sampling_weight_layer(self, model_dict, sampling_layer):
        model_sample_weight_dict = {}
        for key, value in model_dict.items():
            model_sample_weight_dict[key] = get_layer_weight(model_dict[key], sampling_layer)
        return model_sample_weight_dict

    def check_model_integrity(self, model_dict):
        with torch.no_grad():
            aggregation_sample = self.sampling_weight_layer(model_dict, self.sampling_layer)

            # randomly select another layer as checking layer
            random_sample = np.random.choice(
                [x for x in self.params['available_layers'] if x not in [self.sampling_layer]], \
                1, replace=False)

        return

    def weighted_average(self, model_dict, clients_weights):
        """Computes weighted average of atoms with specified weights
        """
        tot_weights = sum(clients_weights.values())
        weighted_updates = dict()
        name_of_models = list(model_dict.keys())

        for name, param in model_dict[name_of_models[0]].state_dict().items():
            struct = torch.zeros(size=param.size(), device=self.cuda_device)
            weighted_updates[name] = struct
            
        for client_name in model_dict:
            for name, param in weighted_updates.items():
                weighted_updates[name] += (clients_weights[client_name] / tot_weights) * get_layer_weight(
                    model_dict[client_name], name)

        return weighted_updates

    def update_main_model_use_aggregated_weights(self, main_model, model_dict, iteration, lr):
        if self.pre_aggregation_method == 'NNM':
            pre_agg_model_dict = self.pre_aggregation_process(model_dict, self.pre_aggregation_method)
            weight_accumulator = self.get_aggregated_weights(pre_agg_model_dict, self.aggregation_method)
        elif self.pre_aggregation_method == 'bucketing':
            pre_agg_model_dict = self.pre_aggregation_process(model_dict, self.pre_aggregation_method)
            weight_accumulator = self.get_aggregated_weights(pre_agg_model_dict, self.aggregation_method)
        else:
            weight_accumulator = self.get_aggregated_weights(model_dict, self.aggregation_method)

        # ---- NEW: BN buffer detection (do NOT aggregate these) ----
        bn_buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
        def is_bn_buffer(key: str) -> bool:
            return key.endswith(bn_buffer_suffixes)

        with torch.no_grad():
            # Use FedAvg semantics by default unless you explicitly want momentum-like blending
            if self.params.get('server_lr_fix', True):
                lr = 1.0
            else:
                lr = float(lr)
            # First round: just copy (for non-BN keys)
            if iteration == 0 or lr >= 0.999999:
                for name, _ in main_model.state_dict().items():
                    dst = get_layer_weight(main_model, name).data
                    src = weight_accumulator[name].to(self.cuda_device)
                    # match dtype
                    if dst.dtype.is_floating_point:
                        dst.copy_(src.to(dst.dtype))
                    else:
                        # e.g., num_batches_tracked (int64)
                        dst.copy_(src.to(dst.dtype))
            else:
                # Blend old and new (for non-BN keys)
                logger.info(f"server lr: {lr}")
                for name, _ in main_model.state_dict().items():
                    dst = get_layer_weight(main_model, name).data
                    src = weight_accumulator[name].to(self.cuda_device)
                    if dst.dtype.is_floating_point:
                        old = dst
                        new = src.to(dst.dtype)
                        dst.copy_((1.0 - lr) * old + lr * new)
                    else:
                        # integer buffers (e.g., num_batches_tracked): cannot blend -> copy new
                        dst.copy_(src.to(dst.dtype))
        
        return main_model

    def send_main_model_to_nodes_and_update_model_dict(self, main_model, model_dict):
        name_of_models = list(model_dict.keys())
        with torch.no_grad():
            for i in range(len(model_dict)):
                for name, param in main_model.state_dict().items():

                    get_layer_weight(model_dict[name_of_models[i]], name).data = \
                        get_layer_weight(main_model, name).data.clone()

        return model_dict

    # random sampling based on parameters on layer selection
    def random_sampling_nodes(self, model, each_layer_sample_num):
        layers = []
        num = 0
        for name, param in model.state_dict().items():
            layers.append((num, name, list(param.size())))
            num += 1
        logger.info(
            f"Network Structure: {str(model.module.name if isinstance(model, nn.DataParallel) else model.name)}")

        for i in layers:
            logger.info(f"{i}")

        weights_sample = []

        for s in each_layer_sample_num:
            layer_num = int(s[0])
            sample_num = int(s[1])
            for i in range(sample_num):
                random_weight = list((np.random.randint(sample_num, layers[layer_num][-1])))
                weights_sample.append([layer_num, random_weight])
        logger.info(f"weights_sample: {weights_sample}")
        return weights_sample

    # plot the similarity matrix of the avg distance or similarity between clients
    def create_similarity_matrix(self, model_dict, layer, distance_matrix):
        d_func = Distance(self.cuda_device)
        sample_weights = []
        # print("model_dict", model_dict)
        for i in range(self.params['number_of_clients']):
            current_model = "model" + str(i)
            # print("layer", layer)
            w_i = get_layer_weight(model_dict[current_model], layer).data
            sample_weights.append(w_i)

        resultTable = []
        for i in range(len(sample_weights)):
            distances = []
            for j in range(len(sample_weights)):
                if distance_matrix == "l2":
                    d = d_func.l2dist(sample_weights[i], sample_weights[j])
                elif distance_matrix == "cos":
                    d = d_func.cosine_similarity(sample_weights[i], sample_weights[j], False)
                elif distance_matrix == 'cka':
                    d = d_func.kernel_CKA(sample_weights[i], sample_weights[j])
                distances.append(d.item())
            resultTable.append(distances)
        return (resultTable)
    
    
    def sample_clients_for_round(self):
        N, k = self.num_clients, self.k_per_round

        # If selecting all, short-circuit (works for both modes)
        if k >= N:
            return list(range(N))

        if self.selection_mode == "random":
            # Uniform draw from all clients
            return self._rng.choice(N, size=k, replace=False).tolist()

        if self.selection_mode == "fixed_pct":
            # k to pick this round
            N, k = self.num_clients, self.k_per_round

            # how many malicious to select this round
            m_target = int(round(k * float(self.malicious_selection_ratio)))
            m_target = max(0, min(k, m_target))  # clamp
            b_target = k - m_target

            # ---- Malicious: always [0, 1, 2, ..., m_target-1] ----
            fixed_mal_ids = list(range(m_target))
            logger.info(f"fixed malicious IDs this round: {fixed_mal_ids}")
            # logger.info(f"self.malicious_ids {self.malicious_ids}")

            # (optional safety) ensure these are actually in your malicious pool
            # comment this out if you already guarantee 0.. are malicious at init
            assert all(mid in self.malicious_ids for mid in fixed_mal_ids), \
                f"Fixed malicious IDs {fixed_mal_ids} must be in self.malicious_ids."

            # ---- Benign: random draw each round ----
            benign_pool = [i for i in range(N) if i not in self.malicious_ids]
            if len(benign_pool) < b_target:
                raise ValueError(f"Not enough benign clients: need {b_target}, have {len(benign_pool)}")

            sel_b = self._rng.choice(benign_pool, size=b_target, replace=False).tolist() if b_target else []

            # Do NOT shuffle so 0,1,2 (etc.) remain visible in logs
            out = fixed_mal_ids + sel_b
            return out

        else: ValueError(f"Unknown selection mode: {self.selection_mode}")


    def log_once(self, key, msg):
        if key not in self._logged_once:
            logger.info(msg)
            self._logged_once.add(key)
            
            
