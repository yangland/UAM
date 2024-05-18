from pathlib import Path
import os
import numpy as np
import pandas as pd
import math
import random
import logging
import os
import warnings
import copy

import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from models.mnist_3l import Mnist_3L
from models.mnist_1l import Mnist_1L
from models.cifar10_6l import Cifar10_6L
from models.resnet18 import ResNet18
# from models.cifar10_resnet18 import ResNet18
from models.cifar10_9l import Cifar10_9L
from models.fashion_cnn import FashionCNN
from models.emnist_cnn import EmnistCNN
from models.cifar10_googlenet import GoogLeNet
from models.cifar10_8l import Cifar10_8L
from models.cifar10_18l import Cifar10_18L
from models.cifar10_vgg import vgg11, vgg13, vgg16
from models.kmnist_12l import Kmnist_12L
from models.cifar100_mobilenetv2 import mobilenetv2
from models.cifar100_mobilenet import mobilenet
from models.cifar100_resnet import resnet18
from csv_logging import Csv_logging
from distance_matrix import Distance
from collections import Counter

from aggr.geometric_median import geometric_median
from aggr.krum import krum
from aggr.foolsgold import foolsgold, foolsgold_new, pendulum, foolsgold_contra
from aggr.coordinate_wise_median import coordinate_wise_median, trimmed_mean
from aggr.mani_byzan import our_attack_dist, our_attack_score
# from rl.naive_mcts import *
# from rl.naive_mcts_simp import *
from rl.neldermead import *
from collections import OrderedDict

logger = logging.getLogger("logger")


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        # self.rl_based_attack = None
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
        if self.params['rl_manual']:
            self.rl_action = np.asarray([float(i) for i in self.params['manual_coordinates']])
        else:
            self.rl_action = np.asarray([float('Inf') for i in self.params['x_start']])
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
        # self.sigmoid_c2 = self.params['sigmoid_c2']
        # self.sigmoid_a1 = self.params['sigmoid_a1']
        self.effective_action_domain = self.params['effective_action_domain']
        self.empirical_points = self.params['empirical_points'].copy()
        self.rounds_before_restart = self.params['rounds_before_restart']
        self.rl_mode = self.params['rl_mode']
        self.pre_aggregation_method = self.params['pre_aggregation_method']
        # self.domain_corners = []

        if self.params.get('select_precentage') is None or not self.params['select_precentage']:
            self.clients_selection = None
        else:
            self.clients_selection = int(self.params['select_precentage'] * self.number_of_clients / 100)

        if self.if_series:
            self.master_path = f'./logs/series_{self.name}_{series_time}'
            self.folder_path = f'{self.master_path}/{self.name}_{current_time}'
        else:
            self.folder_path = f'./logs/{self.name}_{current_time}'
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log_{self.name}_{current_time}.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

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

    # Create the main model and initialization
    def fl_initialization(self):
        # logger.info("helper.fl_initialization debugging 1: create main model")
        if self.cuda_device.type != 'cpu':
            logger.info(
                f"device: {self.cuda_device} - {self.cuda_device.type} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
            # memories = [x / 1024 ** 2 for x in torch.cuda.mem_get_info()]
            # logger.info(f"free: {memories[0]} MB, total: {memories[1]} MB")
        else:
            logger.info(f"device: {self.cuda_device.type}")
        nn_model = eval(self.params['nn_model'])
        self.main_model = nn_model()
        self.main_model.to(self.cuda_device)
        # nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=100)
        # gpu_ids = self.params['gpu_ids']
        # self.main_model = nn.DataParallel(self.main_model, device_ids=gpu_ids)
        # print(f"nn_model = eval({self.params['nn_model']})")
        # print(f"self.main_model = nn.DataParallel(self.main_model, device_ids={gpu_ids})")

        self.main_criterion = nn.CrossEntropyLoss()

        # logger.info("helper.fl_initialization debugging 2: create_model_optimizer_criterion_dict")
        self.model_dict, self.optimizer_dict, self.criterion_dict = self.create_model_optimizer_criterion_dict(
            list(range(self.params['number_of_clients'])))
        self.sample_nodes = self.random_sampling_nodes(self.main_model, self.params['each_layer_list'])

        self.name_of_models = list(self.model_dict.keys())
        self.name_of_optimizers = list(self.optimizer_dict.keys())
        self.name_of_criterions = list(self.criterion_dict.keys())

        # logger.info("helper.fl_initialization debugging 3: send_main_model_to_nodes_and_update_model_dict")
        self.model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict)
        # logger.info("helper.fl_initialization debugging 3.5: get_malicious_name")
        self.attack_method = self.params['attack_method']
        self.malicious_names, self.malicious_dict, self.honest_names = self.get_malicious_name()

        if len(self.malicious_names) == 0:
            self.params['malicious_attack'] = False
        else:
            self.params['malicious_attack'] = True
        # For the malicious clients create a seperated model_dict for undisclosed, analysing purposed training
        # self.model_dict_m, self.optimizer_dict_m, self.criterion_dict_m = self.create_model_optimizer_criterion_dict(
        #     self.malicious_names)

        # logger.info("helper.fl_initialization debugging 4: sampling honest, malicious clients")
        # sampling 4 honest, 4 malicious clients
        self.malicious_sample = random.sample(self.malicious_names, min(4, self.params['number_of_malicious']))
        self.honest_sample = random.sample([i for i in list(range(self.number_of_clients))
                                            if i not in self.malicious_names], 4)

        # logger.info("helper.fl_initialization debugging 5: initialize reputation score dictionary")
        # reputation score starts with 1.0
        self.reputation_dict = {"model" + str(i): 1 for i in range(self.number_of_clients)}
        self.contra_sorted_dict = dict()

        # logger.info("helper.fl_initialization debugging 6: RL-based attack initialization")
        # self.rl_based_attack = self.params['rl_based_attack']

        # print("self.attack_method", self.attack_method)

        # create the object of Naive_Mcts class
        if self.attack_method == 'RL':
            # {0: ['label_flipping', [10, 3, 31, 8, 17, 37], [[1, 5],[5,1]]]}
            # print("self.malicious_dict", self.malicious_dict)
            # malicious_mode = self.malicious_dict[0][0]
            # malicious_group_list = self.malicious_dict[0][1]
            # print("self.malicious_dict", self.malicious_dict)
            # if self.malicious_dict == {}:
            #     self.g_main = [[-1, -1], [-1, -1]]
            # else:
            #     self.g_main = [[self.malicious_dict[0][2], self.malicious_dict[0][3]],
            #                [self.malicious_dict[0][3], self.malicious_dict[0][2]]]
            #TODO fixing the RL for untargeted
            self.g_main = [[-1, -1], [-1, -1]]

            """
            self.mali_mcts = Naive_Mcts_Simp(name='malicious_1', malicious_mode=malicious_mode,
                                             num_classes=self.num_classes,
                                             malicious_group_list=malicious_group_list,
                                             goal=goal, pi_0_eps=self.params['pi_0_eps'],
                                             init_search_len=self.params['init_search_len'])
            """

            self.neldermead = NelderMead(name='NelderMead1', x_start=self.rl_action, step=self.step, progress=True,
                                         rl_manual=self.params['rl_manual'],
                                         manual_coordinates=self.params['manual_coordinates'],
                                         empirical_points=self.empirical_points)

            # print("self.neldermead.empirical_points", self.neldermead.empirical_points)
        return None

    def create_honest_malicious_data(self):
        sample_dict_train = self.load_data()

        # print("sample_dict_train ['sample0']", sample_dict_train['sample0'])
        # print("sample_dict_train.keys()", sample_dict_train.keys())
        self.x_train_dict, self.y_train_dict = self.create_subsamples(sample_dict=sample_dict_train,
                                                                      x_data=self.x_train,
                                                                      y_data=self.y_train, x_name="x_train",
                                                                      y_name="y_train")

        self.name_of_x_train_sets = list(self.x_train_dict.keys())  # length is the # of clients
        self.name_of_y_train_sets = list(self.y_train_dict.keys())

        # spot check self.y_train_dict
        logger.info("spot check on y_train_dict:")
        keys = np.random.choice(list(self.y_train_dict), size=2, replace=False)
        # print("keys", keys)
        for key in keys:
            # print("self.y_train_dict[key]", self.y_train_dict[key])
            # print("self.y_train_dict[key].size()", self.y_train_dict[key].size())
            c = Counter(self.y_train_dict[key].tolist())
            logger.info(f"{key} total {str(len(self.y_train_dict[key]))}: {str(sorted(c.items()))}")

        if self.params['malicious_attack'] is True:
            if self.attack_method != 'RL':
                logger.info(
                    f"with {self.attack_method} malicious attack, {len(self.malicious_dict)} group(s), malicious clients id:")
            else:
                logger.info(f"with RL malicious attack, {len(self.malicious_dict)} group, malicious clients id:")
            logger.info(f"malicious names: {self.malicious_names}")
            logger.info(f"malicious proportion: {self.malicious_dict}")

            # select malicious names from self.x_train_dict, self.y_train_dict, create a pooled dataset for the malicious group
            # self.pool_dl work like self.god_valid_dl
            x_pool = []
            y_pool = []
            # print("self.malicious_names", self.malicious_names)
            for i in self.malicious_names:
                # print("i", i)
                # print("self.name_of_x_train_sets[i]", self.name_of_x_train_sets[i])
                # print("self.x_train_dict[self.name_of_x_train_sets[i]].size()",
                #       self.x_train_dict[self.name_of_x_train_sets[i]].size())  # torch.Size([1069, 784])
                # print("self.y_train_dict[self.name_of_y_train_sets[i]].size()",
                #       self.y_train_dict[self.name_of_y_train_sets[i]].size())

                x_pool.append(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device))
                y_pool.append(self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))

            x_pool = torch.cat(x_pool, dim=0)
            y_pool = torch.cat(y_pool, dim=0)
            # print("x_pool", x_pool.size())
            # print("y_pool", y_pool.size())

            logger.info(
                f"Classes distribution on the pooled data (one malicious group, {len(self.malicious_names)} members):")
            c = Counter(y_pool.tolist())
            # print("Pooled " + str(len(self.malicious_names)) + ": " + "total " + str(len(y_pool)) + ": " + str(sorted(c.items())))
            logger.info(f"total {len(y_pool)}: {sorted(c.items())}")

            pool_ds = TensorDataset(x_pool, y_pool)
            self.pool_dl = DataLoader(pool_ds, batch_size=self.params['batch_size'], shuffle=True)

            # change for two-models, based on input malicious_proportion or RL based
            # print("self.attack_method", self.attack_method)
            if self.attack_method == 'explicit boosting':
                # in non-RL based attack the malicious dataset is fixed, created before the train_iteration starts
                for i, param in self.malicious_dict.items():
                    # print("i, param", i, param)
                    # malicious_mode = param[0]
                    # malicious_group_list = param[1]
                    # from_label = param[2]
                    # to_label = param[3]

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
            elif self.attack_method == 'RL':
                # in RL based attack, the malicious dataset in changing in each iteration
                assert len(self.malicious_dict) == 1, "RL based attack only support 1 group of malicious"
                # RL based attack need different malicious dataset every iteration
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

        for i in range(self.params['iterations']):
            logger.info("\nIteration {}:".format(int(i + 1)))
            mali_loss_record, mali_accuracy_record, mali_asr_record, model_output_na = self.recording_validation(i,
                                                                                                                 "mali")
            if model_output_na:
                logger.info(f"model output nan, stop at epoch: {i}")
                break

            # checking on the main_model
            # logger.info(f"Debugging self.main_model weights "
            #             f"{get_layer_weight(self.main_model, self.params['weights_layer'])}")

            # 1. Send the main model to the local models
            self.model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict)
            # model_dict_m also got the model from server
            # self.model_dict_m = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict_m)

            if self.params['malicious_attack']:
                if self.attack_method == 'RL':
                    # 1. Record the reward from the previous round with the current feedback
                    # self.mali_mcts.record(mali_accuracy_record, mali_asr_record)
                    self.neldermead.record(mali_accuracy_record, self.rl_action, progress=True)

                    # 2. Naive MCTS with arm expansion
                    # action = self.mali_mcts.mcts()

                    # 2. NelderMead
                    if not self.neldermead.rl_manual:
                        # first check the points in the empirical_points
                        # print("self.neldermead.empirical_points2", self.neldermead.empirical_points)
                        if len(self.neldermead.empirical_points) != 0:
                            empirical_action = random.choice(self.neldermead.empirical_points)
                            self.neldermead.empirical_points.remove(empirical_action)
                            self.rl_action = empirical_action
                        else:
                            if self.neldermead.init_spot_search:
                                # find the best among the empirical_points
                                # print("self.neldermead.history", self.neldermead.history)
                                rewards_empirical = [row[2] for row in self.neldermead.history[1:]]
                                # print("rewards_empirical", rewards_empirical)
                                best_empirical = \
                                    self.neldermead.history[rewards_empirical.index(min(rewards_empirical)) + 1][0]
                                # print("best_empirical", best_empirical)
                                self.neldermead.restart(new_start=best_empirical)
                                self.rl_action = self.neldermead.nelder_mead()
                                self.neldermead.init_spot_search = False
                            else:
                                # If reached rounds_before_restart, randomly restart from domain without edge
                                if self.neldermead.reach_restart_threshold(self.rounds_before_restart):
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
                                self.rl_action = self.neldermead.nelder_mead()
                                while self.rl_action is False:
                                    self.rl_action = self.neldermead.nelder_mead()
                    else:
                        self.rl_action = self.neldermead.manual_coordinates

                    logger.info(f"NelderMead's action: {self.rl_action}")
                else:
                    warnings.warn("Please use rl_manual to set up manual attack, set rl_based_sttack = True")
            else:
                pass
            
            # 1.5 select clients for this round's FL if self.clients_selection = True
            # if self.clients_selection:
            #     selected_model_dict = (self.model_dict, self.number_of_clients, self.clients_selection)

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
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            # Normalize the weights
            # with torch.no_grad():
            #     self.main_model.weight.div_(torch.norm(self.main_model.weight, dim=1, keepdim=True))

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
        # print("god_loss_record", god_loss_record)
        # print("god_accuracy_records", god_accuracy_records)
        # print("god_asr_records", god_asr_records)
        return self.global_results, god_accuracy_records, god_loss_records, god_asr_records

    def process_with_local_training(self, iteration):
        """
        if self.rl_action is not None:
            logger.info(f"Naive MCTS's action: {action.name}")
        else:
            logger.info(f"Naive MCTS's action: {action}")
        g_iter, g_locals = self.malicious_goals_on_action(action)
        """

        if len(self.malicious_names) != 0:
            goal_conc, attack_magnitude = action_to_params(self.rl_action,
                                                           self.sigmoid_c1,
                                                           self.params['action_conv_func'], self.params['rl_mode'])
            # print("goal_conc, attack_magnitude", goal_conc, attack_magnitude)
            if self.attack_method == 'RL':
                if self.rl_mode == "untargeted_label_flip":
                    g_locals = self.malicious_goals_on_goal_conc(goal_conc)
                    logger.info("g_locals: ")
                    for key, value in g_locals.items():
                        logger.info(f"{key}: {dict(value)}")

                    # use g_local to create local malicious datasets
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
                # manual based label flipping attack (untarget1)
                # the x_train_m_dict and y_train_m_dict do not need to be updated every iteration
                pass
        else:
            attack_magnitude = 1.0
            goal_conc = 0
            # if self.params['rl_uniform']:
            #     goal_conc = None
            # print("goal_conc: ", goal_conc)
        # 3. Group training based on the honest datasets and the malicious datasets (current iteration)
        # same as the rest of the code in iteration
        # self.params['respect_server_model'] = action.details['respect_server']
        # self.params['malicious_group_agg'] = action.details['mali_group_agg']
        # self.params['reverse_grad'] = action.details['reverse_grad']

        # elif self.params['malicious_attack'] and (self.params['rl_based_attack'] is False):
        # whether malicious clients respect model from server in the iterations starts from 1 (not 0)

        # honest_model_names = ["model" + str(x) for x in list(range(self.number_of_clients)) if
        #                       x not in self.malicious_names]
        # honest_model_dict = dict((k, self.model_dict[k]) for k in honest_model_names)
        # """
        # if self.params['respect_server_model']:
        #     # All clients get the main model
        #     self.model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model, self.model_dict)
        # else:
        #     # Honest clients still gets the server's model
        #     honest_model_dict = self.send_main_model_to_nodes_and_update_model_dict(self.main_model,
        #                                                                             honest_model_dict)
        #
        # if self.params['malicious_attack'] and self.params['malicious_group_agg']:
        #     # Use FedAvg to aggregate malicious model and send to group's malicious client
        #     malicious_model_dict = dict()
        #     for param in self.malicious_dict.values():
        #         group_names = ["model" + str(s) for s in param[1]]
        #         group_dict = dict((k, self.model_dict[k]) for k in group_names)
        #         # Avg
        #         weight_accumulator = self.get_aggregated_weights(group_dict, "fedavg")
        #
        #         # Select one model as a main model for the malicious group
        #         malicious_main = group_dict[group_names[0]]
        #         other_m_members = {i: group_dict[i] for i in group_dict if i != group_names[0]}
        #
        #         # if self.params['reverse_grad'] is False:
        #
        #         for name, param in malicious_main.state_dict().items():
        #             get_layer_weight(malicious_main, name).data = weight_accumulator[name].data.clone()
        #
        #         # Distribute to all other malicious clients in the group
        #         malicious_model_local = self.send_main_model_to_nodes_and_update_model_dict(malicious_main,
        #                                                                                     other_m_members)
        #         # Add back the malicious main
        #         malicious_model_dict[group_names[0]] = malicious_main
        #
        #         # else:
        #         #     # reverse the malicious_main's gradients
        #         #     for name, param in malicious_main.state_dict().items():
        #         #         get_layer_weight(malicious_main, name).data = torch.neg(weight_accumulator[name].data).clone()
        #         #     # print("if self.params['reverse_grad'] is True")
        #         #     malicious_model_local = self.send_main_model_to_nodes_and_update_model_dict(malicious_main,
        #         #                                                                                 other_m_members)
        #         #     # need to reverse the group_names[0]
        #         #     malicious_model_dict[group_names[0]] = malicious_main
        #         malicious_model_dict = {**malicious_model_dict, **malicious_model_local}
        # else:
        # """
        # malicious_model_names = ["model" + str(x) for x in list(range(self.number_of_clients)) if
        #                          x in self.malicious_names]
        # malicious_model_dict = dict((k, self.model_dict[k]) for k in malicious_model_names)
        #
        # self.model_dict = {**malicious_model_dict, **honest_model_dict}

        # print("self.model_dict", self.model_dict) # missing
        # print("malicious_model_dict", malicious_model_dict.keys())
        # print("honest_model_dict", honest_model_dict.keys())

        select_model_dict, idx_clients = client_random_selection(self.model_dict, self.number_of_clients, self.clients_selection)

        if bool(set(idx_clients) & set(self.malicious_names)):
            print("This iteration has malicious client(s)")
            self.iteration_mali=True
        else:
            self.iteration_mali=False

        logger.info("training on the end nodes...")
        grads_regular = self.start_train_end_node_process(self.params['malicious_attack'],
                                                          self.malicious_names,
                                                          self.sample_nodes, iteration + 1,
                                                          self.params['plot_individual'],
                                                          self.params['plot_global'],
                                                          self.params['weights_layer'],
                                                          self.params['distance_matrix'],
                                                          self.rl_mode, idx_clients)
        if len(self.malicious_names) != 0:
            self.attacker_modify_trained_weights(grads_regular, attack_magnitude, goal_conc, self.rl_mode)
        return None

    def attacker_modify_trained_weights(self, grads_regular, attack_magnitude, goal_conc, rl_mode):
        with torch.no_grad():
            if self.attack_method == 'explicit boosting':
                K = 5 # 10
                # boost towards the direction of malicious training
                for model_name in self.model_dict.keys():
                    if int(model_name[5:]) in self.malicious_names:
                        for layer_name, param in self.main_model.state_dict().items():
                            get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                get_layer_weight(self.model_dict_before[model_name], layer_name).clone() \
                                + K * \
                                get_layer_weight(grads_regular[model_name], layer_name).clone()
            elif self.attack_method == 'little':
                # 1. average the malicious clients as mean
                # 2. Z_max = 0.08, based on n = 20, m = 5
                # 3. minus Z_max * std from mean
                # {key: model_dict[key] for key in buckets_group[b]}
                # Z_max = 0.5
                Z_max = 0.08

                malicious_models = {key: self.model_dict["model" + str(key)] for key in self.malicious_names}
                mean = self.get_aggregated_weights(malicious_models, 'fedavg')
                std_dict = {}

                for layer_name, param in self.main_model.state_dict().items():
                    layer_list = []
                    for model_name in self.model_dict.keys():
                        if int(model_name[5:]) in self.malicious_names:
                            model_dict_after = get_weights_from_model_dict(self.model_dict)
                            layer_weight = get_layer_weight(model_dict_after[model_name], layer_name).clone()
                            layer_list.append(layer_weight)
                    stacked_layer_w = torch.stack(layer_list)
                    # get standard deviation
                    std = torch.std(stacked_layer_w.float(), dim=0, correction=0)
                    std_dict[layer_name] = std

                for model_name in self.model_dict.keys():
                    if int(model_name[5:]) in self.malicious_names:
                        for layer_name, param in self.main_model.state_dict().items():
                            get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                mean[layer_name].float() + Z_max * std_dict[layer_name].float()
                logger.info("deliver little attack")
            elif self.attack_method in ['min-max', 'min-sum']:
                n_attacker = 5
                malicious_models = {key: self.model_dict["model" + str(key)] for key in self.malicious_names}
                mean = self.get_aggregated_weights(malicious_models, 'fedavg')
                mal_updates = {}

                for layer_name, param in self.main_model.state_dict().items():
                    layer_list = []
                    for model_name in self.model_dict.keys():
                        if int(model_name[5:]) in self.malicious_names:
                            model_dict_after = get_weights_from_model_dict(self.model_dict)
                            layer_weight = get_layer_weight(model_dict_after[model_name], layer_name).clone()
                            layer_list.append(layer_weight)
                    stacked_layer_w = torch.stack(layer_list)

                    if self.attack_method == 'min-max':
                        mal_update = our_attack_dist(stacked_layer_w, mean[layer_name], n_attacker,
                                                     self.cuda_device, dev_type='unit_vec')
                    elif self.attack_method == 'min-sum':
                        mal_update = our_attack_score(stacked_layer_w, mean[layer_name], n_attacker,
                                                      self.cuda_device, dev_type='unit_vec')
                    mal_updates[layer_name] = mal_update

                for model_name in self.model_dict.keys():
                    if int(model_name[5:]) in self.malicious_names:
                        for layer_name, param in self.main_model.state_dict().items():
                            get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                mal_updates[layer_name]

            elif self.attack_method == 'min-sum':
                pass
            elif self.attack_method == 'RL':
                if rl_mode == "untargeted_label_flip":
                    for model_name in self.model_dict.keys():
                        if int(model_name[5:]) in self.malicious_names:
                            for layer_name, param in self.main_model.state_dict().items():
                                get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                    get_layer_weight(self.model_dict_before[model_name], layer_name).clone() \
                                    + attack_magnitude * \
                                    get_layer_weight(grads_regular[model_name], layer_name).clone()
                        else:
                            # leave the weights as it is
                            pass
                elif rl_mode == "reverse_grad":
                    if goal_conc is None:
                        for model_name in self.subset_dict.keys():
                            if int(model_name[5:]) in self.malicious_names:
                                for layer_name, param in self.main_model.state_dict().items():
                                    # logger.info("reverse_grad weights calculation")
                                    # logger.info(f"Malicious weight before {get_layer_weight(self.model_dict[model_name], layer_name)}")
                                    # logger.info(f"Malicious grads {get_layer_weight(grads_regular[model_name], layer_name)}")
                                    if self.params.get('Median_att') is None or not self.params['Median_att']:
                                        get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                            get_layer_weight(self.model_dict_before[model_name], layer_name).clone() \
                                            - attack_magnitude * \
                                            get_layer_weight(grads_regular[model_name], layer_name).clone()
                                    else:
                                        # median attack - beta * median( vi )
                                        # blackbox, only the malicious clients
                                        if self.params.get('White_box') is None or not self.params['White_box']:
                                            client_list = self.malicious_names
                                        else:
                                            client_list = self.honest_names
                                        client_names = ["model" + str(s) for s in client_list]
                                        model_list = {key: self.model_dict[key] for key in client_names}

                                        # model_median = coordinate_wise_median(model_list)

                                        model_median = self.get_aggregated_weights(model_list, 'fedavg')
                                        # print("model_median is FedAvg")

                                        # print("keys of model", model_median.keys())
                                        # print("layer_name", layer_name)
                                        # attack_magnitude = 1000
                                        get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                            - model_median[layer_name].clone() * attack_magnitude

                    else:
                        for layer_name, param in self.main_model.state_dict().items():
                            layers = []
                            for i in self.malicious_names:
                                model_name = "model" + str(i)
                                single_layer = torch.flatten(
                                    get_layer_weight(self.model_dict[model_name], layer_name).data.clone())
                                layers.append(single_layer)
                                # print("single_layer", single_layer)
                            t = torch.stack(layers)
                            mean_t = torch.mean(t.float(), 0)

                            for i in self.malicious_names:
                                model_name = "model" + str(i)
                                shape = get_layer_weight(self.model_dict[model_name], layer_name).data.size()
                                single_layer = torch.flatten(
                                    get_layer_weight(self.model_dict[model_name], layer_name).data.clone())
                                v_i_prime = (1 - goal_conc) * single_layer + goal_conc * mean_t
                                # normalization
                                v_i_prime = v_i_prime / torch.norm(v_i_prime) * torch.norm(mean_t)
                                v_i_prime = v_i_prime.view(shape)

                                get_layer_weight(self.model_dict[model_name], layer_name).data = \
                                    get_layer_weight(self.model_dict_before[model_name], layer_name).clone() \
                                    - attack_magnitude * v_i_prime
                elif rl_mode == "zeros":
                    for model_name in self.model_dict.keys():
                        if int(model_name[5:]) in self.malicious_names:
                            for layer_name, param in self.main_model.state_dict().items():
                                layer_size = get_layer_weight(self.model_dict[model_name], layer_name).data.size()
                                get_layer_weight(self.model_dict[model_name], layer_name).data = torch.zeros(
                                    layer_size).to(
                                    self.cuda_device)
                else:
                    warnings.warn(f"Can not recognize this rl_mode {rl_mode}")

        return None

    def malicious_goals_on_goal_conc(self, goal_conc):
        g_main = self.g_main
        # miss_mapped_classes = [i[0] for i in g_main]
        # g_iter = None
        g_locals = {}
        """
        alpha1 = action.details['alpha1']
        beta1 = action.details['beta1']
        beta2 = action.details['beta2']
        if_reverse = action.details['reverse_grad']
        # print("alpha1", alpha1)
        # print("beta1", beta1)
        # print("beta2", beta2)

        # get g_iter
        if if_reverse is False:
            # 1. randomly select alpha1 # of pairs in the g_main
            if alpha1 == len(g_main):
                g_iter = dict(g_main)
            else:
                idx = np.random.randint(len(g_main), size=alpha1)
                # print("g_main", g_main)
                # print("alpha_1", alpha1)
                # print("idx", idx)
                g_iter = dict(np.asarray(g_main)[idx, :])

            if beta2 != 0:
                g_iter = self.goal_randomizer(g_iter, beta2)
        else:
            # keep g_main as correct mapping and same with g_iter
            g_main = []
            for i in list(range(self.num_classes)):
                g_main.append([i, i])

            g_iter = dict(g_main)

        # get g_locals
        if beta1 == 0:
            for i in self.malicious_names:
                g_locals[i] = g_iter
        else:
            for i in self.malicious_names:
                # print("beta1 not 0")
                g_locals[i] = self.goal_randomizer(g_iter, beta1)
                # print("g_locals[i]", g_locals[i])
        # print("g_locals on action", g_locals)
        """

        for i in self.malicious_names:
            g_locals[i] = self.goal_randomizer(g_main, goal_conc)
        return g_locals

    def goal_randomizer(self, goal, goal_conc):
        """
        all_classes = list(range(0, self.num_classes))
        selected_classes = np.random.choice(all_classes, beta, replace=False)
        selected_classes.sort()
        new_goal = OrderedDict()

        # for from_label in selected_classes:
        #     if from_label in goal.keys():
        #         # if incorrect, correct it
        #         del goal[from_label]
        #     else:
        #         # if correct, randomly assign a mismatch label
        #         available_classes = [item for item in list(range(0, self.num_classes)) if
        #                              item not in [from_label]]
        #         to_label = random.choice(available_classes)
        #         goal[from_label] = to_label

        # just random
        for from_label in selected_classes:
            available_classes = [item for item in list(range(0, self.num_classes))]
            to_label = random.choice(available_classes)
            new_goal[from_label] = to_label

        # to-do mix of unchosen old goal and the new goal
        """
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
            if self.attack_method == 'RL':
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

    def get_malicious_name(self):
        """
        # old self.malicious_name as list
        # Create list for clients who are Malicious
        np.random.seed(self.params['random_seed'])
        malicious_names = np.random.choice(range(self.params['number_of_clients']), \
                                           self.params['number_of_malicious'], replace=False)
        malicious_names.sort()
        """
        # new self.malicious_name as a dictionary
        malicious_dict = dict()
        malicious_selected = np.random.choice(range(self.params['number_of_clients']), \
                                              self.params['number_of_malicious'], replace=False)
        malicious_selected.sort()
        honest_selected = [i for i in list(range(self.params['number_of_clients'])) if i not in malicious_selected]
        malicious_preprocessing = list(malicious_selected)
        # logger.info("helper.get_malicious_name debugging 1: create malicious_selected")

        if self.attack_method in ['explicit boosting', 'traditional']:
            malic_prop = self.params['malicious_proportion']

            sum_prop = 0
            for i in range(len(malic_prop)):
                # ensure the num of malicious and the malic_prop % are match
                assert (self.params['number_of_malicious'] * malic_prop[i][1]) % 1 <= 0.001, \
                    "num of malicious * each malic_prop % has to be int"
                sum_prop += malic_prop[i][1]
                group_selected = np.random.choice(malicious_preprocessing, \
                                                  int(self.params['number_of_malicious'] * malic_prop[i][1]),
                                                  replace=False)

                malicious_mode = malic_prop[i][0]
                if malic_prop[i][0] in ['untargeted1', 'untargeted2']:
                    # -1 due to untargeted
                    from_label = -1
                    to_label = -1
                else:
                    from_label = malic_prop[i][2]
                    to_label = malic_prop[i][3]
                malicious_dict[i] = [malicious_mode, list(group_selected), from_label, to_label]
                malicious_preprocessing = [x for x in malicious_preprocessing if x not in group_selected]

            assert sum_prop - 1 <= 0.001, "malic_prop % have to sum to 1"
            # print(malicious_selected)
            # [1  3  4  5  8 10 13 17 18 23 28 31 34 36 37 38]
            # print(malicious_dict)
            # {0: ['backdoor', [5, 1, 18, 13, 34, 4, 28, 38, 36], 2, 7], 1: ['label_flipping', [10, 3, 31, 8, 17, 37], 1, 5]}
        elif self.attack_method == 'little':
            malicious_mode = 'None'
            goal = [-1, -1]
            malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
        # elif self.attack_method == 'explicit boosting':
        #     malicious_mode = 'untargeted'
        #     to_map = list(range(self.num_classes))
        #     goal = create_mismatch_mapping(to_map)
        #     malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
        elif self.attack_method == 'RL':
            # in RL model the main goal can be ['untargeted'], ['random_pair'] or any pairs of labels, such as [1,7]
            # logger.info("helper.get_malicious_name debugging 2: rl_based_attack is True")
            malicious_mode = 'None'

            if self.params['Goal'] == 'untargeted':
                malicious_mode = 'untargeted'
                to_map = list(range(self.num_classes))
                goal = create_mismatch_mapping(to_map)
            elif self.params['Goal'] == 'random_pair':
                [from_label, to_label] = np.random.choice(self.params['num_classes'], 2, replace=False)
                goal = [[from_label, to_label], [to_label, from_label]]
                malicious_mode = 'label_flipping'
            elif type(self.params['Goal']) == list:
                malicious_mode = 'label_flipping'
                goal = self.params['Goal']
            else:
                warnings.warn('Unknown attack_method!')
            malicious_dict[0] = [malicious_mode, malicious_preprocessing, goal]
            # logger.info("helper.get_malicious_name debugging 4: malicious_dict created")
            print("malicious_dict", malicious_dict)
        return list(malicious_selected), malicious_dict, honest_selected

    def create_rl_malicious_datasets(self, x_train_dict, y_train_dict, malicious_names, g_locals):
        x_train_m_dict = {}
        y_train_m_dict = {}

        for i in malicious_names:
            # print("i malicious names", i)
            x_train_m_dict["x_train_m_" + str(i)] = x_train_dict["x_train" + str(i)]
            # modify y labels based on the action dictionary
            y_modified = []
            for j in range(len(y_train_dict['y_train' + str(i)])):
                # print("j", j)
                # print("g_locals[i].keys()", g_locals[i].keys())
                # print("y_train_dict['y_train' + str(i)][j].cpu()", y_train_dict['y_train' + str(i)][j].cpu())
                if y_train_dict['y_train' + str(i)][j].cpu().item() not in g_locals[i].keys():
                    # for classes not in the action, keep the same
                    y_modified.append(y_train_dict['y_train' + str(i)][j].cpu())
                else:
                    # for classes included in the action, modify based on the g_local dict
                    y_modified.append(torch.tensor(g_locals[i][y_train_dict['y_train' + str(i)][j].cpu().item()]))
                    # print("rl dataset:")
                    # print(y_train_dict['y_train' + str(i)][j].cpu().item())
                    # print(g_locals[i][y_train_dict['y_train' + str(i)][j].cpu().item()])
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
                            # print("current_x.size()",current_x.size()) # [3, 32, 32]
                            # print("current_x[0]", current_x[0])
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
        # print("build_classes_dict", classes_dict)
        # print("classes_dict.keys()", classes_dict.keys())
        # print("classes_dict['label7']", classes_dict['label7'])
        return classes_dict, list(class_ids)

    def get_x_y_datasets(self, dataset):
        x_data = []
        y_data = []
        for ind, data in enumerate(dataset):  # for cifar: 50000
            x, y = data
            # print("x.size()", x.size()) #[3, 32, 32]
            # print("y", y) # class
            # adjust data dimension
            if self.params['type'] == 'mnist':
                x_data.append(torch.flatten(x[0]))
            elif self.params['type'] == 'cifar10':
                x_data.append(x)
            elif self.params['type'] == 'fashion_mnist':
                x_data.append(x)
            elif self.params['type'] == 'emnist':
                x_data.append(x)
            elif self.params['type'] == 'kmnist':
                x_data.append(x)
            elif self.params['type'] == 'cifar100':
                x_data.append(x)
            elif self.params['type'] == 'imagenette':
                x_data.append(x)
            y_data.append(y)
        x_data = torch.stack(x_data)
        y_data = torch.as_tensor(y_data)

        # print("x_data.size()", x_data.size()) # ([50000, 3, 32, 32])
        # print("y_data.size()", y_data.size()) # ([50000])
        # print("y_data", y_data)
        return (x_data, y_data)

    # Data Loading and Processing Functions
    def load_data(self):
        transform_vanilla = transforms.Compose([transforms.ToTensor(), ])
        if self.params['type'] == 'mnist':
            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transform_vanilla)
        elif self.params['type'] == 'cifar10':
            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_vanilla)
        elif self.params['type'] == 'fashion_mnist':
            self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform_vanilla)
        elif self.params['type'] == 'emnist':
            self.train_dataset = datasets.EMNIST('./data', train=True, download=True, transform=transform_vanilla,
                                                 split="letters")
            self.test_dataset = datasets.EMNIST('./data', train=False, transform=transform_vanilla, split="letters")
        elif self.params['type'] == 'kmnist':
            self.train_dataset = datasets.KMNIST('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset = datasets.KMNIST('./data', train=False, transform=transform_vanilla)
        elif self.params['type'] == 'cifar100':
            self.train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_vanilla)
            self.test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_vanilla)
        elif self.params['type'] == 'imagenette':
            transform = transforms.Compose([
                    transforms.CenterCrop(160),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            self.train_dataset = datasets.ImageFolder("./data/imagenette2-160/train", transform)
            self.test_dataset = datasets.ImageFolder("./data/imagenette2-160/val", transform)
        else:
            warnings.warn("Data type not recognized in load_data")

        # print("self.test_dataset", self.test_dataset)
        logger.info('reading data done')

        self.classes_dict, self.class_ids = self.build_classes_dict()

        # order for class ids when the id in y data not start with 0
        self.label_classname_dict = dict(zip(self.class_ids, self.class_names))
        # print("self.label_classname_dict", self.label_classname_dict)

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

    def get_subsamples_indices(self, label_dict, number_of_clients, amount, num_classes, class_ids, NonIID, alpha):
        # print("label_dict keys", label_dict.keys())
        sample_dict = dict()
        batch_size = int(math.floor(amount / number_of_clients))

        if NonIID == False:
            for i in range(number_of_clients):  # create 100 number_of_clients
                sample_name = "sample" + str(i)
                sample_list = []
                # IID distribution, each number class gets same samples
                for j in range(num_classes):  # for each number 0 - 9
                    label_name = str("label") + str(j)
                    a = label_dict[label_name][i * batch_size:(i + 1) * batch_size]
                    sample_list.extend(a)
                sample_dict.update({sample_name: np.asarray(sample_list)})  # each sample 450, 100 samples
        else:
            image_nums = []
            sample_dict = {}

            # created the keys for sample_dict
            for i in range(number_of_clients):
                sample_name = "sample" + str(i)
                sample_dict.update({sample_name: []})

            if self.params.get("if_quantity_based_noniid") is None or self.params["if_quantity_based_noniid"] is False:
                # distribution based label skew based of Dirichlet distribution
                for label in class_ids:
                    image_num = []
                    sampled_probabilities = amount * np.random.dirichlet(
                        np.array(number_of_clients * [alpha]))
                    class_label_len = len(label_dict[str("label") + str(label)])

                    for sample in range(number_of_clients):
                        # at least 1 sample image per client
                        no_imgs = math.ceil(sampled_probabilities[sample])
                        label_name = str("label") + str(label)

                        cut_off = min(class_label_len, no_imgs)
                        sampled_list = label_dict[label_name][:cut_off]
                        label_dict[label_name] = label_dict[label_name][cut_off:]
                        image_num.append(len(sampled_list))
                        dumb2 = sampled_list

                        class_label_len = class_label_len - len(sampled_list)
                        sample_name = "sample" + str(sample)
                        image_nums.append(image_num)
                        sample_dict[sample_name] = sample_dict[sample_name] + dumb2
            else:
                # quantity-based label skew, limit the class number (theta) for client
                theta = self.params['quantity_based_theta']
                sets_per_client = math.floor(self.num_classes * theta)
                sets_per_class = {}

                for label, indices in label_dict.items():
                    sets_per_class[label] = math.floor(len(indices) / (number_of_clients * theta)) - 1

                for sample in range(number_of_clients):
                    # print(sample, "class_ids", class_ids, sets_per_client)
                    if len(class_ids) > sets_per_client:
                        selected_class_ids = random.sample(class_ids, sets_per_client)
                    else:
                        selected_class_ids = class_ids

                    for label in selected_class_ids:
                        label_name = str("label") + str(label)
                        class_label_len = len(label_dict[label_name])
                        cut_off = min(class_label_len, sets_per_class[label_name])
                        sampled_list = label_dict[label_name][:cut_off]
                        label_dict[label_name] = label_dict[label_name][cut_off:]

                        sample_name = "sample" + str(sample)
                        sample_dict[sample_name] = sample_dict[sample_name] + sampled_list

                        if len(label_dict[label_name]) < sets_per_class[label_name] * 0.1:
                            class_ids.remove(label)

        return sample_dict

    def create_subsamples(self, sample_dict, x_data, y_data, x_name, y_name):
        # print("sample_dict keys", sample_dict.keys())
        x_data_dict = dict()
        y_data_dict = dict()

        for i in range(len(sample_dict)):  ### len(sample_dict)= number of clients
            xname = x_name + str(i)
            yname = y_name + str(i)
            sample_name = "sample" + str(i)
            # print("sample_name", sample_name)
            # print("sample_dict[sample_name]", sample_dict[sample_name])

            # use the global index to reconnect x and y
            sample_dict[sample_name].sort()
            x_info = [x_data[j] for j in sample_dict[sample_name]]
            # print("x_info", x_info)
            # print("i", i)
            if x_info == []:
                print("i", i)
            x_info = torch.stack(x_info)
            x_info.to(self.cuda_device)
            x_data_dict.update({xname: x_info})
            y_info = torch.as_tensor([y_data[j] for j in sample_dict[sample_name]])
            y_info.to(self.cuda_device)
            y_data_dict.update({yname: y_info})

        # print("x_data_dict.keys()", x_data_dict.keys())
        # print("y_data_dict", y_data_dict)
        return (x_data_dict, y_data_dict)

    # Training and Validation Functions
    @staticmethod
    def train_local_model(model, train_loader, criterion, optimizer):
        model.train()
        train_loss = 0.0
        correct = 0

        for data, target in train_loader:
            # print("In training:")
            # # print("data", data)
            # print("data size", data.size()) #[32, 3, 32, 32]
            # print("target", target)
            # print("target", target.size())

            output = model(data)

            # print("train output", output)
            # print("train target", target)

            loss = criterion(output, target)

            # print("loss", loss)
            optimizer.zero_grad()
            loss.backward()
            # added model clipping, did not work
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            optimizer.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        return train_loss / len(train_loader), correct / len(train_loader.dataset)

    def validation(self, model, test_loader, criterion):
        model.eval()
        test_loss = float(0.0)
        correct = float(0.0)
        attack_success_rate_dict = {}

        targeted_goals = dict()
        if self.attack_method == 'RL':
            if self.g_main not in ['untargeted']:
                for i in self.g_main:
                    targeted_goals[str(i[0]) + "," + str(i[1])] = [[0, 'label_flipping']]
        elif self.attack_method == 'explicit boosting':
            # print("self.malicious_dict", self.malicious_dict)
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
                elif self.params['type'] == 'cifar10':
                    # print("data.shape", data.shape) # [64, 3, 32, 32] # 64 as 2 * batch size
                    data = data
                elif self.params['type'] == 'emnist':
                    data = data
                elif self.params['type'] == 'kmnist':
                    data = data
                elif self.params['type'] == 'fashion_mnist':
                    data = data
                else:
                    warnings.warn("data type not in helper.validation")

                # print("data.is_cuda()", data.is_cuda)
                # print("next(net.parameters()).is_cuda", next(model.parameters()).is_cuda)
                # print("data.shape", data.shape)
                # print("target", target)

                # not because of 'nan' in the main model
                is_nan = torch.stack([torch.isnan(p).any() for p in self.main_model.parameters()]).any()
                if is_nan:
                    logger.info(f"Found 'nan' in the main model")

                output = model(data)
                # logger.info(f"test_loss1, {test_loss}")
                # logger.info(f"output, {output}")
                # logger.info(f"target, {target}")

                model_output_na = False
                if torch.isnan(output).any():
                    model_output_na = True
                    # logger.info(f"validation server model output is NA, {torch.isnan(output).any()}")
                    # logger.info(f"data for server model sample, {data}")
                    # logger.info(f"data max and min, {torch.max(data), torch.min(data)}")

                # logger.info(f"target has NA, {torch.isnan(target).any()}")
                # logger.info(f"criterion(output, target), {criterion(output, target)}")
                # logger.info(f"criterion(output, target).item(), {criterion(output, target).item()}")
                test_loss += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                # logger.info(f"prediction, {prediction}")
                # print("output", output) # all lines are the same
                # print("prediction", prediction) # only one class
                # print("prediction size", prediction.size()) # 64
                # print("target", target) # different digits
                # print("target", target.size()) # 64

                # print("prediction.eq(target.view_as(prediction))", prediction.eq(target.view_as(prediction)))
                # print("prediction.eq(target.view_as(prediction)).sum()", prediction.eq(target.view_as(prediction)).sum())
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                # logger.info(f"correct1, {correct}")

                # per class accuracy
                prediction = prediction.flatten()
                target = target.flatten()

                # print("self.class_ids", self.class_ids)
                # print("self.class_names", self.class_names)

                for label, prediction in zip(target, prediction):
                    # label_out = self.class_names[self.class_ids.index(label)]
                    output_classname = self.label_classname_dict[int(label)]
                    if label == prediction:
                        correct_pred[output_classname] += 1
                    total_pred[output_classname] += 1

                    # # Attack Success Rate
                    # if label == self.from_label:
                    #     Ns += 1
                    #     if prediction == self.to_label:
                    #         Nt += 1

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
            # print("classname, per_class_accuracy", classname, per_class_accuracy)

        # print("attack_success_rate_dict", attack_success_rate_dict)

        for target_name, param in attack_success_rate_dict.items():
            # Nt / Ns
            if param[2] != 0:
                attack_success_rate = 100 * float(param[3]) / param[2]
            else:
                attack_success_rate = -1
            attack_success_rate_dict[target_name].append(attack_success_rate)

        # logger.info(f"test_loss2, {test_loss}")
        # logger.info(f"correct2, {correct}")
        return test_loss, correct, attack_success_rate_dict, per_class_accuracy_dict, model_output_na

    """
    def get_gradients_bet_weights(self, weights_dict_before, model_dict_after, model_dict):
        gradients = {}
        for model_name in model_dict.keys():
            gradients[model_name] = {}
            for layer_name in self.main_model.state_dict().keys():
                gradient_layer_i = model_dict_after[model_name][layer_name] - weights_dict_before[model_name][
                    layer_name]
                gradients[model_name][layer_name] = gradient_layer_i
        return gradients
    """

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

        # if rl_mode == "untargeted_label_flip":
        #     # save the model weights before training
        #     model_dict_before = get_weights_from_model_dict(self.model_dict)
        #     self.model_dict_before = model_dict_before
        # elif rl_mode == "reverse_grad":
        #     model_dict_before_m = get_weights_from_model_dict(self.model_dict_m)
        # print("idx_clients", idx_clients)
        self.subset_dict = {key: value for key, value in self.model_dict.items() if int(key[5:]) in idx_clients}
        # print("self.subset_dict", self.subset_dict)
        self.model_dict_before = get_weights_from_model_dict(self.subset_dict)

        for epoch in range(1, self.params['numEpoch'] + 1):
            for i in idx_clients:
                if malicious_attack is True and self.iteration_mali is True: 
                    """
                    if i not in malicious_names:
                        # honest clients uses original dataset
                        train_ds = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                                 self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                        train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)
                    else:
                        if rl_mode == "reverse_grad":
                            # always train the self.model_dict_m with correct data
                            train_ds_h = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                                       self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                            train_dl_h = DataLoader(train_ds_h, batch_size=self.params['batch_size'], shuffle=True)

                            model_m = self.model_dict_m[self.name_of_models[i]]
                            criterion_m = self.criterion_dict_m[self.name_of_criterions[i]]
                            optimizer_m = self.optimizer_dict_m[self.name_of_optimizers[i]]

                            # self.train_local_model(model_m, train_dl_h, criterion_m, optimizer_m)
                        elif rl_mode == "untargeted_label_flip":
                            # malicious client uses malicious dataset
                            train_ds = TensorDataset(self.x_train_m_dict["x_train_m_" + str(i)].to(self.cuda_device),
                                                     self.y_train_m_dict["y_train_m_" + str(i)].to(
                                                         self.cuda_device))  # malicious dataset
                            train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)
                    """
                    train_ds = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                             self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                    train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)

                    if i in malicious_names:
                        if rl_mode == "untargeted_label_flip" or self.attack_method == 'explicit boosting':
                            # print('explicit boosting in start_train_end_node_process')
                            train_ds = TensorDataset(self.x_train_m_dict["x_train_m_" + str(i)].to(self.cuda_device),
                                                     self.y_train_m_dict["y_train_m_" + str(i)].to(
                                                         self.cuda_device))  # malicious dataset
                            train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True )

                    model = self.model_dict[self.name_of_models[i]]
                    criterion = self.criterion_dict[self.name_of_criterions[i]]
                    optimizer = self.optimizer_dict[self.name_of_optimizers[i]]

                    # the local training process
                    train_loss, train_accuracy = self.train_local_model(model, train_dl, criterion, optimizer)
                    # if i % 10 == 0:
                    #     print("Debugging", i, ":", train_loss, train_accuracy)
                # malicious_attack is False
                else:
                    train_ds = TensorDataset(self.x_train_dict[self.name_of_x_train_sets[i]].to(self.cuda_device),
                                             self.y_train_dict[self.name_of_y_train_sets[i]].to(self.cuda_device))
                    train_dl = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True )

                    model = self.model_dict[self.name_of_models[i]]
                    criterion = self.criterion_dict[self.name_of_criterions[i]]
                    optimizer = self.optimizer_dict[self.name_of_optimizers[i]]

                    train_loss, train_accuracy = self.train_local_model(model, train_dl, criterion, optimizer)
                    # test_loss, test_accuracy = validation(model, test_dl, criterion)
                    # print("end node train_loss, train_accuracy", train_loss, train_accuracy)

            if plot_indiv is True:
                for sample in sample_list:
                    self.plot_node_weights(self.model_dict, sample[0], sample[1], iteration)

            if plot_glob is True:
                iteration_result = self.create_similarity_matrix(self.model_dict, layer, distance_matrix)
                self.global_results.append([iteration, epoch, layer, distance_matrix, iteration_result])

        # After all epochs in one FL iteration, the weights are in the model dictionary, we need to get the gradients
        # subset_dict = {key: value for key, value in self.model_dict() if key in idx_clients}
        model_dict_after = get_weights_from_model_dict(self.subset_dict)
        # model_dict_after_m = get_weights_from_model_dict(self.model_dict_m)

        # logger.info(f"Debugging check weights, {model_dict_after[self.name_of_models[0]].keys()}")
        # if len(self.malicious_names) != 0:
        #     selected_mali_model = "model" + str(self.malicious_names[-1])

            # logger.info(f"Debugging check weights, {model_dict_after[self.name_of_models[0]].keys()}")
            # logger.info(f"Debugging check malicious client weights {selected_mali_model}, "
            #             f"{get_layer_weight(model_dict_after[selected_mali_model], self.params['weights_layer'])}")
 

        honest_model_names = ["model" + str(x) for x in list(range(self.number_of_clients)) if
                              x not in self.malicious_names]
        selected_honest_model = honest_model_names[-1]

        # logger.info(f"Debugging check honest client weights {selected_honest_model}, "
        #             f"{get_layer_weight(model_dict_after[selected_honest_model], self.params['weights_layer'])}")

        # after weights - before weights = gradients
        grads_regular = self.get_gradients_bet_models(self.model_dict_before, model_dict_after)
        # grads_mali_honest = self.get_gradients_bet_weights(model_dict_before_m, model_dict_after_m,
        #                                                    self.model_dict_m)

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
                                             lr=self.params['learning_rate'], momentum=self.params['momentum'])
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

            # logger.info(f"buckets_group: {buckets_group}")

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

    def get_aggregated_weights(self, model_dict, aggregation_method):
        name_of_models = list(model_dict.keys())
        number_of_clients = len(model_dict)
        weight_accumulator = {}
        malicious_weight_sel = None

        # print("name_of_models[0]", name_of_models[0])
        for name, param in model_dict[name_of_models[0]].state_dict().items():
            struct = torch.zeros(size=param.size(), device=self.cuda_device)
            weight_accumulator[name] = struct

        # clients' model integrity check
        # if self.params['check_integrity']:
        #     engineered_model = self.check_model_integrity(model_dict)

        # FedAvg
        if aggregation_method == 'fedavg':
            # print('agg method is fedavg')
            with torch.no_grad():
                for i in range(number_of_clients):
                    for key, value in weight_accumulator.items():
                        weight_accumulator[key] += get_layer_weight(model_dict[name_of_models[i]],
                                                                    key).data.clone()

                for key, value in weight_accumulator.items():
                    weight_accumulator[key] = value / number_of_clients

        elif aggregation_method == 'geom_median':
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
                # print("model_sample_weight_dict.keys()", model_sample_weight_dict.keys())
                geo_med, client_weights = geometric_median(model_sample_weight_dict, number_of_clients,
                                                           method='weiszfeld', options={})
                # client_weights save to csv, simplified version to logger
                malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
                ordered_client_w_values = self.ordered_client_weight_values(client_weights)
                self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

                logger_line = []

                for i in self.malicious_sample:
                    key = "model" + str(i) + "_M"
                    pair = "{}: {:.1f}".format(key, client_weights["model" + str(i)])
                    logger_line.append(pair)

                for i in self.honest_sample:
                    key = "model" + str(i)
                    pair = "{}: {:.1f}".format(key, client_weights[key])
                    logger_line.append(pair)

                logger.info(f"client_weights in aggr: {logger_line}")
            weight_accumulator = self.weighted_average(model_dict, client_weights)
        elif aggregation_method == 'single-krum':
            weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=1)
        elif aggregation_method == 'multi-krum':
            weight_accumulator = self.krum_get_weight_accumulator(model_dict, krum_m=self.params['krum_m'])
        elif aggregation_method == 'foolsgold':
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)

                for name, weight in model_sample_weight_dict.items():
                    if name in self.memory_dict.keys():
                        self.memory_dict[name].append(weight)
                        if len(self.memory_dict[name]) == self.history_len + 1:
                            # keep the history length not increasing
                            self.memory_dict[name].pop(0)
                    else:
                        self.memory_dict[name] = [weight]

                # print("len of model1", len(self.memory_dict["model1"]))

                historical_weights = {}
                for name, weights in self.memory_dict.items():
                    for w in weights:
                        if name in historical_weights.keys():
                            historical_weights[name] += w
                        else:
                            historical_weights[name] = w

                client_weights = foolsgold(historical_weights)

                # debugging with different aggregation rules
                # client_weights = foolsgold(model_sample_weight_dict)
                # client_weights = pendulum(model_sample_weight_dict, self.count)

                # self.count += 1
                malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
                ordered_client_w_values = self.ordered_client_weight_values(client_weights)
                self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

                logger_line1 = []
                logger_line2 = []
                # logger unzero weights
                for name, weight in client_weights.items():
                    if weight != 0:
                        if int(name[5:]) in self.malicious_sample:
                            key = name + "_M"
                            pair = "{}: {:.4f}".format(key, client_weights[name])
                            logger_line1.append(pair)
                        else:
                            pair = "{}: {:.4f}".format(name, client_weights[name])
                            logger_line1.append(pair)
                    else:
                        if int(name[5:]) in self.malicious_sample:
                            key = name + "_M"
                            pair = "{}: {:.4f}".format(key, client_weights[name])
                            logger_line2.append(pair)
                        else:
                            pair = "{}: {:.4f}".format(name, client_weights[name])
                            logger_line2.append(pair)

                logger.info(f"foolsgold non zero weights: {logger_line1}")
                logger.info(f"foolsgold zero weights: {logger_line2}")

            weight_accumulator = self.weighted_average(model_dict, client_weights)
        elif aggregation_method == 'contra':
            logger.info(f"reputation_dict: {self.reputation_dict}")

            # clients selection based on reputation
            total = sum(self.reputation_dict.values(), 0.0)
            if total == 0:
                total = 1e-7
            self.reputation_dict = {k: v / total for k, v in self.reputation_dict.items()}
            sorted_keys = sorted(self.reputation_dict, key=self.reputation_dict.get, reverse=True)

            for w in sorted_keys:
                self.contra_sorted_dict[w] = (self.reputation_dict[w] * self.contra_f * (
                        1 - self.contra_f)) + self.contra_f
            key_list = list(self.contra_sorted_dict.keys())
            value_list = list(self.contra_sorted_dict.values())
            # agent_name_keys = random.choices(key_list, weights=value_list, k=self.params["contra_clients_selection"])
            # in case value_list has negative values
            if min(value_list) < 0:
                value_list = [x - min(value_list) for x in value_list]
            agent_name_keys = np.random.choice(key_list, self.params["contra_clients_selection"],
                                               p=value_list / np.sum(value_list), replace=False)
            # agent_name_keys = list(set(agent_name_keys))
            # print("len of agent_name_keys", len(agent_name_keys))

            logger_line = []
            for name in agent_name_keys:
                if int(name[5:]) in self.malicious_names:
                    logger_line.append(name + "_M")
                else:
                    logger_line.append(name)
            logger.info(f"contra selected clients: {logger_line}")

            # Keep the history
            model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
            with torch.no_grad():
                for name, weight in model_sample_weight_dict.items():
                    if name in self.memory_dict.keys():
                        self.memory_dict[name].append(weight)
                        if len(self.memory_dict[name]) == self.history_len + 1:
                            # keep the history length not increasing
                            self.memory_dict[name].pop(0)
                    else:
                        self.memory_dict[name] = [weight]

                historical_weights = {}
                for name, weights in self.memory_dict.items():
                    for w in weights:
                        if name in historical_weights.keys():
                            historical_weights[name] += w
                        else:
                            historical_weights[name] = w

            # use foolsgold
            selected_model_dict = dict()

            # print("self.model_dict keys", self.model_dict.keys())
            # print("agent_name_keys", agent_name_keys)

            with torch.no_grad():
                for name, weight in self.model_dict.items():
                    if name in agent_name_keys:
                        selected_model_dict[name] = weight
                # print("len of selected_model_dict", len(selected_model_dict))
                # print("selected model dict keys", selected_model_dict.keys())

                contra_sample_weight_dict = self.sampling_weight_layer(selected_model_dict, self.sampling_layer)

            # update the reputation dict
            client_weights, self.reputation_dict = foolsgold_new(contra_sample_weight_dict, self.reputation_dict)

            # for not selected clients, their weights are 0
            for name in model_dict.keys():
                if name not in client_weights.keys():
                    client_weights[name] = 0

            malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
            ordered_client_w_values = self.ordered_client_weight_values(client_weights)
            self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

            # normalize the reputation
            maxr = max(self.reputation_dict.values())
            for k in self.reputation_dict:
                self.reputation_dict[k] = self.reputation_dict[k] / maxr

            weight_accumulator = self.weighted_average(model_dict, client_weights)
        elif aggregation_method == 'coordinate_wise_median':
            weight_accumulator = coordinate_wise_median(model_dict)
        elif aggregation_method == 'trimmed_mean':
            weight_accumulator = trimmed_mean(model_dict, self.params['trimmed_mean_m'])
        elif aggregation_method == 'bulyan':
            # Use the clients selected from multi-krum
            with torch.no_grad():
                model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
                muti_krum_m, krum_clients, client_weights = krum(model_sample_weight_dict, self.params['krum_f'],
                                                                 number_of_clients - 2 * self.params['bulyan_m'])
                krum_selected_model_dict = {key: model_dict[key] for key in krum_clients}

                # trimmed_mean
                weight_accumulator = trimmed_mean(krum_selected_model_dict, 2 * self.params['bulyan_m'])
        else:
            warnings.warn("unknown aggregation method")

        if malicious_weight_sel is not None:
            logger.info(f"malicious weight selected %: {malicious_weight_sel * 100 :.2f}")

        return weight_accumulator

    def krum_get_weight_accumulator(self, model_dict, krum_m):
        with torch.no_grad():
            model_sample_weight_dict = self.sampling_weight_layer(self.model_dict, self.sampling_layer)
            muti_krum_m, krum_clients, client_weights = krum(model_sample_weight_dict, self.params['krum_f'],
                                                             krum_m)

            malicious_weight_sel = self.malicious_weight_percentage(client_weights, self.malicious_names)
            ordered_client_w_values = self.ordered_client_weight_values(client_weights)
            self.weights_csv_logger.append_csv(ordered_client_w_values + [malicious_weight_sel])

            krum_clients = list(map(lambda i: int(i[5:]), krum_clients))
            logger_line = []

            logger_clients = unique(self.malicious_sample + self.honest_sample + krum_clients)
            for i in logger_clients:
                if i in self.malicious_names:
                    key = "model" + str(i) + "_M"
                    pair = "{}: {:.1f}".format(key, client_weights["model" + str(i)])
                    logger_line.append(pair)
                else:
                    key = "model" + str(i)
                    pair = "{}: {:.1f}".format(key, client_weights[key])
                    logger_line.append(pair)

            mistakes = [i for i in krum_clients if i in self.malicious_names]
            # for name, score in muti_krum_m:
            #     if int(name[5:]) in mistakes:
            #         name = name + "_M"
            #     pair = "{}: {:.4f}".format(name, score)
            #     logger_line.append(pair)

            if len(mistakes) > 0:
                logger.info(f"mistakenly chosen: {mistakes}")
            logger.info(f"krum score: {logger_line}")
        weight_accumulator = self.weighted_average(model_dict, client_weights)
        return weight_accumulator

    def malicious_weight_percentage(self, client_weights, malicious_names):
        # print("client_weights", client_weights)
        # print("malicious_names", malicious_names)
        malicious_weight = 0
        for key, value in client_weights.items():
            if int(key[5:]) in malicious_names:
                # print("key", key, value)
                malicious_weight += value
        total_weights = sum(client_weights.values())

        # print("malicious_weight", malicious_weight)
        # print("total_weights", total_weights)
        # print("mali percentage", malicious_weight/total_weights)
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

            # to add cos similiary comparsion

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

        # print("clients_weights.keys()", clients_weights.keys())
        # for w, p in zip(clients_weights, model_dict):
        #     for name, data in weighted_updates.items():
        #         # print("w", w)
        #         temp = (clients_weights[w] / tot_weights).to(self.device)
        #         print ("p", p)
        #         temp = temp * torch.tensor(p[name]).to(self.device)
        #
        #         data = data.to(self.device)
        #
        #         if temp.dtype!=data.dtype:
        #             temp = temp.type_as(data)
        #         data.add_(temp)
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

        with torch.no_grad():
            if self.params['server_lr_fix'] is False:
                if iteration == 0:
                    for name, param in main_model.state_dict().items():
                        # directly use weight_accumulator as server's weight
                        get_layer_weight(main_model, name).data = weight_accumulator[name].data.clone()
                    # use weight_accumlator * learning rate
                else:
                    logger.info(f"server lr: {self.server_lr}")
                    for name, param in main_model.state_dict().items():
                        # 3 way to learn from the new weights and keep the old part of the old weights

                        # 1. Add the new weight * lr to the existed weight
                        # get_layer_weight(main_model, name).data.add_(weight_accumulator[name].data.clone() * lr)

                        # 2. Use the new weight as weight, give up the existed weight
                        # get_layer_weight(main_model, name).data = weight_accumulator[name].data.clone()

                        # 3. Distribute between the existed weight and the new weights using lr, together is 1
                        get_layer_weight(main_model, name).data = (1 - lr) * get_layer_weight(main_model,
                                                                                              name).data.clone() + \
                                                                  lr * weight_accumulator[name].data.clone()

                    self.server_lr *= (
                            1. / (1. + self.server_lr_decay * iteration))  # if iteration is in decay schedule
                    if self.server_lr < self.server_lr_floor:
                        self.server_lr = self.server_lr_floor
                    # self.server_lr *= (1. / (1. + self.server_lr_decay ))
            else:
                # fixed learning rate without a changing schedule
                logger.info(f"server lr: {self.server_lr}")
                for name, param in main_model.state_dict().items():
                    get_layer_weight(main_model, name).data = (1 - self.server_lr) * get_layer_weight(
                        main_model, name).data.clone() + \
                                                              self.server_lr * weight_accumulator[
                                                                  name].data.clone()

        return main_model

    def send_main_model_to_nodes_and_update_model_dict(self, main_model, model_dict):
        name_of_models = list(model_dict.keys())
        with torch.no_grad():
            for i in range(len(model_dict)):
                # for name, param in main_model.named_parameters():
                #     layer = name.split('.')[0]
                #     seg = name.split('.')[-1]
                #     getattr(getattr(model_dict[self.name_of_models[i]], layer), seg).data = \
                #         getattr(getattr(main_model, layer), seg).data.clone()
                for name, param in main_model.state_dict().items():
                    # layer = name.split('.')[0]
                    # seg = name.split('.')[-1]
                    # getattr(getattr(model_dict[self.name_of_models[i]], layer), seg).data = \
                    #     getattr(getattr(main_model, layer), seg).data.clone()
                    # if if_reverse is False:

                    get_layer_weight(model_dict[name_of_models[i]], name).data = \
                        get_layer_weight(main_model, name).data.clone()

                    # else:
                    #     # print("type before torch.neg", type(get_layer_weight(main_model, name)))
                    #     # to fix, gradient to reverse, not weight
                    #     get_layer_weight(model_dict[name_of_models[i]], name).data = \
                    #         torch.neg(self.params['reverse_grad_epsilon']*
                    #                   get_layer_weight(main_model, name).data).clone()

                    # indices = torch.tensor([0, 1]).to(device)
                    # print("Org tensor", torch.index_select(get_layer_weight(main_model, name).data, 0, indices))
                    # print("Negative tensor",
                    #       torch.index_select(get_layer_weight(model_dict[name_of_models[i]], name).data, 0, indices))

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
    """
    weights_dict = {}
    for key, value in model_dict.items():
        # print("get_layer_weight(value, layer_name)", get_layer_weight(value, layer_name))
        weights = {}
        for layer_name in self.main_model.state_dict().keys():
            weight = get_layer_weight(value, layer_name).detach().clone()
            weights[layer_name] = weight
        weights_dict[key] = weights
    return weights_dict
    """
    model_dict_copy = copy.deepcopy(model_dict)
    return model_dict_copy

def client_random_selection(model_dict, num_clients, m):
    idxs_clients = np.random.choice(range(num_clients), m, replace=False)

    model_names = []
    for i in idxs_clients:
        model_names.append("model" + str(i))
    
    selected_model_dict = {k: model_dict[k] for k in model_names}
    return selected_model_dict, idxs_clients

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
