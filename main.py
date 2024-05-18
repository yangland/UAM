from helper import Helper
from helper import *
import os
import yaml
import time
import datetime
import random
import os
import argparse
import logging
import shutil
import numpy as np
from visualization import plotting_global, plotting_training_curves,\
     images_to_video, plot_node_w_order, plot_node_w_order_1d
from csv_logging import Csv_logging
import os
# import torch


def experiment_params(current_time, series_time, params_loaded, cuda_device):
    logger = logging.getLogger("logger")
    dataset_type = params_loaded['type']

    # cuda_device, _ = get_cuda_w_most_memory()

    if dataset_type == 'mnist':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'), cuda_device=cuda_device)
    elif dataset_type == 'cifar10':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'cifar10'), cuda_device=cuda_device)
    elif dataset_type == 'emnist':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'emnist'), cuda_device=cuda_device)
    elif dataset_type == 'fashion_mnist':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'fashion_mnist'), cuda_device=cuda_device)
    elif dataset_type == 'kmnist':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'kmnist'), cuda_device=cuda_device)
    elif dataset_type == 'cifar100':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'cifar100'), cuda_device=cuda_device)
    elif dataset_type == 'imagenette':
        helper = Helper(current_time=current_time, series_time=series_time, params=params_loaded,
                        name=params_loaded.get('name', 'imagenette'), cuda_device=cuda_device)    
    else:
        helper = None
        print("type not configured in the main.py ")
    logger.info(f'Running Parameters:')

    for key, value in helper.params.items():
        logger.info(f"{key}:  {helper.params[key]}")
    logger.info(f'\nhelper done')

    # Initialization
    helper.fl_initialization()
    logger.info(f'initialization done')

    # Loading data
    helper.create_honest_malicious_data()

    # print("helper.params['malicious_attack']", helper.params['malicious_attack'])

    if helper.attack_method != 'RL'  and helper.params['malicious_attack'] is True:
        logger.info(f'No RL attack(baseline), create honest malicious data done')
    elif helper.attack_method == 'RL' and helper.params['malicious_attack'] is True:
        # use RL to create each malicious clients' malicious training data
        # 2. Naive MCTS with arm expansion (input: malicious view accuracy, attack success rate;
        #                                                                         return: attacking parameters)

        # 3. Group attack based on the attacking parameters (input: attacking parameters;
        #                                                           return: malicious datasets for each client)
        # 4. Visualization on the objects of a group of clients
        # * malicious training
        logger.info(f'RL based attacking, create honest malicious data done')
    else:
        logger.info(f'There is No attacker')

    # Training
    global_results, test_accuracy_records, test_loss_records, attack_success_rate_records = helper.train_iterations()
    logger.info(f'training done')

    # print("global_results", global_results)
    # print("test_accuracy_records", test_accuracy_records)
    # print("test_loss_records", test_loss_records)

    # RL's dominate strategy
    dominate_strategy = None
    if params_loaded['attack_method'] == 'RL':
        dominate_strategy = helper.neldermead.get_dominate_strategy()
        print("dominate_strategy", dominate_strategy)

    # Plotting
    if helper.params['plot_global']:
        plotting_global(helper.folder_path, helper.params['number_of_clients'], helper.params['malicious_attack'],
                        helper.params['malicious_proportion'], global_results)
    if helper.params['plot_train']:
        plotting_training_curves(helper.folder_path, helper.params['weights_layer'],
                                 helper.params['distance_matrix'], test_accuracy_records, test_loss_records)

    if helper.params['plot_rl'] and helper.params["number_of_malicious"] != 0:
        if helper.neldermead is not None:
            if helper.rl_mode == "untargeted_label_flip":
                plot_node_w_order(helper.folder_path, helper.neldermead.history)
            elif helper.rl_mode == "reverse_grad":
                if helper.search_dimension == 1:
                    # to do creat a 1D plot for zoom factor's moving
                    plot_node_w_order_1d(helper.folder_path, helper.neldermead.history)
                elif helper.search_dimension == 2:
                    plot_node_w_order_1d(helper.folder_path, helper.neldermead.history)
    logger.info(f'plotting done')

    # debugging Plotting
    # test_accuracy_records = [1.23, 2.43, 4.00, 5.68, 5]
    # test_loss_records = [4, 3.678, 2.4, 1.01, 4.68]
    # plotting_training_curves(helper.folder_path, helper.params['weights_layer'],
    #                           helper.params['distance_matrix'], test_accuracy_records, test_loss_records)

    # Converting global_results plots to video
    if helper.params['create_video']:
        image_path = os.path.join(helper.folder_path, 'global_results')
        video_name = helper.params['weights_layer'] + '_' + helper.params['distance_matrix']
        images_to_video(image_path, helper.folder_path, 6, ".png", video_name, ".mp4")
        logger.info(f'video output done')

    # logger.disabled()
    return test_accuracy_records[-1], test_loss_records[-1], attack_success_rate_records[-1], helper, dominate_strategy


def handle_rlba_false(params_loaded, series_time, dataset_type, cuda_device):
    """
    handle case: rs_based_attack is false
    """
    # loop of parameters
    # series_csv_logger = Csv_logging('series', [], 1, [])
    print("Non-RL based series experiments")

    if_noniid_s = params_loaded['if_noniid_s']
    dirichlet_alpha_s = params_loaded['dirichlet_alpha_s']

    # malicious_mode_s = params_loaded['malicious_mode_s']

    targeted_pair_s = params_loaded['targeted_pair_s']
    weights_malicious_group_s = params_loaded['weights_malicious_group_s']
    respect_server_model_s = params_loaded['respect_server_model_s']
    malicious_group_agg_s = params_loaded['malicious_group_agg_s']

    non_iidness = []

    for if_noniid in if_noniid_s:
        if if_noniid == True:
            for dirichlet_alpha in dirichlet_alpha_s:
                non_iidness.append([if_noniid, dirichlet_alpha])
        else:
            non_iidness.append([if_noniid, -1])

    aggregation_method_s = params_loaded['aggregation_method_s']
    # malicious_attack_s = params_loaded['malicious_attack_s']
    number_of_malicious_s = params_loaded['number_of_malicious_s']
    pre_aggr_method_s = params_loaded['pre_aggr_method_s']

    malicious_config1 = []
    # one malicious and no attack
    # for malicious_attack in malicious_attack_s:
    #     if malicious_attack == False:
    #         # if_malicious_attack, number_of_malic, malicious_proportion
    #         # no attack
    #         malicious_config1.append([malicious_attack, 0, [['untargeted1', 1, -1, -1]]])
    #     else:
    #         for number_of_malicious in number_of_malicious_s:
    #             for malicious_mode in malicious_mode_s:
    #                 if malicious_mode not in ['untargeted1', 'untargeted2']:
    #                     malicious_proportion = [malicious_mode, 1] + targeted_pair_s[0]
    #                 else:
    #                     malicious_proportion = [malicious_mode, 1, -1, -1]
    #                 malicious_config1.append([malicious_attack, number_of_malicious, [malicious_proportion]])

    # for number_of_malicious in number_of_malicious_s:
    #     if number_of_malicious != 0:
    #         for malicious_mode in malicious_mode_s:
    #             if malicious_mode not in ['untargeted1', 'untargeted2']:
    #                 malicious_proportion = [malicious_mode, 1] + targeted_pair_s[0]
    #             else:
    #                 malicious_proportion = [malicious_mode, 1, -1, -1]
    #             malicious_config1.append([True, number_of_malicious, [malicious_proportion]])
    #     else:
    #         malicious_config1.append([False, 0, [['untargeted1', 1, -1, -1]]])


    malicious_config = malicious_config1 #+ malicious_config2

    malicious_config3 = []
    for if_respect in respect_server_model_s:
        if if_respect:
            malicious_config3.append([if_respect, 0])
        else:
            for if_agg in malicious_group_agg_s:
                malicious_config3.append([if_respect, if_agg])

    # print("non_iidness", non_iidness)
    # print("malicious_config", malicious_config)
    # print(*malicious_config, sep="\n")
    # print("malicious_config3", malicious_config3)

    # remove all the series dict items as all parameters are in lists
    del params_loaded['if_noniid_s']
    del params_loaded['dirichlet_alpha_s']
    # del params_loaded['malicious_mode_s']
    del params_loaded['targeted_pair_s']
    del params_loaded['weights_malicious_group_s']
    del params_loaded['respect_server_model_s']
    del params_loaded['malicious_group_agg_s']
    del params_loaded['aggregation_method_s']
    del params_loaded['pre_aggr_method_s']
    # del params_loaded['malicious_attack_s']
    del params_loaded['number_of_malicious_s']


    # count = 0
    # print("lens", len(non_iidness), len(aggregation_method_s), len(malicious_config), len(malicious_config3))
    # 3 5 57 3
    total_num_exp = len(non_iidness) * len(aggregation_method_s) * len(malicious_config) * len(malicious_config3)

    # reverse 1 list as some did not run compelete
    # non_iidness.reverse()

    # convert the main loop into 2 parts: 1. list generation 2. iteration (with control of starting i)
    master_loop = []
    for non_i in non_iidness:
        for aggregation_method in aggregation_method_s:
            for mc in malicious_config:
                for mc3 in malicious_config3:
                    params = dict()
                    params['if_noniid'] = non_i[0]
                    params['dirichlet_alpha'] = non_i[1]

                    params['aggregation_method'] = aggregation_method

                    params['malicious_attack'] = mc[0]
                    params['number_of_malicious'] = mc[1]
                    params['malicious_proportion'] = mc[2]

                    params['respect_server_model'] = mc3[0]
                    params['malicious_group_agg'] = mc3[1]
                    master_loop.append(params)

    print("non-rl master_loop", *master_loop, sep='\n')
    # looping over the master_loop, the starting number can change to desired positon
    start_pos = 0
    end_pos = len(master_loop)
    for i in range(start_pos, end_pos):
        params_loaded['if_noniid'] = master_loop[i]['if_noniid']
        params_loaded['dirichlet_alpha'] = master_loop[i]['dirichlet_alpha']
        params_loaded['aggregation_method'] = master_loop[i]['aggregation_method']
        params_loaded['malicious_attack'] = master_loop[i]['malicious_attack']
        params_loaded['number_of_malicious'] = master_loop[i]['number_of_malicious']
        params_loaded['malicious_proportion'] = master_loop[i]['malicious_proportion']
        params_loaded['respect_server_model'] = master_loop[i]['respect_server_model']
        params_loaded['malicious_group_agg'] = master_loop[i]['malicious_group_agg']

        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

        print(i+1, "th experiment of", total_num_exp)
        # print("params_loaded", params_loaded)
        # print("current and series times", current_time, series_time)

        test_accuracy, test_loss, attack_success_rate, helper, dominate_strategy = \
            experiment_params(current_time, series_time, params_loaded, cuda_device)

        del params_loaded['current_time']
        logger.handlers.clear()

        # writing into one master csv file
        line = [current_time, params_loaded['if_noniid'], params_loaded['dirichlet_alpha'], \
                params_loaded['aggregation_method'], 'NA', params_loaded['malicious_attack'], \
                params_loaded['number_of_malicious'], \
                params_loaded['malicious_proportion'], params_loaded['respect_server_model'], \
                params_loaded['malicious_group_agg'], test_accuracy, test_loss, \
                dominate_strategy, attack_success_rate]

        series_csv_logger.append_save_csv(line, helper.master_path, series_time, dataset_type)


def rlba_attack_once(params_loaded, series_time, curr_master, i, total_num_exp, dataset_type, cuda_device):
    params_loaded['if_noniid'] = curr_master['if_noniid']
    params_loaded['dirichlet_alpha'] = curr_master['dirichlet_alpha']
    params_loaded['aggregation_method'] = curr_master['aggregation_method']
    params_loaded['number_of_malicious'] = curr_master['number_of_malicious']
    params_loaded['pre_aggregation_method'] = curr_master['pre_aggregation_method']
    params_loaded['attack_method'] = curr_master['attack_method']
    if params_loaded['number_of_malicious'] == 0:
        params_loaded['malicious_attack'] = False
        # params_loaded['attack_method'] = 'None'
        # params_loaded['respect_server_model'] = True
        # print("params_loaded['number_of_malicious'] == 0")
    else:
        params_loaded['malicious_attack'] = True

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    print(i+1, "th experiment of", total_num_exp)
    # print("params_loaded", params_loaded)
    # print("current and series times", current_time, series_time)

    test_accuracy, test_loss, attack_success_rate, helper, dominate_strategy = \
        experiment_params(current_time, series_time, params_loaded, cuda_device)

    del params_loaded['current_time']
    logger.handlers.clear()

    # writing into one master csv file
    line = [
        current_time,
        params_loaded['if_noniid'],
        params_loaded['dirichlet_alpha'],
        params_loaded['aggregation_method'],
        params_loaded['pre_aggregation_method'],
        params_loaded['malicious_attack'],
        params_loaded['number_of_malicious'],
        params_loaded['malicious_proportion'],
        'NA',
        'NA',
        test_accuracy,
        test_loss,
        dominate_strategy,
        attack_success_rate
    ]
    series_csv_logger.append_save_csv(line, helper.master_path, series_time, dataset_type)


def handle_rlba_true(params_loaded, series_time, dataset_type, cuda_device):
    """
    handle case: rs_based_attack is True
    """
    # RL based series
    # print("RL based series experiments")
    # series_csv_logger = Csv_logging('series', [], 1, [])

    if_noniid_s = params_loaded['if_noniid_s']
    dirichlet_alpha_s = params_loaded['dirichlet_alpha_s']

    non_iidness = []

    for if_noniid in if_noniid_s:
        if if_noniid == True:
            for dirichlet_alpha in dirichlet_alpha_s:
                non_iidness.append([if_noniid, dirichlet_alpha])
        else:
            non_iidness.append([if_noniid, -1])

    aggregation_method_s = params_loaded['aggregation_method_s']
    # malicious_attack_s = params_loaded['malicious_attack_s']
    number_of_malicious_s = params_loaded['number_of_malicious_s']
    pre_aggr_method_s = params_loaded['pre_aggr_method_s']


    del params_loaded['if_noniid_s']
    del params_loaded['dirichlet_alpha_s']
    del params_loaded['aggregation_method_s']
    del params_loaded['pre_aggr_method_s']
    # del params_loaded['malicious_attack_s']
    del params_loaded['number_of_malicious_s']

    # total_num_exp = len(non_iidness) * len(aggregation_method_s) * (len(number_of_malicious_s))

    master_loop = []
    for non_i in non_iidness:
        for aggregation_method in aggregation_method_s:
            for pre_agr_method in pre_aggr_method_s:
                for nm in number_of_malicious_s:
                    params = dict()
                    params['attack_method'] = params_loaded['attack_method']
                    if nm != 0:
                        params['if_noniid'] = non_i[0]
                        params['dirichlet_alpha'] = non_i[1]
                        params['aggregation_method'] = aggregation_method
                        params['number_of_malicious'] = nm
                        params['pre_aggregation_method'] = pre_agr_method
                        master_loop.append(params)
                    else:
                        # zero attacker
                        params['if_noniid'] = non_i[0]
                        params['dirichlet_alpha'] = non_i[1]
                        params['aggregation_method'] = aggregation_method
                        params['number_of_malicious'] = nm
                        params['pre_aggregation_method'] = 'None'
                        master_loop.append(params) if params not in master_loop else master_loop

    total_num_exp = len(master_loop)
    print("rl master_loop", *master_loop, sep='\n')

    start_pos = 0
    end_pos = len(master_loop)
    for i in range(start_pos, end_pos):
        rlba_attack_once(params_loaded, series_time, master_loop[i], i, total_num_exp, dataset_type, cuda_device)

# def get_cuda_w_most_memory():
#     free_memories = []
#     for i in range(torch.cuda.device_count()):
#         info = torch.cuda.mem_get_info(i)
#         free_memories.append(info[0] / 1024 ** 2)
#     # print("free_memories", free_memories)
#     max_value = max(free_memories)
#     max_index = free_memories.index(max_value)
#     return max_index, free_memories

def main():
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        # dictionary
        params_loaded = yaml.safe_load(f)
    curr_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    # print("curr_time", curr_time )

    cuda_device = params_loaded['cuda_device']

    if params_loaded['if_series']:

        # move the parameter files in the same directory
        dataset_name = params_loaded['type']
        master_path = f'./logs/series_{dataset_name}_{curr_time}'
        try:
            os.mkdir(master_path)
        except OSError as error:
            print(error)
        shutil.copy(args.params, master_path)

        if params_loaded['attack_method'] != 'RL':
            # handle_rlba_false(params_loaded, curr_time, params_loaded['type'], cuda_device)
            handle_rlba_true(params_loaded, curr_time, params_loaded['type'], cuda_device)
        else:
            handle_rlba_true(params_loaded, curr_time, params_loaded['type'], cuda_device)

    else:
        # one iteration
        experiment_params(curr_time, curr_time, params_loaded, cuda_device)


if __name__ == '__main__':
    series_csv_logger = Csv_logging('series', [], 1, [])
    main()
