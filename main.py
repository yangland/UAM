from helper import Helper
import os
import yaml
import datetime
import argparse
import logging
import shutil
from visualization import plotting_global, plotting_training_curves,\
     images_to_video, plot_node_w_order, plot_node_w_order_1d
from csv_logging import Csv_logging


def exp_w_plotting(current_time, series_time, params_loaded, cuda_device):
    logger = logging.getLogger("logger")
    dataset_type = params_loaded['type']

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
        print("Data type not configured in the main.py ")
    logger.info(f'Running Parameters:')

    for key, value in helper.params.items():
        logger.info(f"{key}:  {helper.params[key]}")
    logger.info(f'\nhelper done')

    # Initialization
    helper.fl_initialization()
    logger.info(f'initialization done')

    # Loading data
    helper.create_honest_malicious_data()

    # Training
    global_results, test_accuracy_records, test_loss_records, attack_success_rate_records = helper.train_iterations()
    logger.info(f'training done')


    # UAM's dominate strategy
    dominate_strategy = None
    if params_loaded['attack_method'] == 'UAM':
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

    # Converting global_results plots to video
    if helper.params['create_video']:
        image_path = os.path.join(helper.folder_path, 'global_results')
        video_name = helper.params['weights_layer'] + '_' + helper.params['distance_matrix']
        images_to_video(image_path, helper.folder_path, 6, ".png", video_name, ".mp4")
        logger.info(f'video output done')

    return test_accuracy_records[-1], test_loss_records[-1], attack_success_rate_records[-1], helper, dominate_strategy


def run_exp_once(params_loaded, series_time, curr_master, i, total_num_exp, dataset_type, cuda_device):
    params_loaded['if_noniid'] = curr_master['if_noniid']
    params_loaded['dirichlet_alpha'] = curr_master['dirichlet_alpha']
    params_loaded['aggregation_method'] = curr_master['aggregation_method']
    params_loaded['number_of_malicious'] = curr_master['number_of_malicious']
    params_loaded['pre_aggregation_method'] = curr_master['pre_aggregation_method']
    params_loaded['attack_method'] = curr_master['attack_method']
    if params_loaded['number_of_malicious'] == 0:
        params_loaded['malicious_attack'] = False
    else:
        params_loaded['malicious_attack'] = True

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    print(i+1, "th experiment of", total_num_exp)

    test_accuracy, test_loss, attack_success_rate, helper, dominate_strategy = \
        exp_w_plotting(current_time, series_time, params_loaded, cuda_device)

    del params_loaded['current_time']

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


def run_exp_series(params_loaded, series_time, dataset_type, cuda_device):
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
    number_of_malicious_s = params_loaded['number_of_malicious_s']
    pre_aggr_method_s = params_loaded['pre_aggr_method_s']


    del params_loaded['if_noniid_s']
    del params_loaded['dirichlet_alpha_s']
    del params_loaded['aggregation_method_s']
    del params_loaded['pre_aggr_method_s']
    del params_loaded['number_of_malicious_s']

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
    print("master_loop", *master_loop, sep='\n')

    start_pos = 0
    end_pos = len(master_loop)
    for i in range(start_pos, end_pos):
        run_exp_once(params_loaded, series_time, master_loop[i], i, total_num_exp, dataset_type, cuda_device)



def main():
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    series_csv_logger = Csv_logging('series', [], 1, [])

    with open(f'./{args.params}', 'r') as f:
        # dictionary
        params_loaded = yaml.safe_load(f)
    curr_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

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

        run_exp_series(params_loaded, curr_time, params_loaded['type'], cuda_device)

    else:
        # one experiment
        exp_w_plotting(curr_time, curr_time, params_loaded, cuda_device)
        # run_exp_once(params_loaded, curr_time, params_loaded, 1, 1, dataset_name, cuda_device)
        

if __name__ == '__main__':
    main()
