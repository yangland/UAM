import csv
import os
from csv import writer

class Csv_logging:
    def __init__(self, name, class_names, number_of_clients, malicious_names):
        self.name = name
        # god validation accuracy and loss
        self.god_valid_al_fileHeader = ["iteration", "accuracy", "loss", "action1", "action2"]
        self.god_valid_al_results = []

        # god validation per-class and Attack Success Rate (ASR)
        self.god_valid_pca_fileHeader = ["iteration"] + class_names + ['attack_success_rate']
        self.god_valid_pca_results = []

        # malicious validations
        self.mali_valid_al_fileHeader = ["iteration", "accuracy", "loss"]
        self.mali_valid_al_results = []
        self.mali_valid_pca_fileHeader = ["iteration"] + class_names + ['attack_success_rate']
        self.mali_valid_pca_results = []

        # Fagg's clients weights and malicious client's weights per iteration
        self.client_weights = []
        self.weights_fileHeader = []

        for name in range(number_of_clients):
            model_name = "model" + str(name)
            if name in malicious_names:
                self.weights_fileHeader.append(model_name + "_(M)")
            else:
                self.weights_fileHeader.append(model_name)
        self.weights_fileHeader.append("malicious_weight_selected")

        self.series_master = []

        self.series_master_fileHeader = [
            'current_time', 'if_noniid', 'dirichlet_alpha', 'aggregation_method',
            'pre_aggregation_method', 'malicious_attack',
            'number_of_malicious', 'malicious_proportion', 'attack_method',
            'malicious_group_agg', 'test_accuracy', 'test_loss', 'dominate_strategy',
            'attack_success_rate'
        ]




    def append_csv(self, record):
        if self.name == 'god_valid_al':
            self.god_valid_al_results.append(record)
        elif self.name == 'god_valid_pca':
            self.god_valid_pca_results.append(record)
        elif self.name == 'mali_valid_al':
            self.mali_valid_al_results.append(record)
        elif self.name == 'mali_valid_pca':
            self.mali_valid_pca_results.append(record)
        elif self.name == 'Fagg_weights':
            self.client_weights.append(record)


    def save_csv(self, folder_path, current_time):
        if self.name == 'god_valid_al':
            god_valid_al_csvFile = open(f'{folder_path}/god_valid_al_{current_time}.csv', "w+")
            god_valid_al_writer = csv.writer(god_valid_al_csvFile, lineterminator='\n')
            god_valid_al_writer.writerow(self.god_valid_al_fileHeader)
            god_valid_al_writer.writerows(self.god_valid_al_results)
            god_valid_al_csvFile.close()

        elif self.name == 'god_valid_pca':
            god_valid_pca_csvFile = open(f'{folder_path}/god_valid_pca_{current_time}.csv', "w+")
            god_valid_pca_writer = csv.writer(god_valid_pca_csvFile, lineterminator='\n')
            god_valid_pca_writer.writerow(self.god_valid_pca_fileHeader)
            god_valid_pca_writer.writerows(self.god_valid_pca_results)
            god_valid_pca_csvFile.close()

        elif self.name == 'mali_valid_al':
            mali_valid_al_csvFile = open(f'{folder_path}/mali_valid_al_{current_time}.csv', "w+")
            mali_valid_al_writer = csv.writer(mali_valid_al_csvFile, lineterminator='\n')
            mali_valid_al_writer.writerow(self.mali_valid_al_fileHeader)
            mali_valid_al_writer.writerows(self.mali_valid_al_results)
            mali_valid_al_csvFile.close()

        elif self.name == 'mali_valid_pca':
            mali_valid_pca_csvFile = open(f'{folder_path}/mali_valid_pca_{current_time}.csv', "w+")
            mali_valid_pca_writer = csv.writer(mali_valid_pca_csvFile, lineterminator='\n')
            mali_valid_pca_writer.writerow(self.mali_valid_pca_fileHeader)
            mali_valid_pca_writer.writerows(self.mali_valid_pca_results)
            mali_valid_pca_csvFile.close()

        elif self.name == 'Fagg_weights':
            weights_csvFile = open(f'{folder_path}/weights_result_{current_time}.csv', "w+")
            weights_writer = csv.writer(weights_csvFile, lineterminator='\n')
            weights_writer.writerow(self.weights_fileHeader)
            weights_writer.writerows(self.client_weights)
            weights_csvFile.close()

    # def append_save_csv(self, record, folder_path, current_time, dataset_type):
    #     if self.name == 'series':
    #         path = f'{folder_path}/series_master_{dataset_type}_{current_time}.csv'

    #         if os.path.exists(path):
    #             # print("os path exists")
    #             with open(path, 'a', newline='\n') as f_object:
    #                 # print("writing here")
    #                 writer_object = writer(f_object)
    #                 writer_object.writerow(record)
    #                 f_object.close()
    #         else:
    #             series_csvFile = open(path, "w+")
    #             series_writer = csv.writer(series_csvFile, lineterminator='\n')
    #             series_writer.writerow(self.series_master_fileHeader)
    #             series_writer.writerow(record)
    #         # series_csvFile.close()
    
    def append_save_csv(self, record, folder_path, current_time, dataset_type):
        if self.name == 'series':
            path = f'{folder_path}/series_master_{dataset_type}_{current_time}.csv'
            exists = os.path.exists(path)
            with open(path, 'a', newline='\n') as f:
                w = csv.writer(f, lineterminator='\n')
                if not exists:
                    w.writerow(self.series_master_fileHeader)
                w.writerow(record)
