import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text
from datetime import datetime
from datetime import date
# from moviepy.editor import *
import moviepy.video.io.ImageSequenceClip
import math


def plotting_global(folder_path, number_of_clients, malicious_attack, malicious_mode, global_results):
    matrix_master = []
    for i in range(len(global_results)):
        matrix_master.append(global_results[i][-1])

    matrix_master = np.asarray(matrix_master)
    v_min = np.nanmin(matrix_master)
    v_max = np.nanmax(matrix_master)

    global_folder_path = os.path.join(folder_path, 'global_results')

    for i in range(len(global_results)):
        iteration = global_results[i][0]
        epoch = global_results[i][1]
        layer = global_results[i][2]
        mode = global_results[i][3]
        result_matrix = global_results[i][-1]

        plot_matrix(result_matrix, layer, iteration, mode, epoch, v_min, v_max,
                    number_of_clients, malicious_attack, malicious_mode, num_ticks=9, file_location=global_folder_path)


# global matrix
def plot_matrix(result, layer, iteration, distance_matrix, epoch, vmin, vmax, number_of_clients,
                malicious_attack, malicious_mode, num_ticks, file_location):
    matplotlib.use('Agg')
    number_of_clients = number_of_clients
    labels = list(range(0, number_of_clients))

    fig, ax = plt.subplots(figsize=(20, 15))
    # cax = ax.matshow(result, interpolation='nearest')
    if distance_matrix == 'cos':
        colormap = 'OrRd'
    elif distance_matrix == 'l2':
        colormap = 'YlGnBu'
    elif distance_matrix == 'cka':
        colormap = 'BuPu'
    else:
        colormap = 'viridis'

    cax = ax.imshow(result, vmin=vmin, vmax=vmax, cmap=colormap,
                    interpolation='nearest')  # cmap='YlGnBu', 'OrRd'
    # ax.grid(True)
    title_string = 'FL Clients Similarity Matrix IID' + " " + str(distance_matrix) + " " + "iter" + \
                   str(iteration) + "_" + "epoch" + str(epoch) + "_" + "m" + str(malicious_attack)
    if malicious_attack is True:
        title_string = title_string + "_" + str(malicious_mode)

    plt.title(title_string, fontsize=15)
    plt.xticks(range(number_of_clients), labels, rotation=90)
    plt.yticks(range(number_of_clients), labels)
    # ticks = np.linspace(vmin, vmax, num_ticks)
    # print("vmin, vmax", vmin, vmax)
    ticks = np.arange(vmin, vmax, max(round((vmax - vmin) / (num_ticks - 1), 2), 0.01))
    ticks = np.append(ticks, vmax)
    cbar = fig.colorbar(cax, ax=ax)

    font_size = 10  # Adjust as appropriate.
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=font_size)

    # create a time stamp
    today = date.today().strftime('%Y%m%d')
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    plt_name = "Glob" + "_" + layer + "_" + "I" + str(iteration) + "_" + \
               "E" + str(epoch) + "_" + str(today) + "_" + str(current_time) + ".png"

    if not os.path.exists(file_location):
        os.makedirs(file_location)

    plt.savefig(file_location + '/' + plt_name, format='png')
    plt.clf()
    plt.close()


def plotting_training_curves(folder_path, weights_layer, distance_matrix, test_accuracy_records, test_loss_records):
    # replace nan with 0
    test_accuracy_records = [0 if math.isnan(x) else x for x in test_accuracy_records]
    test_loss_records = [0 if math.isnan(x) else x for x in test_loss_records]

    # summary of test accuracy records
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    x = np.array(range(len(test_accuracy_records)))
    y = np.array(test_accuracy_records)
    ax = fig.add_subplot(111)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('iteration')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y, label='Validation')
    plt.legend()
    """
    for i, j in zip(x, y):
        ax.annotate("{:.4f}".format(j), xy=(i, j), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
    """
    # use adjusttext to avoid overlapping on the text annotation on datapoint
    texts = [plt.text(x[i], y[i], '%s' % round(y[i], 3)) for i in range(len(x))]
    adjust_text(texts, x, y, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt_name1 = str(weights_layer).replace(".", "_") + "_" + str(
        distance_matrix) + '_' + 'test_accuracy_records' + '.png'
    plt.savefig(folder_path + '/' + plt_name1, format='png')

    # summarize history for loss
    plt.clf()
    plt.cla()
    plt.close()
    fig = plt.figure(figsize=(8, 6))
    y = np.array(test_loss_records)
    ax = fig.add_subplot(111)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y, label='Validation', color='orange')
    plt.legend()
    """
    for i, j in zip(x, y):
        ax.annotate("{:.4f}".format(j), xy=(i, j), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
    """
    texts = [plt.text(x[i], y[i], '%s' % round(y[i], 3)) for i in range(len(x))]
    adjust_text(texts, x, y, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt_name2 = str(weights_layer).replace(".", "_") + "_" + str(distance_matrix) + '_' + 'test_loss_records' + '.png'
    plt.savefig(folder_path + '/' + plt_name2, format='png')
    return None

def plot_node_w_order(folder_path, history):
    history.pop(0)
    nodes = [row[0] for row in history]

    x = [point[0] for point in nodes]
    y = [point[1] for point in nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, color='c', marker='.', linestyle='dashed')

    plt.ylabel('zoom facter')
    plt.xlabel('goal concentration')

    improved_nodes = []
    for i in range(len(nodes)):
        if history[i][2] <= 0:
            improved_nodes.append(history[i][0])

    texts = [plt.text(x[i], y[i], i + 1) for i in range(len(history))]
    adjust_text(texts, x, y, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    x = [point[0] for point in improved_nodes]
    y = [point[1] for point in improved_nodes]

    ax.scatter(x, y, color='red')
    title_string = "UAM agent search path"
    plt.title(title_string, fontsize=15)
    plt_name = 'rl_search_path' + '.png'

    plt.savefig(folder_path + '/' + plt_name, format='png')
    # plt.show()
    return None


def plot_node_w_order_1d(folder_path, history):
    history.pop(0)
    nodes = [row[0] for row in history]
    v = [row[2] for row in history]
    n = len(nodes)

    x = [i for i in range(1, n + 1)]
    if len(nodes[0])==1:
        y = [point[0] for point in nodes]
        plot_uam_search(nodes, history, v, folder_path, x, y, plot_name="am")
    elif len(nodes[0])==2:
        y1 = [point[0] for point in nodes]
        y2 = [point[1] for point in nodes]
        plot_uam_search(nodes, history, v, folder_path, x, y1, plot_name="am")
        plot_uam_search(nodes, history, v, folder_path, x, y2, plot_name="gc")

    return None

def plot_uam_search(nodes, history, v, folder_path, x, y, plot_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, color='c', marker='.', linestyle='dashed')

    j = None
    plt.xlabel('FL iterations')
    if plot_name == 'am':
        plt.ylabel('attack magnitude')
        j = 0
    elif plot_name == 'gc':
        plt.ylabel('goal concentration')
        j = 1

    improved_nodes = []
    for i in range(len(nodes)):
        if history[i][2] <= 0:
            improved_nodes.append([history[i][0], i + 1])

    texts = [plt.text(x[i], y[i], round(v[i], 3)) for i in range(len(history))]
    adjust_text(texts, x, y, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    x = [point[1] for point in improved_nodes]
    y = [point[0][j] for point in improved_nodes]

    ax.scatter(x, y, color='red')
    title_string = "UAM agent search path"
    plt.title(title_string, fontsize=15)
    plt_name = 'rl_search_path_' + plot_name + '.png'

    plt.savefig(folder_path + '/' + plt_name, format='png')


def images_to_video(image_folder_path: str, output_path: str, fps, extension: str, video_name: str, output_format: str):
    images = [image_folder_path + '/' + img for img in os.listdir(image_folder_path) if img.endswith(extension)]
    images.sort(key=lambda s: os.path.getmtime(os.path.join(s)))

    movie_clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps)
    movie_clip.write_videofile(output_path + '/' + video_name + output_format, codec="libx264")
    return None

