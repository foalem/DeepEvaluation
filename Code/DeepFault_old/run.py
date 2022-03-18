"""
This is the main file that executes the flow of DeepFault
"""
import numpy as np

from test_nn import test_model
from os import path
from spectrum_analysis import *
from utils import save_perturbed_test_groups, load_perturbed_test_groups
from utils import load_suspicious_neurons, save_suspicious_neurons
from utils import create_experiment_dir, get_trainable_layers
from utils import load_classifications, save_classifications
from utils import save_layer_outs, load_layer_outs, construct_spectrum_matrices
from utils import load_MNIST, load_CIFAR, load_model
from utils import filter_val_set, save_original_inputs
from input_synthesis import synthesize
from sklearn.model_selection import train_test_split
import datetime
import argparse
import random
import tensorflow as tf
import os

experiment_path = "experiment_results"
# model_path = "./neural_networks"
group_index = 1
__version__ = "v1.0"


def color_preprocessing(x_test):
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_test


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Spectrum Based Fault Localization for Deep Neural Networks'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version", help="show program version",
                        action="version", version="DeepFault " + __version__)
    parser.add_argument("-M", "--model", help="The model to be loaded. The \
                        specified model will be analyzed.", choices=["lenet5", "resnet20", "vgg16"], required=True)
    parser.add_argument("-SE", "--seed", help="The seed to be used (mnist\
                        or cifar10).", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to localize dominant neurons", choices=["tarantula", "ochiai", "dstar"], required=True)
    parser.add_argument("-D", "--distance", help="the distance between the \
                        original and the mutated image.", type=float)
    # parser.add_argument("-C", "--class", help="the label of inputs to \
    #                     analyze.", type=int)
    # parser.add_argument("-AC", "--activation", help="activation function \
    #                     or hidden neurons. it can be \"relu\" or \"leaky_relu\"")
    # parser.add_argument("-SN", "--suspicious_num", help="number of suspicious \
    #                     neurons we consider", type=int)
    # parser.add_argument("-SS", "--step_size", help="multiplication of \
    #                     gradients by step size", type=float)
    # parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
    #                     the cases where you need to run the same experiment \
    #                     multiple times)", type=int)
    # parser.add_argument("-S", "--seed", help="Seed for random processes. \
    #                     If not provided seed will be selected randomly.", type=int)
    parser.add_argument("-ST", "--star", help="DStar\'s Star \
                        hyperparameter. Has an effect when selected approach is\
                        DStar", type=int)
    # parser.add_argument("-LOG", "--logfile", help="path to log file")

    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args['model']
    seed_class = args['seed']
    # selected_class = args['class'] if not args['class'] == None else 0
    # step_size = args['step_size'] if not args['step_size'] == None else 1
    distance = args['distance'] if not args['distance'] == None else 0.1
    approach = args['approach']
    # susp_num = args['suspicious_num'] if not args['suspicious_num'] == None else 1
    # repeat = args['repeat'] if not args['repeat'] == None else 1
    # seed = args['seed'] if not args['seed'] == None else random.randint(0, 10)
    star = args['star'] if not args['star'] == None else 3
    # logfile_name = args['logfile'] if not args['logfile'] == None else 'result.log'
    parse_arguments()


    # model_name = "lenet5"
    # seed_class = "0"
    # # selected_class = args['class'] if not args['class'] == None else 0
    # # step_size = args['step_size'] if not args['step_size'] == None else 1
    # distance = 0.1
    # approach = 'tarantula'
    # # susp_num = args['suspicious_num'] if not args['suspicious_num'] == None else 1
    # # repeat = args['repeat'] if not args['repeat'] == None else 1
    # # seed = args['seed'] if not args['seed'] == None else random.randint(0, 10)
    # star = 3
    # logfile_name = 'result.log'

    # model_name = "lenet5"
    # model_name = "vgg16"
    # model_name = "resnet20"
    # dataset = "mnist"
    # dataset = "cifar"
    # selected_class = 0
    # step_size = 1
    # distance = 0.1
    # approach = 'tarantula'
    # approach = 'ochiai'
    # approach = 'dstar'
    # approach = 'random'
    # susp_num = 10
    repeat = 1
    random_seed = 10
    # star = 3
    # logfile_name = 'result.log'

    # file_path = "./lenet5.h5"
    # file_path = "./vgg16.h5"
    # file_path = "./resnet.h5"

    if model_name == 'lenet5':
        dataset = "mnist"
        step_size = 1
        susp_num = 1
    else:
        dataset = "cifar"
        step_size = 10
        susp_num = 1

    selected_class = int(seed_class)

    logfile_name = os.path.join("sythetic_data", model_name, "seed_" + str(seed_class) + "_" + model_name + "_" +
                                approach + "_result.log")

    ####################
    # 0) Load MNIST or CIFAR10 data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(one_hot=True)
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=1 / 6.0,
                                                      random_state=random_seed)

    logfile = open(logfile_name, 'a')

    ####################
    # 1) Load the pretrained network.
    try:
        # model = load_model(path.join(model_path, model_name))
        # model = load_model(file_path)
        # model.summary()
        file_path = os.path.join("models", model_name + '.h5')
        model = tf.keras.models.load_model(file_path)
    except:
        logfile.write("Model not found! Provide a pre-trained model as input.")
        exit(1)

    experiment_name = create_experiment_dir(experiment_path, model_name,
                                            selected_class, step_size,
                                            approach, susp_num, repeat)

    # Fault localization is done per class.
    X_val, Y_val = filter_val_set(selected_class, X_test, Y_test)
    if dataset == 'mnist':
        X_val = X_val.reshape((-1, 28, 28, 1))
    elif dataset == 'cifar':
        X_val = color_preprocessing(X_val)

    # X_val = color_preprocessing(X_val)
    # model_outputs = model.predict(X_val)
    # model_predictions = np.argmax(model_outputs, axis=1)
    # ground_truth = np.argmax(Y_val, axis=1)
    # a = model_predictions == ground_truth
    # a1 = np.sum(a)
    # accuracy = np.sum(model_predictions == ground_truth) / len(ground_truth)
    # print(accuracy)

    ####################
    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for test input x.
    filename = experiment_path + '/' + model_name + '_' + str(selected_class)
    try:
        correct_classifications, misclassifications = load_classifications(filename, group_index)
        layer_outs = load_layer_outs(filename, group_index)
    except:
        correct_classifications, misclassifications, layer_outs, predictions = \
            test_model(model, X_val, Y_val)
        save_classifications(correct_classifications, misclassifications,
                             filename, group_index)
        save_layer_outs(layer_outs, filename, group_index)

    ####################
    # 3) Receive the correct classifications  & misclassifications and identify
    # the suspicious neurons per layer
    trainable_layers = get_trainable_layers(model)
    scores, num_cf, num_uf, num_cs, num_us = construct_spectrum_matrices(model,
                                                                         trainable_layers,
                                                                         correct_classifications,
                                                                         misclassifications,
                                                                         layer_outs)

    filename = experiment_path + '/' + model_name + '_C' + str(selected_class) + '_' + \
               approach + '_SN' + str(susp_num)

    if approach == 'tarantula':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = tarantula_analysis(trainable_layers, scores,
                                                       num_cf, num_uf, num_cs, num_us,
                                                       susp_num)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'ochiai':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = ochiai_analysis(trainable_layers, scores,
                                                    num_cf, num_uf, num_cs, num_us,
                                                    susp_num)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'dstar':
        try:
            suspicious_neuron_idx = load_suspicious_neurons(filename, group_index)
        except:
            suspicious_neuron_idx = dstar_analysis(trainable_layers, scores,
                                                   num_cf, num_uf, num_cs, num_us,
                                                   susp_num, star)

            save_suspicious_neurons(suspicious_neuron_idx, filename, group_index)

    elif approach == 'random':
        # Random fault localization has to be run after running Tarantula,
        # Ochiai and DStar with the same parameters.

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
                   + '_tarantula_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_tarantula = load_suspicious_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
                   + '_ochiai_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_ochiai = load_suspicious_neurons(filename, group_index)

        filename = experiment_path + '/' + model_name + '_C' + str(selected_class) \
                   + '_dstar_' + 'SN' + str(susp_num)

        suspicious_neuron_idx_dstar = load_suspicious_neurons(filename, group_index)

        forbiddens = suspicious_neuron_idx_ochiai + suspicious_neuron_idx_tarantula + \
                     suspicious_neuron_idx_dstar

        forbiddens = [list(forb) for forb in forbiddens]

        available_layers = list(([elem[0] for elem in suspicious_neuron_idx_tarantula]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_ochiai]))
        available_layers += list(set([elem[0] for elem in suspicious_neuron_idx_dstar]))

        suspicious_neuron_idx = []
        while len(suspicious_neuron_idx) < susp_num:
            l_idx = random.choice(available_layers)
            n_idx = random.choice(range(model.layers[l_idx].output_shape[1]))

            if [l_idx, n_idx] not in forbiddens and [l_idx, n_idx] not in suspicious_neuron_idx:
                suspicious_neuron_idx.append([l_idx, n_idx])

    logfile.write('Suspicous neurons: ' + str(suspicious_neuron_idx) + '\n')

    ####################
    # 4) Run Suspiciousness-Guided Input Synthesis Algorithm
    # Receive the set of suspicious neurons for each layer from Step 3 # and
    # will produce new inputs based on the correct classifications (from the
    # testing set) that exercise the suspicious neurons

    perturbed_xs = []
    perturbed_ys = []

    # select 10 inputs randomly from the correct classification set.
    # selected = np.random.choice(list(correct_classifications), 10)
    #
    # # zipped_data = zip(, )
    # x_original = list(np.array(X_val)[selected])
    # y_original = list(np.array(Y_val)[selected])

    file_name = "seed_" + str(seed_class) + "_" + model_name + "_data_prep.npy"
    file_path = os.path.join("data_seeds", model_name, file_name)
    x_original = list(np.load(file_path))
    file_name = "seed_" + str(seed_class) + "_" + model_name + "_label.npy"
    file_path = os.path.join("data_seeds", model_name, file_name)
    y_original = np.load(file_path)
    y_original = list(tf.keras.utils.to_categorical(y_original, num_classes=10))

    # save_original_inputs(x_original, filename, group_index)

    syn_start = datetime.datetime.now()
    x_perturbed = synthesize(model, x_original, suspicious_neuron_idx, step_size, distance)
    syn_end = datetime.datetime.now()

    # perturbed_xs = perturbed_xs + x_perturbed
    # perturbed_ys = perturbed_ys + y_perturbed

    # reshape them into the expected format
    # perturbed_xs = np.asarray(perturbed_xs).reshape(np.asarray(perturbed_xs).shape[0], *X_val[0].shape)
    # perturbed_ys = np.asarray(perturbed_ys).reshape(np.asarray(perturbed_ys).shape[0], 10)

    # save perturtbed inputs
    # filename = path.join(experiment_path, experiment_name)
    # save_perturbed_test_groups(x_perturbed, y_original, filename, group_index)

    file_name = "seed_" + str(seed_class) + "_" + model_name + "_" + approach + "_data_prep_synth.npy"
    file_path = os.path.join("sythetic_data", model_name, file_name)
    np.save(file_path, np.array(x_perturbed))
    file_name = "seed_" + str(seed_class) + "_" + model_name + "_" + approach + "_label.npy"
    file_path = os.path.join("sythetic_data", model_name, file_name)
    np.save(file_path, np.array(y_original))

    ####################
    # 5) Test if the mutated inputs are adversarial
    score = model.evaluate([x_perturbed], [y_original], verbose=0)
    logfile.write('Model: ' + model_name + ', Class: ' + str(selected_class) +
                  ', Approach: ' + approach + ', Distance: ' +
                  str(distance) + ', Score: ' + str(score) + '\n')

    logfile.write('Input Synthesis Time: ' + str(syn_end - syn_start) + '\n')

    print(score)

    logfile.close()

    '''
    Currently not available
    ####################
    # 6) retrain the model
    # train_model_fault_localisation(model, x_perturbed, y_perturbed, len(x_perturbed))
    model.fit(x_perturbed, y_perturbed, batch_size=32, epochs=10, verbose=1)

    ####################
    # 7) retest the model
    test_model(model, X_test, Y_test)
    '''
