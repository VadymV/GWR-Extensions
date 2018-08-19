# Copyright (C) 2018  Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
# Date created: 29 July 2018
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Utility class.
"""
import glob
import os
import pickle
import numpy as np
import argparse
import pandas as pd

from numba import jit
from random import shuffle


class Utilities:

    @staticmethod
    def remove_saved_files(files):
        for file in files:
            os.remove(os.path.join(os.getcwd(), file))

    @staticmethod
    def add_instances(row, category_column, instance_column):
        """
        A label of an instance is the combination of its category number and instance number in this category.

        :param instance_column:
        :param category_column:
        :param row: A row of the input data.
        :return: A label of an instance.
        """
        instance = float(str(int(row[category_column])) + "" + str(int(row[instance_column] - 1)))
        return instance

    @staticmethod
    def select_frames_per_session(data, number_frames, factor, category_column, sess_column, image_name_column,
                                  train_session):

        unique_classes = np.unique(data[:, category_column])
        for class_label in unique_classes:
            for session in train_session:
                class_label_data = data[np.in1d(data[:, category_column], class_label)]
                session_data = class_label_data[np.in1d(class_label_data[:, sess_column], session)]
                min_id = np.amin(session_data[:, image_name_column])
                threshold = min_id + number_frames * factor - 1
                frames_to_delete = np.logical_and.reduce([data[:, category_column] == class_label,
                                                          data[:, sess_column] == session,
                                                          data[:, image_name_column] > threshold])
                data = data[np.logical_not(frames_to_delete)]

        return data

    @staticmethod
    def reduce_number_of_frames(data, factor):
        """
        Delete frames.
        :param data:  Data.
        :param factor: An even integer value. Max is 6.
        :return:
        """
        assert factor == 2 or 4 or 6
        iterations = factor / 2
        for i in range(0, int(iterations)):
            data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)
        return data

    @staticmethod
    def fit_class_labels(column, num_classes):
        """
        Brings the labels of features into the range from 0 to num_classes.
        :param column: A column with data labels.
        :param num_classes: A number of classes.
        :return: Class labels with ids from 0 to num_classes.
        """
        for k in range(0, num_classes):
            minimum = np.amin(column[column >= k])
            column[column == minimum] = k
        return column

    @staticmethod
    def describe_data(data, dataset_name):
        if dataset_name == 'icwt':
            dimension = data.shape[1]
            class_labels = data[:, dimension - 6]
            dimension -= 6
            result = data[:, 0:dimension]
        if dataset_name == 'core50':
            dimension = data.shape[1]
            class_labels = data[:, dimension - 1]
            dimension -= 4
            result = data[:, 0:dimension]
        if dataset_name == 'nico':
            dimension = data.shape[1]
            class_labels = data[:, dimension - 3]
            dimension -= 4
            result = data[:, 0:dimension]

        return result, result.shape[1], class_labels

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Features learning with GWR-network")
        parser.add_argument("--dataset", dest="dataset", type=str, required=False,
                            help="Path to dataset?")
        parser.add_argument("--example", dest="example", type=int, default=1, help="Start example?")
        parser.add_argument("--threshold", dest="threshold", type=float, default=0.87, help="Threshold (0.87)")
        parser.add_argument("--num_weights", dest="num_weights", type=int, default=15, help="Number of wights (15)")
        parser.add_argument("--epochs", dest="epochs", type=int, default=1, help="Number of epochs (1)")
        parser.add_argument("--num_classes", dest="num_classes", type=int, default=8, help="Number of classes (8)")
        parser.add_argument("--dataset_name", dest="dataset_name", type=str, default=0, help="Dataset name")

        return parser.parse_args()

    @staticmethod
    def load_data(file_name):
        """
        Reads data from a csv file.

        Args:
            :param file_name: A path to a file.

        Returns:
            Data.
        """
        data = pd.read_csv(file_name, header=None, dtype=np.float64, sep=',')

        return data

    @staticmethod
    def check_novelty(my_network, data, mode=''):

        bmuWeights, bmuActivation, bmuLabel = my_network.predict(data, my_network.numWeights, my_network.dimension,
                                                                 my_network.numNodes, my_network.varAlpha,
                                                                 my_network.recurrentWeights, my_network.alabels)
        with open('./saved_data/novelty_activations' + '_' + str(mode) + '.file', "wb") as f:
            pickle.dump(bmuActivation, f, pickle.HIGHEST_PROTOCOL)
        return np.average(bmuActivation), np.std(bmuActivation)

    @staticmethod
    def save_predicted_metrics(rgwr, iteration=1):
        conf_matrix = pickle.load(open("./saved_data/conf_matrix_test.file", "rb"))
        with open(('./saved_data/conf_matrix' + str(iteration) + '.file'), "wb") as f:
            pickle.dump(conf_matrix, f, pickle.HIGHEST_PROTOCOL)
        with open(('./saved_data/qerror' + str(iteration) + '.file'), "wb") as f:
            pickle.dump(rgwr.qrror, f, pickle.HIGHEST_PROTOCOL)
        with open(('./saved_data/num_neurons' + str(iteration) + '.file'), "wb") as f:
            pickle.dump(rgwr.nNN, f, pickle.HIGHEST_PROTOCOL)
        with open(('./saved_data/fcounter'  + str(iteration) + '.file'), "wb") as f:
            pickle.dump(rgwr.fcounter, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def remove_files(path_to_folder):
        files = glob.glob(path_to_folder)
        for f in files:
            os.remove(f)

    @staticmethod
    @jit
    def normalize_data(data, columns):
        new_data = np.copy(data)
        for i in range(0, columns):
            max_column = max(data[:, i])
            min_column = min(data[:, i])
            for j in range(0, data.shape[0]):
                new_data[j, i] = (data[j, i] - min_column) / (max_column - min_column)
        return new_data

    @staticmethod
    def reorder_objects(data, objects_column):
        objects = np.unique(data[:, objects_column])
        shuffle(objects)
        iteration = 0
        for object_label in objects:
            d = data[np.in1d(data[:, objects_column], object_label)]
            if iteration == 0:
                new_data = np.copy(d)
            else:
                new_data = np.append(new_data, d)
            iteration += 1

        return new_data



