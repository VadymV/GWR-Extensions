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
Methods that define different learning strategies.
"""

import random
import numpy as np
import pickle

from gammagwr import GammaGWR
from utilities import Utilities

CONTEXT_BETA = 0.7
LEARNING_RATE_B = 0.3
LEARNING_RATE_N = 0.003
FILES_FOLDER = './saved_data/*'
DATA_DIMENSION = 256  # Number of features in the dataset.

CATEGORIES_50_NICO = ['ball', 'ball', 'ball', 'ball', 'ball', 'bottle', 'bottle', 'bottle', 'bottle', 'bottle',
           'candle', 'candle', 'candle', 'candle', 'candle', 'chocolate', 'chocolate', 'chocolate',
           'chocolate', 'chocolate', 'cup', 'cup', 'cup', 'cup', 'cup', 'hairbrush', 'hairbrush',
           'hairbrush', 'hairbrush', 'hairbrush', 'paintbrush', 'paintbrush', 'paintbrush', 'paintbrush', 'paintbrush',
                      'pencil case', 'pencil case',
           'pencil case', 'pencil case', 'pencil case', 'tetra pack', 'tetra pack', 'tetra pack', 'tetra pack', 'tetra pack',
                      'tube', 'tube', 'tube',
           'tube', 'tube']

CATEGORIES_50_ICWT = ['book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book',
                          'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush',
                          'hairbrush', 'hairbrush', 'hairbrush', 'hair clip', 'hair clip', 'hair clip', 'hair clip',
                          'hair clip', 'hair clip', 'hair clip', 'hair clip', 'hair clip', 'hair clip', 'flower', 'flower',
                          'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'glass',
                          'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass']

CATEGORIES_50_CORe50 = ['plug adapter', 'plug adapter', 'plug adapter', 'plug adapter', 'plug adapter',
                        'mobile phone', 'mobile phone', 'mobile phone', 'mobile phone', 'mobile phone',
                        'scissors', 'scissors', 'scissors', 'scissors', 'scissors',
                        'light bulb', 'light bulb', 'light bulb', 'light bulb', 'light bulb',
                        'can', 'can', 'can', 'can', 'can',
                        'glasses', 'glasses', 'glasses', 'glasses', 'glasses',
                        'ball', 'ball', 'ball', 'ball', 'ball',
                        'marker', 'marker', 'marker', 'marker', 'marker',
                        'cup', 'cup', 'cup', 'cup', 'cup',
                        'remote control', 'remote control', 'remote control', 'remote control', 'remote control']


class Learning:

    @staticmethod
    def iterative_learning(train_data, test_data, args, category_column):
        rgwr = GammaGWR()
        utils = Utilities()

        utils.remove_files(FILES_FOLDER)  # Clear the directory for new data.

        train_accuracies = []
        test_accuracies = []
        mini_batch_size = 5

        iterations = 10
        all_object_classes = np.unique(train_data[:, category_column])
        random.shuffle(all_object_classes)

        rgwr.initNetwork(DATA_DIMENSION, numClasses=args.num_classes, numWeights=args.num_weights)

        for iteration in range(0, iterations):
            objects_to_learn = all_object_classes[mini_batch_size * iteration:mini_batch_size * iteration +
                                                  mini_batch_size]

            # Learn the model.
            train_data_prepared = train_data[np.in1d(train_data[:, category_column], objects_to_learn)]

            train_dataset, train_dimension, train_labelSet = utils.describe_data(train_data_prepared, args.dataset_name)
            rgwr.train(train_dataset, train_labelSet, maxEpochs=args.epochs, insertionT=args.threshold,
                       beta=CONTEXT_BETA, epsilon_b=LEARNING_RATE_B, epsilon_n=LEARNING_RATE_N)

            # Test the model.
            test_dataset, test_dimension, test_labelSet = utils.describe_data(test_data, args.dataset_name)
            train_accuracy = rgwr.evaluate_model(rgwr, train_dataset, train_labelSet, mode='train')
            test_accuracy = rgwr.evaluate_model(rgwr, test_dataset, test_labelSet, mode='test')

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            utils.save_predicted_metrics(rgwr, iteration)

            if iteration == 0:
                number_neurons = pickle.load(open("./saved_data/num_neurons" + str(iteration) + '.file', "rb"))
            else:
                previous_neurons = pickle.load(open("./saved_data/num_neurons" + str(iteration) + '.file', "rb"))
                number_neurons = np.append(number_neurons, previous_neurons)

        with open('./saved_data/test_accuracies.file', "wb") as f:
            pickle.dump(test_accuracies, f, pickle.HIGHEST_PROTOCOL)
        with open('./saved_data/train_accuracies.file', "wb") as f:
            pickle.dump(train_accuracies, f, pickle.HIGHEST_PROTOCOL)
        with open("./saved_data/num_neurons.file", "wb") as f:
            pickle.dump(number_neurons, f, pickle.HIGHEST_PROTOCOL)

        print("Object classes order: ", all_object_classes)
        print("Train accuracies: ", train_accuracies)
        print("Test accuracies: ", test_accuracies)

    @staticmethod
    def batch_learning(train_data, test_data, args):
        rgwr = GammaGWR()
        utils = Utilities()

        utils.remove_files(FILES_FOLDER)  # Clear the directory for new data.

        # Learn the model.
        train_dataset, train_dimension, train_labelSet = utils.describe_data(train_data, args.dataset_name)
        rgwr.initNetwork(train_dimension, numClasses=args.num_classes, numWeights=args.num_weights)
        rgwr.train(train_dataset, train_labelSet, maxEpochs=args.epochs, insertionT=args.threshold,
                   beta=CONTEXT_BETA, epsilon_b=LEARNING_RATE_B, epsilon_n=LEARNING_RATE_N)

        # Test the model.
        test_dataset, test_dimension, test_labelSet = utils.describe_data(test_data, args.dataset_name)
        rgwr.evaluate_model(rgwr, train_dataset, train_labelSet, mode='train')
        rgwr.evaluate_model(rgwr, test_dataset, test_labelSet, mode='test')

        utils.save_predicted_metrics(rgwr, '')

