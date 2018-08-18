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
Object learning with NICO Vision using the RGWR network.
"""
import random
import numpy as np

from gammagwr import GammaGWR
from utilities import Utilities
from learning import Learning


if __name__ == "__main__":

    # ------------------------------------ Global variables ------------------------------------------------------------
    BATCH_LEARNING = 0  # Evaluate as batch?
    ITERATIVE_LEARNING = 0  # Evaluate as iterative learning?
    NOVELTY_DETECTION = 1  # Evaluate for novelty detection?
    TRAIN_SESSIONS = [2]  # Select sessions for training.
    TEST_SESSIONS = [1]  # Select sessions for testing.
    TRAIN_INSTANCES = range(0, 50)  # Select instances for training.
    TEST_INSTANCES = range(0, 50)  # Select instances for testing.
    INSTANCE_COLUMN = 257  # Object class labels.
    SESSION_COLUMN = 258  # Sessions.
    IMAGE_NAME_COLUMN = 259  # Name of the image.
    CONTEXT_BETA = 0.7
    LEARNING_RATE_B = 0.3
    LEARNING_RATE_N = 0.003
    DATA_DIMENSION = 256  # Number of features in the dataset.
    FILES_FOLDER = './saved_data/*'
    FACTOR_FRAMES = 2  # Every Nth frame will be selected. Only 2 and 4 are reasonable values. Original number of
    # frames is 25. In this case it will be reduced to 12 and 6, respectively.

    # Used for plotting.
    CATEGORIES_50_NICO = ['book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book',
                          'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush', 'hairbrush',
                          'hairbrush', 'hairbrush', 'hairbrush', 'hair clip', 'hair clip', 'hair clip', 'hair clip',
                          'hair clip', 'hair clip', 'hair clip', 'hair clip', 'hair clip', 'hair clip', 'flower', 'flower',
                          'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'flower', 'glass',
                          'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass']

    # ------------------------------------ Initialization --------------------------------------------------------------

    rgwr = GammaGWR()
    utils = Utilities()
    learning = Learning()
    args = utils.parse_arguments()

    # Get data.
    original_data = utils.load_data(args.dataset).values
    original_data_normalized = utils.normalize_data(original_data, DATA_DIMENSION)
    # original_data_normalized = original_data

    # Get training data.
    train_data = original_data_normalized[np.in1d(original_data_normalized[:, SESSION_COLUMN], TRAIN_SESSIONS)]
    train_data = train_data[np.in1d(train_data[:, INSTANCE_COLUMN], TRAIN_INSTANCES)]
    train_data = utils.reduce_number_of_frames(train_data, FACTOR_FRAMES)

    # Get testing data.
    test_data = original_data_normalized[np.in1d(original_data_normalized[:, SESSION_COLUMN], TEST_SESSIONS)]
    test_data = test_data[np.in1d(test_data[:, INSTANCE_COLUMN], TEST_INSTANCES)]

    # ------------------------------------ Batch learning---------------------------------------------------------------
    if BATCH_LEARNING:
        learning.batch_learning(train_data, test_data, args)

    # ------------------------------------ Iterative learning ----------------------------------------------------------
    if ITERATIVE_LEARNING:
        learning.iterative_learning(train_data, test_data, args, INSTANCE_COLUMN)

    # ------------------------------------ Novelty detection ----------------------------------------------------------
    if NOVELTY_DETECTION:
        utils.remove_files(FILES_FOLDER)  # Clear the directory for new data.
        test_accuracies = []
        learnt_objects = []
        fp = []
        tp = []
        fn = []
        tn = []
        novel_objects_detected = 0
        learnt_objects_detected = 0
        all_object_classes = np.unique(train_data[:, INSTANCE_COLUMN])
        random.shuffle(all_object_classes)

        rgwr.initNetwork(DATA_DIMENSION, numClasses=args.num_classes, numWeights=args.num_weights)
        activation_mean = 0
        activation_sd = 0

        for i in all_object_classes[:30]:
            # Learn the model.
            train_data_prepared = train_data[np.in1d(train_data[:, INSTANCE_COLUMN], i)]
            train_dataSet, train_dimension, train_labelSet = utils.describe_data(train_data_prepared,
                                                                                 args.dataset_name)
            rgwr.train(train_dataSet, train_labelSet, maxEpochs=args.epochs, insertionT=args.threshold,
                       beta=CONTEXT_BETA, epsilon_b=LEARNING_RATE_B, epsilon_n=LEARNING_RATE_N)
            learnt_objects.append(i)

            train_data_prepared = train_data[np.in1d(train_data[:, INSTANCE_COLUMN], learnt_objects)]
            train_dataSet, train_dimension, train_labelSet = utils.describe_data(train_data_prepared,
                                                                                 args.dataset_name)
            novelty = utils.check_novelty(rgwr, train_dataSet, mode='train')
            activation_mean = novelty[0]
            activation_sd = novelty[1]

        random.shuffle(all_object_classes)
        iteration = 1
        for object_to_learn in all_object_classes[:50]:
            train_data_prepared = train_data[np.in1d(train_data[:, INSTANCE_COLUMN], object_to_learn)]
            train_dataSet, train_dimension, train_labelSet = utils.describe_data(train_data_prepared, args.dataset_name)
            novelty_value = utils.check_novelty(rgwr, train_dataSet, mode='evaluate')[0]
            threshold = activation_mean - 1 * activation_sd

            print("---- Iteration ----: ", iteration)
            print("Object to learn: ", object_to_learn)
            print("Learnt objects: ", learnt_objects)
            print("Novelty value: ", novelty_value)
            print("Activation mean: ", activation_mean)
            print("Activation sd: ", activation_sd)
            print("Threshold: ", threshold)
            if novelty_value < threshold:
                print("Assumption. Object is novel: ", object_to_learn)
                print("Learnt objects: ", learnt_objects)
                rgwr.train(train_dataSet, train_labelSet, maxEpochs=args.epochs, insertionT=args.threshold,
                           beta=CONTEXT_BETA, epsilon_b=LEARNING_RATE_B, epsilon_n=LEARNING_RATE_N)
                if object_to_learn not in learnt_objects:
                    learnt_objects.append(object_to_learn)
                    tp.append(object_to_learn)
                else:
                    fp.append(object_to_learn)
                novel_objects_detected += 1
            else:
                if object_to_learn not in learnt_objects:
                    fn.append(object_to_learn)
                else:
                    tn.append(object_to_learn)
                # Test the model.
                print("Assumption. Object is not novel: ", object_to_learn)
                test_data_prepared = test_data[np.in1d(test_data[:, INSTANCE_COLUMN], object_to_learn)]
                test_dataSet, test_dimension, test_labelSet = utils.describe_data(test_data_prepared, args.dataset_name)
                test_accuracy = rgwr.evaluate_model(rgwr, test_dataSet, test_labelSet, mode='test')
                test_accuracies.append(test_accuracy)
                learnt_objects_detected += 1

            iteration += 1

            # utils.save_predicted_metrics(rgwr, iteration)

        print("Test accuracies: ", test_accuracies)
        print("Test accuracy: ", sum(test_accuracies) / len(test_accuracies))
        print("Probable novel objects: ", novel_objects_detected)
        print("Probable known objects: ", learnt_objects_detected)
        print("FP: ", fp)
        print("TP: ", tp)
        print("FN: ", fn)
        print("TN: ", tn)
        precision = len(tp) / (len(tp) + len(fp))
        recall = len(tp) / (len(tp) + len(fn))
        F1 = 2 * (precision * recall) / (precision + recall)

        print('F1: ', F1)




