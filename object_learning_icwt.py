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
Object learning with ICWT using the RGWR network.
"""
import numpy as np

from gammagwr import GammaGWR
from utilities import Utilities
from learning import Learning


if __name__ == "__main__":

    # ------------------------------------ Global variables ------------------------------------------------------------
    BATCH_LEARNING = 0  # Evaluate as batch?
    ITERATIVE_LEARNING = 1  # Evaluate as iterative learning?
    ONE_DAY = [1, 3, 5, 7]  # Select a day to evaluate. Only two days for a one category are available.
    # Second day contains even numbers. In total 8.

    CATEGORIES = [0, 10, 11, 5, 6]  # Select categories to evaluate.
    TRAIN_SESSIONS = [2, 3, 4, 5]  # All sessions, except MIX.
    TEST_SESSIONS = [1]  # MIX session.
    CAMERA = 1  # Two cameras available. right=2.
    NUMBER_FRAMES = 75  # The number of frames per session. 4 sessions make 300 (75 * 4) session for an object.
    CATEGORY_COLUMN = 256
    INSTANCE_COLUMN = 257
    SESSION_COLUMN = 258
    DAY_COLUMN = 259
    CAMERA_COLUMN = 260
    IMAGE_NAME_COLUMN = 261
    DATA_DIMENSION = 256
    FACTOR_FRAMES = 2  # Every Nth frame will be selected. Only 2 and 4 are reasonable values. Original number of
    # frames is 8. In this case it will be reduced to 4 and 2, respectively.

    # ------------------------------------ Initialization --------------------------------------------------------------

    rgwr = GammaGWR()
    utils = Utilities()
    learning = Learning()
    args = utils.parse_arguments()

    # Get data.
    original_data = utils.load_data(args.dataset).values
    original_data_normalized = utils.normalize_data(original_data, DATA_DIMENSION)

    original_data_day_one = original_data_normalized[np.in1d(original_data_normalized[:, DAY_COLUMN], ONE_DAY)]
    original_data_left_camera = original_data_day_one[np.in1d(original_data_day_one[:, CAMERA_COLUMN], CAMERA)]
    selected_data = original_data_left_camera[np.in1d(original_data_left_camera[:, CATEGORY_COLUMN], CATEGORIES)]

    # Comment if categorization instead of identification to use. For the rest of the evaluation the CATEGORY column
    # will be used as the class label column for the objects.
    instances = np.apply_along_axis(utils.add_instances, category_column=CATEGORY_COLUMN,
                                    instance_column=INSTANCE_COLUMN, axis=1, arr=selected_data)
    selected_data[:, CATEGORY_COLUMN] = instances

    # Bring class labels to the range within the number of the classes.
    fitted_class_labels = utils.fit_class_labels(selected_data[:, CATEGORY_COLUMN], args.num_classes)
    selected_data[:, CATEGORY_COLUMN] = fitted_class_labels

    train_data = selected_data[np.in1d(selected_data[:, SESSION_COLUMN], TRAIN_SESSIONS)]
    train_data = utils.reduce_number_of_frames(train_data, FACTOR_FRAMES)
    train_data = utils.select_frames_per_session(train_data, NUMBER_FRAMES, FACTOR_FRAMES, CATEGORY_COLUMN,
                                                 SESSION_COLUMN, IMAGE_NAME_COLUMN, TRAIN_SESSIONS)

    test_data = selected_data[np.in1d(selected_data[:, SESSION_COLUMN], TEST_SESSIONS)]

    # ------------------------------------ Batch learning---------------------------------------------------------------
    if BATCH_LEARNING:
        learning.batch_learning(train_data, test_data, args)

    # ------------------------------------ Iterative learning ----------------------------------------------------------
    if ITERATIVE_LEARNING:
        learning.iterative_learning(train_data, test_data, args, CATEGORY_COLUMN)


