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
Object learning with CORe50 using the RGWR network.
"""
import numpy as np
0
from gammagwr import GammaGWR
from utilities import Utilities
from learning import Learning

if __name__ == "__main__":

    # ------------------------------------ Global variables ------------------------------------------------------------
    BATCH_LEARNING = 0  # Evaluate as batch?
    ITERATIVE_LEARNING = 1  # Evaluate as iterative learning?
    IMAGE_NAME_COLUMN = 256
    SESSION_COLUMN = 257
    CATEGORY_COLUMN = 258
    INSTANCE_COLUMN = 259
    TRAIN_SESSIONS = [1, 2, 4, 5, 6, 8, 9, 11]
    TEST_SESSIONS = [3, 7, 10]
    DATA_DIMENSION = 256
    FACTOR_FRAMES = 4  # Every Nth frame will be selected. Only 2 and 4 are reasonable values. Original number of
    # frames is 8. In this case it will be reduced to 4 and 2, respectively.

    # ------------------------------------ Initialization --------------------------------------------------------------

    rgwr = GammaGWR()
    utils = Utilities()
    learning = Learning()
    args = utils.parse_arguments()

    # Get data.
    original_data = utils.load_data(args.dataset).values
    original_data_normalized = utils.normalize_data(original_data, DATA_DIMENSION)

    train_data = original_data_normalized[np.in1d(original_data_normalized[:, SESSION_COLUMN], TRAIN_SESSIONS)]
    train_data = utils.reduce_number_of_frames(train_data, FACTOR_FRAMES)

    test_data = original_data_normalized[np.in1d(original_data_normalized[:, SESSION_COLUMN], TEST_SESSIONS)]

    # ------------------------------------ Batch learning---------------------------------------------------------------
    if BATCH_LEARNING:
        learning.batch_learning(train_data, test_data, args)

    # ------------------------------------ Iterative learning ----------------------------------------------------------
    if ITERATIVE_LEARNING:

        learning.iterative_learning(train_data, test_data, args, INSTANCE_COLUMN)
