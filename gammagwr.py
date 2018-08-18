# Copyright (C) 2018  German I. Parisi (german.parisi@gmail.com)
#
# Modifications copyright (C) 2018  Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
# Date modified: 29 July 2018
#
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
Gamma-GWR (Recurrent Grow When Required)

@last-modified: 3 July 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2017) Lifelong Learning of Human Actions with Deep Neural Network Self-Organization. Neural Networks 96:137-149.
"""
import pickle

import numpy as np
import math

from numba import jit
from sklearn.metrics import confusion_matrix, f1_score


@jit
def find_best_matching_neurons(num_nodes, num_weights, var_alpha, global_context, recurrent_weights, distances):
    """
    Finds the best-matching neurons. The following lines of code in this method were extracted from the provided
    file 'gammagwr.py' to speed up the performance during the training using numba library (https://numba.pydata.org/)
    with @jit annotation.
    """

    for i in range(0, num_nodes):
        gammaDistance = 0.0
        for j in range(0, num_weights):
            gammaDistance += (var_alpha[j] * (np.sqrt(np.sum((global_context[j] - recurrent_weights[i, j]) ** 2))))
        distances[i] = gammaDistance


class GammaGWR:

    def initNetwork(self, dimension, numWeights, numClasses):
        self.numNodes = 2
        self.dimension = dimension
        self.numWeights = numWeights
        self.numOfClasses = numClasses
        self.recurrentWeights = np.zeros((self.numNodes, self.numWeights, self.dimension))
        self.alabels = np.zeros((self.numNodes, self.numOfClasses))
        self.globalContext = np.zeros((self.numWeights, self.dimension))
        self.edges = np.zeros((self.numNodes, self.numNodes))
        self.ages = np.zeros((self.numNodes, self.numNodes))
        self.habn = np.ones(self.numNodes)
        self.varAlpha = self.gammaWeights(self.numWeights)

    def gammaWeights(self, nw):
        iWe = np.zeros(nw)  # iWe[:] = 1 / nWeights
        for h in range(0, len(iWe)):
            iWe[h] = np.exp(-h)  # 1. / nWeights + np.exp(-h)
        iWe[:] = iWe[:] / sum(iWe)  # print iWe[:]
        return iWe

    def habituateNeuron(self, index, tau):
        self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)

    def updateNeuron(self, index, epsilon):
        deltaWeights = np.zeros((self.numWeights, self.dimension))
        for i in range(0, self.numWeights):
            deltaWeights[i] = np.array([np.dot((self.globalContext[i] - self.recurrentWeights[index, i]), epsilon)]) * self.habn[index]
        self.recurrentWeights[index] += deltaWeights

    def updateLabelHistogram(self, bmu, label):
        if (label != -1):
            for a in range(0, self.numOfClasses):
                if (a == label):
                    self.alabels[bmu, a] += self.aIncreaseFactor
                else:
                    self.alabels[bmu, a] -= self.aDecreaseFactor
                    if (self.alabels[bmu, a] < 0):
                        self.alabels[bmu, a] = 0

    def updateEdges(self, fi, si):
        neighboursFirst = np.nonzero(self.edges[fi])
        if (len(neighboursFirst[0]) >= self.maxNeighbours):
            remIndex = -1
            maxAgeNeighbour = 0
            for u in range(0, len(neighboursFirst[0])):
                if (self.ages[fi, neighboursFirst[0][u]] > maxAgeNeighbour):
                    maxAgeNeighbour = self.ages[fi, neighboursFirst[0][u]]
                    remIndex = neighboursFirst[0][u]
            self.edges[fi, remIndex] = 0
            self.edges[remIndex, fi] = 0
        self.edges[fi, si] = 1

    def removeOldEdges(self):
        for i in range(0, self.numNodes):
            neighbours = np.nonzero(self.edges[i])
            for j in range(0, len(neighbours[0])):
                if (self.ages[i, j] >= self.maxAge):
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0

    def removeIsolatedNeurons(self):
        indCount = 0
        while (indCount < self.numNodes):
            neighbours = np.nonzero(self.edges[indCount])
            if (len(neighbours[0]) < 1):
                self.recurrentWeights = np.delete(self.recurrentWeights, indCount, axis=0)
                self.alabels = np.delete(self.alabels, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=1)
                self.ages = np.delete(self.ages, indCount, axis=0)
                self.ages = np.delete(self.ages, indCount, axis=1)
                self.habn = np.delete(self.habn, indCount)
                self.numNodes -= 1
                print("(-- " + str(indCount) + ")")
            else:
                indCount += 1

    def train(self, dataSet, labelSet, maxEpochs, insertionT, beta, epsilon_b, epsilon_n):
        self.dataSet = dataSet
        self.samples = self.dataSet.shape[0]
        self.labelSet = labelSet
        self.maxEpochs = maxEpochs
        self.insertionThreshold = insertionT
        self.varBeta = beta
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n

        self.habThreshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = 10000
        self.maxAge = 200
        self.maxNeighbours = 6
        self.aIncreaseFactor = 1.
        self.aDecreaseFactor = 0.01

        self.nNN = np.zeros(self.maxEpochs)  # nNN[0] = 2
        self.qrror = np.zeros((self.maxEpochs, 2))
        self.fcounter = np.zeros((self.maxEpochs, 2))

        if (self.recurrentWeights[0:2, 0].all() == 0):
            self.recurrentWeights[0, 0] = self.dataSet[0]
            self.recurrentWeights[1, 0] = self.dataSet[1]

        # Start training
        Ti = 0
        previousBMU = np.zeros((1, self.numWeights, self.dimension))
        cu_qrror = np.zeros(self.samples)
        cu_fcounter = np.zeros(self.samples)
        print("Starting with " + str(self.numNodes) + " neurons...")

        for epoch in range(0, self.maxEpochs):
            for iteration in range(0, self.samples):
                self.globalContext[0] = self.dataSet[iteration]
                label = self.labelSet[iteration]
                Ti += 1
                # Update global context
                for z in range(1, self.numWeights):
                    self.globalContext[z] = (self.varBeta * previousBMU[0, z]) + ((1 - self.varBeta) * previousBMU[0, z - 1])
                # Find the best and second-best matching neurons
                distances = np.zeros(self.numNodes)
                numNodes = self.numNodes
                numWeights = self.numWeights
                varAlpha = self.varAlpha
                globalContext = self.globalContext
                recurrentWeights = self.recurrentWeights
                find_best_matching_neurons(numNodes, numWeights, varAlpha, globalContext, recurrentWeights, distances)

                sort_index = np.argsort(distances)
                firstIndex = sort_index[0]
                firstDistance = distances[firstIndex]
                secondIndex = sort_index[1]

                # spatialQE = np.sqrt(np.sum((globalContext[0]+recurrentWeights[firstIndex,0])**2))
                previousBMU[0] = self.recurrentWeights[firstIndex]
                self.ages += 1

                # Compute network activity
                cu_qrror[iteration] = firstDistance
                h = self.habn[firstIndex]
                cu_fcounter[iteration] = h
                a = math.exp(-firstDistance)

                if ((a < self.insertionThreshold) and (h < self.habThreshold) and (self.numNodes < self.maxNodes)):
                    # Add new weight
                    newRecurrentWeight = np.zeros((1, self.numWeights, self.dimension))
                    for i in range(0, self.numWeights):
                        newRecurrentWeight[0, i] = np.array([np.dot(self.recurrentWeights[firstIndex, i] + self.globalContext[i], 0.5)])
                    self.recurrentWeights = np.concatenate((self.recurrentWeights, newRecurrentWeight), axis=0)

                    newAlabel = np.zeros((1, self.numOfClasses))
                    if (label != -1):
                        newAlabel[0, int(label)] = self.aIncreaseFactor
                    self.alabels = np.concatenate((self.alabels, newAlabel), axis=0)

                    newIndex = self.numNodes
                    self.numNodes += 1
                    self.habn.resize(self.numNodes)
                    self.habn[newIndex] = 1

                    # Update edges
                    self.edges.resize((self.numNodes, self.numNodes))
                    self.edges[firstIndex, secondIndex] = 0
                    self.edges[secondIndex, firstIndex] = 0
                    self.edges[firstIndex, newIndex] = 1
                    self.edges[newIndex, firstIndex] = 1
                    self.edges[newIndex, secondIndex] = 1
                    self.edges[secondIndex, newIndex] = 1

                    # Update ages
                    self.ages.resize((self.numNodes, self.numNodes))
                    self.ages[firstIndex, newIndex] = 0
                    self.ages[newIndex, firstIndex] = 0
                    self.ages[newIndex, secondIndex] = 0
                    self.ages[secondIndex, newIndex] = 0

                else:
                    # Adapt weights and context descriptors
                    self.updateNeuron(firstIndex, self.epsilon_b)

                    # Adapt label histogram
                    self.updateLabelHistogram(firstIndex, label)

                    # Habituate BMU
                    self.habituateNeuron(firstIndex, self.tau_b)

                    # Update ages
                    self.ages[firstIndex, secondIndex] = 0
                    self.ages[secondIndex, firstIndex] = 0

                    # Update edges // Remove oldest ones
                    self.updateEdges(firstIndex, secondIndex)
                    self.updateEdges(secondIndex, firstIndex)

                    # Update topological neighbours
                    neighboursFirst = np.nonzero(self.edges[firstIndex])
                    for z in range(0, len(neighboursFirst[0])):
                        neIndex = neighboursFirst[0][z]
                        self.updateNeuron(neIndex, self.epsilon_n)
                        self.habituateNeuron(neIndex, self.tau_n)

            # Remove old edges
            self.removeOldEdges()

            self.nNN[epoch] = self.numNodes
            self.qrror[epoch, 0] = np.mean(cu_qrror)
            self.qrror[epoch, 1] = np.std(cu_qrror)
            self.fcounter[epoch, 0] = np.mean(cu_fcounter)
            self.fcounter[epoch, 1] = np.std(cu_fcounter)

            print("(E: " + str(epoch + 1) + ", NN: " + str(self.numNodes) + ", TQE: " + str(self.qrror[epoch, 0]) + " )")
            # print ("(Tsteps: " + str(Ti) + " )"),
            Ti = 0

        # Remove isolated neurons
        self.removeIsolatedNeurons()

        print("( Network size: " + str(self.numNodes) + " )")

    # Test GWR ################################################################

    @staticmethod
    def predictLabels(bmuLabel, labelSet, numWeights, numOfClasses):
        predictedLabels = np.zeros(len(bmuLabel) - numWeights)
        gtLabels = np.zeros(len(bmuLabel) - numWeights)
        counterAcc = 0
        for i in range(0, len(bmuLabel) - numWeights):
            counter = np.zeros(numOfClasses)
            counterGt = np.zeros(numOfClasses)
            for j in range(0, numWeights):
                counter[int(bmuLabel[i + j])] += 1
                counterGt[int(labelSet[i + j])] += 1
                # counterGt[int(labelSet[i+j+numWeights])] += 1
            predictedLabels[i] = np.argmax(counter)
            gtLabels[i] = np.argmax(counterGt)
            if (predictedLabels[i] == gtLabels[i]):
                counterAcc += 1
        avgAcc = (counterAcc * 100) / len(predictedLabels)

        return predictedLabels, gtLabels, avgAcc

    @staticmethod
    @jit
    def predict(dataSet, numWeights, dimension, numNodes, varAlpha, recurrentWeights, alabels):
        print("Predicting ...", )
        samples = dataSet.shape[0]
        bmuWeights = np.zeros((samples - (numWeights), dimension))
        bmuActivation = np.zeros(samples - (numWeights))
        bmuLabel = -np.ones(samples - (numWeights))
        inputContext = np.zeros((numWeights, dimension))
        for ti in range(0, samples - (numWeights)):
            for i in range(0, numWeights):
                inputContext[i] = dataSet[ti + i]
            distances = np.zeros(numNodes)
            for i in range(0, numNodes):
                gammaDistance = 0.0
                for j in range(0, numWeights):
                    gammaDistance += (varAlpha[j] * (np.sqrt(np.sum((inputContext[j] - recurrentWeights[i, j]) ** 2))))
                distances[i] = gammaDistance
            sort_index = np.argsort(distances)
            firstIndex = sort_index[0]
            bmuWeights[ti] = recurrentWeights[firstIndex, 0]
            bmuActivation[ti] = math.exp(-distances[firstIndex])
            bmuLabel[ti] = np.argmax(alabels[firstIndex, :])

        return bmuWeights, bmuActivation, bmuLabel

    @staticmethod
    def evaluate_model(my_network, data, label_set, mode=''):

        bmuWeights, bmuActivation, bmuLabel = my_network.predict(data, my_network.numWeights, my_network.dimension,
                                                                 my_network.numNodes, my_network.varAlpha,
                                                                 my_network.recurrentWeights, my_network.alabels)
        predictedLabels, ground_truth_labels, accuracy = my_network.predictLabels(bmuLabel, label_set,
                                                                                  my_network.numWeights,
                                                                                  my_network.numOfClasses)

        print("F1 score (" + mode + "): ", f1_score(ground_truth_labels, predictedLabels, average='weighted'))
        cm = confusion_matrix(ground_truth_labels, predictedLabels)
        with open('./saved_data/conf_matrix' + '_' + str(mode) + '.file', "wb") as f:
            pickle.dump(cm, f, pickle.HIGHEST_PROTOCOL)
        with open('./saved_data/f1_score' + '_' + str(mode) + '.file', "wb") as f:
            pickle.dump(f1_score(ground_truth_labels, predictedLabels, average='weighted'), f, pickle.HIGHEST_PROTOCOL)

        with open(('./saved_data/activations' + '_' + str(mode) + '.file'), "wb") as f:
            pickle.dump(bmuActivation, f, pickle.HIGHEST_PROTOCOL)

        if mode == 'train':
            print("Training classification accuracy: " + str(accuracy))
        if mode == 'test':
            print("Test classification accuracy: " + str(accuracy))

        return accuracy
