#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd

sys.path.append('./src/')

# Import the abalone data loader
from dataPreprocessing.loaders import load_abalone


def make_data(configurations):
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    train_dataset = []
    train_labels = []
    test_dataset = []
    test_labels = []

    # Use the load_abalone function to get the data
    X_train, y_train, X_test, y_test = load_abalone()

    # Convert to pandas DataFrames
    trainX = pd.DataFrame(X_train, columns=column_names)
    testX = pd.DataFrame(X_test, columns=column_names)

    # Save the data to CSV files as required by the template
    train_path = os.path.join(configurations['model_dir'], 'train_data.csv')
    test_path = os.path.join(configurations['model_dir'], 'test_data.csv')
    trainX.to_csv(train_path)
    testX.to_csv(test_path)

    train_dataset.append(trainX)
    test_dataset.append(testX)
    train_labels.append(pd.Series(y_train))
    test_labels.append(pd.Series(y_test))

    min_max_dict = getMinMaxRangeOfFeatures(trainX, configurations['feature_names'])
    return train_dataset, train_labels, test_dataset, test_labels, min_max_dict


def evaluate(model, test_dataset, test_labels, isClassification):
    scores = model.evaluate(test_dataset, test_labels, verbose=0)
    if isClassification:
        return scores[1]
    else:
        return scores[0]


def update_batch(model, layer_size, batch_data, batch_label, data_dir, batch_size):
    history = model.fit(batch_data, batch_label, epochs=1, batch_size=batch_size, validation_split=0.2, verbose=0)
    for i in range(layer_size):
        weight = model.layers[i].get_weights()[0]
        bias = model.layers[i].get_weights()[1]
        np.savetxt(data_dir + "/weights_layer%d.csv" % (i), weight, delimiter=",")
        np.savetxt(data_dir + "/bias_layer%d.csv" % (i), bias, delimiter=",")
    return model


def output(model, datapoint):
    x_point = pd.DataFrame(datapoint)
    return model.predict(x_point.transpose())


def getMinMaxRangeOfFeatures(dataset, column_names):
    min_max = {}
    for i in column_names:
        index = column_names.index(i)
        min_max[i] = [min(dataset[column_names[index]]), max(dataset[column_names[index]])]
    return min_max