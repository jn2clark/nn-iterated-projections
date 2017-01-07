# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:07:22 2016

@author: jesseclark
"""

from __future__ import print_function

import core

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.models import model_from_json

import copy
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

OPTIMIZERS = {'Adam':Adam, 'SGD':SGD, 'RMSprop':RMSprop, 'Adagrad':Adagrad}


class ModelDM:

    def __init__(self,lr=.005, img_rows=28, img_cols=28,img_channels=1, n_l1=16, n_l2=None, dout=0, n_filt=8, opt='Adam',
                 regularizer=None, verbose=False, seed=1337, loss='categorical_crossentropy'):

        """
        wrapper for creating the model
        :param lr: learning rate
        :param img_rows: rows
        :param img_cols: columns
        :param img_channels: channels
        :param n_l1: number of nodes in first hidden layer
        :param n_l2: number of nodes in second hidden layer
        :param dout: the dropout fraction (0 = No dropout)
        :param n_filt: number fo conv filters
        :param opt: the optimizer
        :param regularizer: weight for regulzarization on weights (l2)
        :param verbose: print the model
        :param: seed: the seed to set the random  umber generator
        """

        self.lr = lr
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.dout = dout
        self.n_filt = n_filt
        self.opt = OPTIMIZERS[opt]
        self.regularizer = regularizer
        self.verbose = verbose
        self.seed = seed
        self.loss = loss

        # placeholders for training params
        self.nb_epoch = None
        self.batch_size = None
        self.shuffle = True
        self.iterations = None
        self.val_lim = None
        self.n_set = None
        self.early_term = None

        # empty lists for val and acc
        self.test_score = []
        self.test_accuracy = []
        self.train_score = []
        self.train_accuracy = []
        self.dm_errors = []
        self.history = []

        # X vec for DM
        self.X = None
        self.init_weights = None
        self.weights_out = None

        # the keras model
        self.model = None

    def initialize_weights(self, seed=None):
        """
        re-init the model with a new seed
        :param seed: seed for random gen
        """
        if seed is not None:
            self.seed = seed

        self.create_model()

    def get_weights(self):
        """
        gets the current weights from the model
        :return: list of numpy arrays
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        sets the weights of the model
        :param weights: list of numpy arrays
        """
        self.model.set_weights(weights)

    def fit_dm(self, X_train, Y_train, X_test, Y_test, batch_size=256, nb_epoch=10, iterations=10, val_lim=.99, n_set=3,
               early_term=True):
        """
        training weights using the difference map
        :param X_train: array of training data
        :param Y_train: array of training labels
        :param X_test: array of testing data for model evaluation
        :param Y_test: array of testing labels for model evaluation
        :param n_set: the number of sets to split the train data into
        :param iterations: the number of iterations to run the dm training for
        :param val_lim: the validation limit that the projetion onto the training data will be terminated at
        :param nb_epoch: the number of epochs used in the projection.  if val_lim is reached first it will terminate early
        :param batch_size: the batch size used in the projection
        :param early_term: terminate the projection when val_lim reached on training data even if iterations not completed?
        """

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.iterations = iterations
        self.val_lim = val_lim
        self.n_set = n_set
        self.init_weights = copy.deepcopy(self.get_weights())
        self.early_term = early_term

        # see core.dm_train for further explanation of some params
        self.model, self.weights_out, self.test_score, self.test_accuracy, self.train_score, self.train_accuracy, self.dm_errors = core.dm_train(
            self.model, X_train, Y_train, X=self.X, X_test=X_test, Y_test=Y_test, n_set=self.n_set, rand_starts=False,
            switch_projections=True, iterations=self.iterations, val_lim=val_lim, nb_epoch=self.nb_epoch, use_dm=True, seed=self.seed,
            model_func=self.create_model, init_weights=copy.deepcopy(self.init_weights), batch_size=self.batch_size,
            reset_momentum=False, return_average=False, average_start=20, beta=1.0, early_term=self.early_term)

        return self.model, self.weights_out, self.test_score, self.test_accuracy, self.train_score, self.train_accuracy, self.dm_errors


    def fit(self, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, shuffle=True):
        """
        training weights using regular sgd
        :param X_train: array of training data
        :param Y_train: array of training labels
        :param X_test: array of testing data for model evaluation
        :param Y_test: array of testing labels for model evaluation
        :param nb_epoch: the number of epochs used in the projection.  if val_lim is reached first it will terminate early
        :param batch_size: the batch size used in the projection
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        self.history = self.model.fit(X_train, Y_train,
                         self.batch_size, nb_epoch=self.nb_epoch, shuffle=self.shuffle, verbose=1,
                          validation_data=(X_test, Y_test))

        self.weights_out = copy.deepcopy(self.model.get_weights())

        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, verbose=0):
        return self.model.evaluate(x, y, verbose=0)

    def create_model(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        model = Sequential()

        input_shape = (self.img_channels, self.img_rows, self.img_cols)
        model.add(Convolution2D(self.n_filt, 3, 3,
                                border_mode='valid',
                                input_shape=input_shape, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())

        if self.regularizer is None:
            model.add(Dense(self.n_l1))
        else:
            model.add(Dense(self.n_l1, W_regularizer=l2(self.regularizer)))

        model.add(Activation('relu'))
        if self.dout != 0:
            model.add(Dropout(self.dout))

        if self.n_l2 is not None:
            if self.regularizer is None:
                model.add(Dense(self.n_l2))
            else:
                model.add(Dense(self.n_l2, W_regularizer=l2(self.regularizer)))

        model.add(Activation('relu'))

        if self.dout != 0:
            model.add(Dropout(self.dout))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        if self.verbose:
            model.summary()

        model.compile(loss=self.loss,
                      optimizer=self.opt(lr=self.lr),
                      metrics=['accuracy'])

        self.model = model


    def save_model(self, m_name):
        """
        Save keras model to json and weights to h5.
        :param m_name: the save name for the model
        """
        json_string = self.model.to_json()
        open(m_name + '.json', 'w').write(json_string)
        self.model.save_weights(m_name + '.h5')


    def load_model(self, m_name):
        """
        Load keras model from json and weights h5.
        :param m_name: the save name for the model
        """

        self.model = model_from_json(open(m_name + '.json').read())
        self.model.load_weights(m_name + '.h5')
        self.model.compile(loss=self.loss,
                      optimizer=self.opt(lr=self.lr),
                      metrics=['accuracy'])




if __name__ == "__main__":

    nb_epoch = 10
    batch_size = 256
    k_folds = 5


    # get some data
    X_train, Y_train, _, _ = core.get_data(60000, 10000, 10)

    # split training into folds
    X_folds = np.array_split(X_train, k_folds)
    Y_folds = np.array_split(Y_train, k_folds)

    # create the model
    Model = ModelDM(lr=.001, img_rows=28, img_cols=28,img_channels=1, n_l1=16, n_l2=None, dout=0, n_filt=8, opt='Adam',
                 regularizer=None, verbose=True, seed=1338)

    # create the model
    Model.create_model()
    weights_1 = Model.get_weights()

    # re-init
    Model.initialize_weights()
    weights_2 = Model.get_weights()

    # fit normally
    # We use 'list' to copy, in order to 'pop' later on
    k = 1
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    Y_train = list(Y_folds)
    Y_test = Y_train.pop(k)
    Y_train = np.concatenate(Y_train)

    # fit regular
    Model.fit(X_train, Y_train, X_test, Y_test, batch_size, nb_epoch)
    score_reg = Model.evaluate(X_test, Y_test)

    # fit dm
    Model.initialize_weights()
    _, weights_out, test_loss, test_accuracy, train_loss, train_accuracy, dm_errors = Model.fit_dm(X_train,
                                    Y_train, X_test, Y_test, batch_size, nb_epoch, iterations=10,
                                    val_lim=.99, n_set=3, early_term=False)

    # save a model (no suffix, added when saving)
    Model.save_model("/a_model")