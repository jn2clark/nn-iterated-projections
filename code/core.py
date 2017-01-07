# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:07:22 2016

@author: jesseclark
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist,cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l1,l2

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
import copy
from collections import defaultdict
import time

import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def recursive_dict():
    return defaultdict(recursive_dict)


def subtract_weights(X_1, X_2):
    """
    subtracts X_1 from X_2
    :param X_1: list of weights from .get_weights()
    :param X_2: list of weights from .get_weights()
    :return: list of weights
    """
    return average_weights(X_1, X_2, w1=1.0, w2=-1.0)


def add_weights(X_1, X_2):
    """
    adds X_1 from X_2
    :param X_1: list of weights from .get_weights()
    :param X_2: list of weights from .get_weights()
    :return: list of weights
    """
    return average_weights(X_1, X_2, w1=1.0, w2=1.0)


def multiply_weights(X_1, mult):
    """
    multiplies X_1 by scalar mult
    :param X_1: list of weights from .get_weights()
    :param mult: scalar to multiple weights by
    :return: list of weights
    """
    return average_weights(X_1, X_1, w1=mult, w2=0)


def add_X(X_1, X_2):
    """
    Adds the dict of weights X_1 to X_2
    :param X_1: dict of weights where each key are weights from .get_weights() method
    :param X_2: dict of weights where each key are weights from .get_weights() method
    :return: dict of weights
    """
    return {ind:add_weights(copy.deepcopy(X_1[ind]), copy.deepcopy(X_2[ind])) for ind in X_1.keys()}


def subtract_X(X_1, X_2):
    """
    Subtract the dict of weights X_1 to X_2
    :param X_1: dict of weights where each key are weights from .get_weights() method
    :param X_2: dict of weights where each key are weights from .get_weights() method
    :return: dict of weights
    """
    return {ind:subtract_weights(copy.deepcopy(X_1[ind]), copy.deepcopy(X_2[ind])) for ind in X_1.keys()}


def multiply_X(X, mult):
    """
    Scalar multiplication of the dict of weights X
    :param X_1: dict of weights where each key are weights from .get_weights() method
    :param mult: real-valued scalar for multiplying the weights
    :return: dict of weights
    """
    return {ind:multiply_weights(copy.deepcopy(X[ind]), mult) for ind in X.keys()}


def P_avg(X, l1_lambda=None, p=None):
    """
    Average projection operator
    :param X dict of weights where each key are weights from .get_weights() method.
    :param l1_lambda, regularization scalar for the soft-thresholding operation.
    :param p % to keep of non-average, drop-out analogy
    :return: dict of weights where each entry is the average of all entries.
    """
    weights_avg = average_all_weights([copy.deepcopy(X[ind]) for ind in X.keys()])

    if l1_lambda is not None:
        weights_avg = soft_threshold_weights(weights_avg, l1_lambda)

    if p is None:
        return {ind:weights_avg for ind in X.keys()}
    else:
        mask = create_mask(weights_avg,p)
        return {ind:add_weights_mask(X[ind], weights_avg, mask) for ind in X.keys()}



def P_data(X, X_trains, Y_trains, model, nb_epoch, early_term=True, val_max=0.9, batch_size=1024, reset_momentum=False):
    """
    do the projection onto the training data set
    :param X: the weights as a dit.  each entry are the weights for that set
    :param X_trains: a dict of training data.  each entry is the training data for that set
    :param Y_trains: a dict of training labels.  each entry is the training data for that set
    :param model: the model to use for the projetions. sets the lr and optimizer to used
    :param nb_epoch: the number of epochs to project with.  if early_term=True then it will terminate at val_max
    :param early_term: terminate projection early if val_max reached on the training data
    :param val_max: the %val accuracy at which to terminate the projection
    :param batch_size: the batch size used for the porjection
    :return: dict of weights that have been projected onto their respective sets of data
    """

    n_set = len(X)
    X_out = []

    # loop through each X_i and 'project' onto its set
    for qq in range(n_set):
        # set the model with the weights
        model.set_weights(copy.deepcopy(X[qq]))
        if reset_momentum:
            model.optimizer.set_weights([])

        X_i,val_1 = projection(model, X_trains[qq],
                    Y_trains[qq], nb_epoch, early_term=early_term, val_max=val_max, batch_size=batch_size)
        X_out.append(copy.deepcopy(X_i))

    return {ind:X_out[ind] for ind in X.keys()}


def average_all_weights(weights):
    """
    Average a list of weights
    :param weights: a list of weights
    :return: the average of the weights
    """

    # get the number of weights
    nw = len(weights)

    if nw == 0:
        raise ValueError("Length 0 weights")

    # scale
    fact = 1.0/nw

    weights_out = create_empty_weights(weights[0])
    # weights is a list of weights
    for ii in range(nw):
        # go through each weight and add to
        for jj in range(len(weights[ii])):
            weights_out[jj] += fact*weights[ii][jj]

    return weights_out


def average_weights(weights_1, weights_2, w1=0.5, w2=0.5):
    """
    Combine two weights using addition/subtraction
    :param weights_1: weights 1
    :param weights_2: weights 2
    :param w1: the scalar value to apply to weights 1
    :param w2: the scalar value to apply to weights 2
    :return: the weighted sum of weights 1 and 2
    """

    # average weights
    weights_out = []
    for ind in range(len(weights_1)):
        weights_out.append(w1*weights_1[ind] + w2*weights_2[ind])

    return weights_out


def weights_error(weights_1, weights_2):
    """
    Get the l2 norm of weights_1 - weights_2
    :param weights_1:
    :param weights_2:
    :return: scalar error
    """
    # get the error in weights
    weights = average_weights(weights_1, weights_2, w1=1.0, w2=-1.0)

    # get the total to normalize against
    weights_tot = 0
    weights_er = 0
    for ind in range(len(weights_1)):
        weights_tot += np.sum(weights_2[ind]**2)
        weights_er += np.sum(weights[ind]**2)

    return weights_er


def create_empty_weights(weights_in):
    """
    Create a set of 0 weights with same size as weights_in
    :param weights_in: weights that provide size and shape for empty weights
    :return: empty weights (i.e. all 0's)
    """
    weights_out = []
    for ind in range(len(weights_in)):
        weights_out.append(weights_in[ind]*0)

    return weights_out


def create_model(lr=.005, img_rows=28, img_cols=28,img_channels=1, n_l1=16, n_l2=None, dout=.2, n_filt=8, opt=Adam,
                 regularizer=None, verbose=True):
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
    :return: keras model
    """
    model = Sequential()

    input_shape = (img_channels, img_rows, img_cols)
    model.add(Convolution2D(n_filt, 3, 3,
                                border_mode='valid',
                                input_shape=input_shape, subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())

    if regularizer is None:
        model.add(Dense(n_l1))
    else:
        model.add(Dense(n_l1, W_regularizer=l2(regularizer)))

    model.add(Activation('relu'))
    if dout != 0:
        model.add(Dropout(dout))

    if n_l2 is not None:
        if regularizer is None:
            model.add(Dense(n_l2))
        else:
            model.add(Dense(n_l2, W_regularizer=l2(regularizer)))

    model.add(Activation('relu'))

    if dout != 0:
        model.add(Dropout(dout))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    if verbose:
        model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt(lr=lr),
                  metrics=['accuracy'])

    return model



def projection(model, X_train, Y_train, nb_epoch, X_test=[], Y_test=[],
               val=0, early_term=True, extra_its=20, val_max=0.99, batch_size=None):
    """
    :param model: the model used for training on the data
    :param X_train: the training data
    :param Y_train: the training labels
    :param nb_epoch: the number of iterations for the projection
    :param X_test: optional test data
    :param Y_test: optional test labels
    :param val: the init validation value
    :param early_term: terminate before nb_epochs completed if training val reaches val_max
    :param extra_its: additional epochs to perform even if terminated
    :param val_max: the max validation to reach before terminating
    :param batch_size: batch size to use
    :return: weights and final train validation acc
    """

    if batch_size is None:
        batch_size = len(X_train)

    if not early_term:
        history = model.fit(X_train, Y_train,
                    batch_size, nb_epoch=nb_epoch,
                    verbose=0)
        val = history.history['acc'][-1]

    else:
        cc = 0
        loss_all = [100]
        while cc <= nb_epoch and val <= val_max:
            if len(X_test) == 0:
                history = model.fit(X_train, Y_train,
                    batch_size, nb_epoch=1,
                    verbose=0)
                val = history.history['acc'][-1]
            else:
                history = model.fit(X_train, Y_train,
                    batch_size, nb_epoch=1,
                    verbose=0, validation_data=(X_test, Y_test))
                # terminate based on val
                val = history.history['val_acc'][-1]
                loss_all.append(history.history['val_loss'][-1])

                if loss_all[-1] > loss_all[-2]:
                    #print(loss_all)
                    break
                    #print('!')

            cc += 1

        if extra_its > 0:
            for qq in range(extra_its):
                history = model.fit(X_train, Y_train,
                    batch_size, nb_epoch=1,
                    verbose=0)#, validation_data=(X_test, Y_test))

                val = history.history['acc'][-1]

    return model.get_weights(), val


def get_data(n_train, n_test, nb_classes):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_rows, img_cols = (28,28)
    # make some that are the same
    X_digits = {ind:X_train[np.where(y_train == ind)] for ind in range(10) }

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train[:n_train]
    X_test = X_test[:n_test]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train[:n_train], nb_classes)
    Y_test = np_utils.to_categorical(y_test[:n_test], nb_classes)

    return X_train, Y_train, X_test, Y_test


def get_cifar(nb_classes=10):
    # input image dimensions
    # img_rows, img_cols = 32, 32
    # # The CIFAR10 images are RGB.
    # img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test



def add_weights_mask(weights_1, weights_2, mask):
    # weights_1 orig, weights 2 other
    weights_out = []

    for ind in range(len(weights_1)):
        weights_out.append(weights_1[ind]*mask[ind]+(1-mask[ind])*weights_2[ind])

    return weights_out


def create_mask(weights, p=0.5):
    mask_out = []
    for layer in weights:
        mask_out.append((np.random.rand(*layer.shape) > p).astype(int))
    return mask_out


def soft_threshold_weights(weights_in, thresh):
    """
    l1 soft-threshold operation
    :param weights_in: set of weights
    :param thresh: the threshold
    :return: a set of weights
    """

    weights_out = create_empty_weights(weights_in)

    for ind, weight in enumerate(copy.deepcopy(weights_in)):
        # |x| < t = 0
        weight[np.where(np.abs(weight) < thresh)] = 0
        # x >= t -> x-t
        weight[np.where(weight >= thresh)] -= thresh
        # x <= -t -> x+t
        weight[np.where(weight <= -thresh)] += thresh
        weights_out[ind] = weight

    return weights_out


def dm_train(model, X_train, Y_train, X=None, X_test=None, Y_test=None, n_set=2, rand_starts=False, switch_projections=True,
             iterations=20, val_lim=0.99, nb_epoch=1200, use_dm=True, seed=1338, model_func=create_model,
             init_weights=None, batch_size=None, verbose=True, return_average=False, average_start=50, beta=1.0,
             thresh=None, reset_momentum=False, early_term=True):
    """
    training weights using the difference map
    :param model: the model use for the projection step
    :param X_train: array of training data
    :param Y_train: array of training labels
    :param X: dict of weights that comprise the X vector
    :param X_test: array of testing data for model evaluation
    :param Y_test: array of testing labels for model evaluation
    :param n_set: the number of sets to split the train data into
    :param rand_starts: generate random start wieghts? will usee model_func to generate new weights
    :param switch_projections: switch the order of the projections? Pa[2Pb - I] - I vs Pb[2Pa - I] - I
    :param iterations: the number of iterations to run the dm training for
    :param val_lim: the validation limit that the projetion onto the training data will be terminated at
    :param nb_epoch: the number of epochs used in the projection.  if val_lim is reached first it will terminate early
    :param use_dm: use dm update? can also use alternating projetions or any other variant if you want (RAAR, HPR, ASR, HIO)
    :param seed: the random seed if you are generating random weights
    :param model_func: the model function used for generating weights if rand_Start = True
    :param init_weights: the initial weights to use. used if X is None and rand_starts=False
    :param batch_size: the batch size used in the projection
    :param verbose: verbosity

    :return: too much
    """

    if verbose:
        LOGGER.info((n_set, iterations, val_lim, nb_epoch, batch_size))

    algo = {True:'DM', False:'ER'}

    # split up the data into constraint sets
    LOGGER.info("Splitting up data...")
    X_trains = {ind:X_tra for ind,X_tra in enumerate(np.array_split(X_train, n_set)) }
    Y_trains = {ind:Y_tra for ind,Y_tra in enumerate(np.array_split(Y_train, n_set)) }

    # init some lists for storing vals
    scores = []
    scores_train = []
    dm_errors = []

    # create the X vector from the concat of all the weights
    # set random seed for consistency
    if seed is not None:
        LOGGER.info("Setting random seed to {}".format(seed))
        np.random.seed(seed)

    if X is None:
        if rand_starts:
            LOGGER.info("Generating random weights...")
            X = {ind:model_func().get_weights() for ind in range(n_set)}
        else:
            if init_weights is None:
                raise TypeError("init_weights not specified. expected weights from .get_weights() method.")
            LOGGER.info("Setting weights to ones provided...")
            X = {ind:copy.deepcopy(init_weights) for ind in range(n_set)}
    else:
        LOGGER.info("Using existing X")

    # not actually sure this works - need to check
    if reset_momentum:
        LOGGER.info("Resetting momentum after each projection...")

    # get the average of all the weights
    if switch_projections:
        X_A = P_data(X, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size,
                     reset_momentum=reset_momentum)
    else:
        X_A = P_avg(X, thresh)

    LOGGER.info("Starting iterations...")

    # average_weights once converged?
    average_converged_weights = []

    for ii in range(iterations):

        # time the its, good for sanity checking batch size and earl term stuff
        t0 = time.time()

        # x' = x + PA(2PB(x) - x) - PB(x)
        if use_dm:
            # reflection  2PB(x) - x
            X_R = subtract_X(multiply_X(X_A, 2.0), X)

            # project: PA(2PB(x) - x)
            if switch_projections:
                X_P = P_avg(X_R, thresh)
            else:
                X_P = P_data(X_R, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim,
                             batch_size=batch_size, reset_momentum=reset_momentum)

            # PA(2PB(x) - x) - PB(x)
            X_D = subtract_X(X_P, X_A)

            # update X, can add a relaxation here as well if we want. just change the value to < 1
            X = add_X(X, multiply_X(X_D, beta))

            # get error from the previous weights and the new weights
            # approximate using one set only
            dm_error = weights_error(X_A[0], P_avg(X, thresh)[0])
            dm_errors.append(dm_error)

            # get the projection of all the weights for the next iteration
            if switch_projections:
                X_A = P_data(X, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size,
                            reset_momentum=reset_momentum)
            else:
                X_A = P_avg(X, thresh)

            if return_average:
                if ii > average_start:
                    LOGGER.info("Collecting weights for averaging...")
                    average_converged_weights.append(P_avg(X,thresh)[0])

        else:
            # alternating projections
            # x' = PA(PB(x))

            # alternating relaxed proj here if you want to play with it
            # X = add_X(multiply_X(X_A,.25),multiply_X(P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=True, val_max=val_lim, batch_size=batch_size),.75))
            # X = subtract_X(multiply_X(P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=True, val_max=val_lim, batch_size=batch_size),2.0),X_A)
            X = P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size,
                       reset_momentum=reset_momentum)

            # get error from the previous weights and the new weights
            dm_error = weights_error(X_A[0], P_avg(X,thresh)[0])
            dm_errors.append(dm_error)

            X_A = P_avg(X, thresh)


        # set the model to have weights for inference
        model.set_weights(P_avg(X, thresh)[0])
        score = model.evaluate(X_test, Y_test, verbose=0)
        scores.append(score)

        scores_train.append(model.evaluate(X_train[:], Y_train[:], verbose=0))

        if verbose:
            LOGGER.info((n_set, score, ii, scores_train[-1], dm_error, algo[use_dm], time.time()-t0))

    # which way are we outputting the weights?
    weights_out = P_avg(X, thresh)[0]
    if return_average:
        if len(average_converged_weights) != 0:
            LOGGER.info("Averaging over converged weights...")
            weights_out = average_all_weights(average_converged_weights)


    # do some final evaluations
    model.set_weights(weights_out)
    score = model.evaluate(X_test, Y_test, verbose=0)
    LOGGER.info('Test score: {}'.format(score[0]) )
    LOGGER.info('Test accuracy: {}'.format(score[1]) )

    test_score,test_accuracy = zip(*scores)
    train_score,train_accuracy = zip(*scores_train)

    return model,weights_out,test_score,test_accuracy,train_score,train_accuracy,dm_errors


def dm_train_full(model, X_train, Y_train, X=None, X_test=None, Y_test=None, n_set=2, rand_starts=False, switch_projections=True,
             iterations=20, val_lim=0.99, nb_epoch=1200, use_dm=True, seed=1338, model_func=create_model,
             init_weights=None, batch_size=None, verbose=True, return_average=False, average_start=50, beta=1.0,
                  thresh=None, reset_momentum=False):
    """
    training weights using the difference map
    :param model: the model use for the projection step
    :param X_train: array of training data
    :param Y_train: array of training labels
    :param X: dict of weights that comprise the X vector
    :param X_test: array of testing data for model evaluation
    :param Y_test: array of testing labels for model evaluation
    :param n_set: the number of sets to split the train data into
    :param rand_starts: generate random start wieghts? will usee model_func to generate new weights
    :param switch_projections: switch the order of the projections? Pa[2Pb - I] - I vs Pb[2Pa - I] - I
    :param iterations: the number of iterations to run the dm training for
    :param val_lim: the validation limit that the projetion onto the training data will be terminated at
    :param nb_epoch: the number of epochs used in the projection.  if val_lim is reached first it will terminate early
    :param use_dm: use dm update? can also use alternating projetions or any other variant if you want (RAAR, HPR, ASR, HIO)
    :param seed: the random seed if you are generating random weights
    :param model_func: the model function used for generating weights if rand_Start = True
    :param init_weights: the initial weights to use. used if X is None and rand_starts=False
    :param batch_size: the batch size used in the projection
    :param verbose: verbosity

    :return: too much
    """

    if verbose:
        LOGGER.info((n_set, iterations, val_lim, nb_epoch, batch_size))

    algo = {True:'DM', False:'ER'}

    # split up the data into constraint sets
    LOGGER.info("Splitting up data...")
    X_trains = {ind:X_tra for ind,X_tra in enumerate(np.array_split(X_train, n_set)) }
    Y_trains = {ind:Y_tra for ind,Y_tra in enumerate(np.array_split(Y_train, n_set)) }

    # init some lists for storing vals
    scores = []
    scores_train = []
    dm_errors = []

    # create the X vector from the concat of all the weights
    # set random seed for consistency
    if seed is not None:
        LOGGER.info("Setting random seed to {}".format(seed))
        np.random.seed(seed)

    if X is None:
        if rand_starts:
            LOGGER.info("Generating random weights...")
            X = {ind:model_func().get_weights() for ind in range(n_set)}
        else:
            if init_weights is None:
                raise ValueError("init_weights not specified. expected weights from .get_weights() method.")
            LOGGER.info("Setting weights to ones provided...")
            X = {ind:copy.deepcopy(init_weights) for ind in range(n_set)}
    else:
        LOGGER.info("Using existing X")

    X_A1 = P_data(X, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size)
    X_A2 = P_avg(X, thresh)

    LOGGER.info("Starting iterations...")

    # average weights once converged?
    average_converged_weights = []

    for ii in range(iterations):

        # time the its, good for sanity checking batch size and earl term stuff
        t0 = time.time()

        # x' = x + PA(2PB(x) - x) - PB(x)
        if use_dm:
            #reflection  2PB(x) - x
            # X_R1 = subtract_X(multiply_X(X_A1, 2.0), X)
            #
            # X_R2 = subtract_X(multiply_X(X_A2, 2.0), X)
            fa = subtract_X(X_A1,multiply_X(subtract_X(X_A1, X),beta))
            fb = add_X(X_A2,multiply_X(subtract_X(X_A2, X),beta))

            # project: PA(2PB(x) - x)
            X_P1 = P_avg(fa, thresh)

            if reset_momentum:
                model.optimizer.set_weights([])
            X_P2 = P_data(fb, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size)

            # PA(2PB(x) - x) - PB(x)
            #X_D1 = subtract_X(X_P1, X_A1)
            X_D = subtract_X(X_P2, X_P1)

            #X_D = subtract_X(X_P1, X_P2)

            # update X, can add a relaxation here as well if we want. just change the value to < 1
            X = add_X(X, multiply_X(X_D, beta))

            # get error from the previous weights and the new weights
            dm_error = weights_error(X_A2[0], P_avg(X, thresh)[0])
            dm_errors.append(dm_error)

            # get the projection of all the weights for the next iteration
            if reset_momentum:
                model.optimizer.set_weights([])

            X_A1 = P_data(X, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size)
            X_A2 = P_avg(X, thresh)

            if return_average:
                if ii > average_start:
                    LOGGER.info("Collecting weights for averaging...")
                    average_converged_weights.append(P_avg(X,thresh)[0])

        else:
            # alternating projections
            # x' = PA(PB(x))

            # alternating relaxed proj here if you want to play with it
            # X = add_X(multiply_X(X_A,.25),multiply_X(P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=True, val_max=val_lim, batch_size=batch_size),.75))
            # X = subtract_X(multiply_X(P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=True, val_max=val_lim, batch_size=batch_size),2.0),X_A)
            X = P_data(X_A, X_trains, Y_trains, model, nb_epoch, early_term=early_term, val_max=val_lim, batch_size=batch_size)

            # get error from the previous weights and the new weights
            dm_error = weights_error(X_A[0], P_avg(X,thresh)[0])
            dm_errors.append(dm_error)

            X_A = P_avg(X, thresh)

        # set the model to have weights for inference
        model.set_weights(P_avg(X, thresh)[0])
        score = model.evaluate(X_test, Y_test, verbose=0)
        scores.append(score)

        scores_train.append(model.evaluate(X_train[:], Y_train[:], verbose=0))

        if verbose:
            print(n_set, score, ii, scores_train[-1], dm_error, algo[use_dm], time.time()-t0)

    # which way are we outputting the weights?
    weights_out = P_avg(X, thresh)[0]
    if return_average:
        if len(average_converged_weights) != 0:
            LOGGER.info("Averaging over converged weights...")
            weights_out = average_all_weights(average_converged_weights)


    # do some random stuff before outputting
    model.set_weights(weights_out)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    test_score,test_accuracy = zip(*scores)
    train_score,train_accuracy = zip(*scores_train)

    return model,weights_out,test_score,test_accuracy,train_score,train_accuracy,dm_errors



