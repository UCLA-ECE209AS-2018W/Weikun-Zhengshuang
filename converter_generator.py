# -*- coding: utf-8 -*-
""" Converter and generator

This module can use for train as the libary. You don't need modify any functions
in the libary. Please see the description in front of each function if you don't
undersand it. Please import this module if you want use this libary in anthor
project. The mould are assumed to already be preprocessed and spliced into
clips and split into a training, testing, and validation set.
These sets are stored as csv files which are later read in in batches with
functions in this module.

################################################################################
# Author: Weikun Han <weikunhan@gmail.com>
# Crate Date: 03/6/2018
# Update:
# Reference: https://github.com/jhetherly/EnglishSpeechUpsampler
################################################################################
"""

import os
import csv
import numpy as np
import librosa

#####################
# CONVERTER FUNCTIONS
#####################

def get_original_noise_pairs(directory, dataset='train'):
    """ Get the pair directory of original and noise .wav file

    The directory is a string representing directory of the csv files
    containing the actual file name pairs dataset is one of "train," "test,"
    or "validation"

    Args:
        param1 (str): directory
        param2 (str): dataset

    Returns:
        list: each element contain input filename pairs directory of .wav file
              [["original filename", "noise filename"]...]

    """
    original_noise_pairs = []
    input_tmp_path = os.path.join(directory, '{}_files.csv'.format(dataset))

    with open(input_tmp_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            original_noise_pairs.append(row)
    return original_noise_pairs

def bitrates_and_waveforms(filename_pair):
    """ Get the bit rates and waveforms if given a single pair of .wav file

    Given a file name pair this function returns the original and noise bit
    rates as well as the original and noise waveforms

    Args:
        param1 (list): filename_pair

    Returns:
        list: the filst element in list is bit rate and seceond is waveform
              [["original bit rates", "noise bit rates"],
               ["original waveforms", "noise waveforms"]]

    """

    # sr is target sampling rate, and ‘None’ uses the native sampling rate
    original_waveform, original_bitrate = librosa.load(filename_pair[0], sr=None)
    noise_waveform, noise_bitrate = librosa.load(filename_pair[1], sr=None)
    return [[original_bitrate, noise_bitrate],
            [original_waveform, noise_waveform]]

def waveforms(filename_pair, mono=True):
    """ Get two waveforms for single pair of original and noise .wav file

    Given a pair of file names in this function, read in both original waveform
    and noise waveform assumes the file name pair is of the form
    ["original", "noise"] mono selects whether to read in mono or stereo
    formatted waveforms

    Args:
        param1 (list): filename_pair
        param2 (str): mono

    Returns:
        list: the filst element in list is original waveform, and second is
               noise waveform
               ["original mono waveforms", "noise mono waveforms"]

    """

    # Determine the channel for the input
    if mono is True:
        channel_value = 1
    else:
        channel_value = 2

    original_waveform, original_bitrate = librosa.load(filename_pair[0],
                                                       sr=None,
                                                       mono=mono)
    noise_waveform, noise_bitrate = librosa.load(filename_pair[1],
                                                 sr=None,
                                                 mono=mono)
    return [original_waveform.reshape((-1, channel_value)),
            noise_waveform.reshape((-1, channel_value))]

#####################
# GENERATOR FUNCTIONS
#####################

def random_batch(batch_size, filename_pairs, mono=True):
    """ Randomly selets batch within list of target file name pair

    This function randomly selects batch_size number of samples from a list
    of file name pairs. This function is use for training

    Args:
        param1 (int): batch_size
        param2 (list): filename_pairs
        param3 (bool): mono

    Returns:
        list: returns lists containing the true waveforms and noise
              waveforms
              [[["original mono waveforms"]...], [["noise mono waveforms"]...]]

    """
    batch_original_waveform = []
    batch_noise_waveform = []

    # Deterine the range to selet batch
    seleted_range = range(len(filename_pairs))

    # Generate a uniform random sample from np.arange(seleted_range) of size
    # batch_size without replacement:
    chosen_pairs = np.random.choice(seleted_range,
                                    size=batch_size,
                                    replace=False)

    for n in chosen_pairs:
        original_waveform, noise_waveform = waveforms(filename_pairs[n], 
                                                      mono=mono)
        batch_original_waveform.append(original_waveform)
        batch_noise_waveform.append(noise_waveform)
    return [batch_original_waveform, batch_noise_waveform]

def next_batch(batch_size, filename_pairs, mono=True):
    """ Sequentially selects batch within list of target file name pair

    This function sequentially selects batch_size number of samples from a
    list of file name pairs. This function is use for validation

    Args:
        param1 (int): batch_size
        param2 (list): filename_pairs
        param3 (bool): mono

    Yields:
        list: returns lists containing the true waveforms and noise
              waveforms
              [[["original mono waveforms"]...], [["noise mono waveforms"]...]]
    """

    # Determine the total range for seleting
    seleted_size = len(filename_pairs)

    for i in range(0, seleted_size, batch_size):
        batch_original_waveform = []
        batch_noise_waveform = []
        end_index = i + batch_size

        # Check seletion is out of bound
        if end_index >= seleted_size:
            chosen_pairs = filename_pairs[i :]
        else:
            chosen_pairs = filename_pairs[i : end_index]

        for n in chosen_pairs:
            original_waveform, noise_waveform = waveforms(n,
                                                          mono=mono)
            batch_original_waveform.append(original_waveform)
            batch_noise_waveform.append(noise_waveform)

        # Use generators to keep create random batch
        yield [batch_original_waveform, batch_noise_waveform]

