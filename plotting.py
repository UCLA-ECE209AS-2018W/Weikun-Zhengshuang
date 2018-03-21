# -*- coding: utf-8 -*-
""" Plotting

This module can use for plotting. You don't need modify any functions in the 
libary. Please see the description in front of each function if you don't 
undersand it. Please import this module if you want use this libary
in anthor project. 

################################################################################
# Author: Weikun Han <weikunhan@gmail.com>
# Crate Date: 03/16/2018
# Update:
# Reference: https://github.com/jhetherly/EnglishSpeechUpsampler
################################################################################
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa import display
from matplotlib.colors import LogNorm

def compute_snr(origianl_waveform, target_waveform):
    """ Compare snr between the original and target audio

    Signal-to-noise ratio (abbreviated SNR or S/N) is a measure used in science 
    and engineering that compares the level of a desired signal to the level of 
    background noise.
    
    Args:
        param1 (list): origianl_waveform
        param2 (list): target_waveform

    Returns:
        float: compute SNR for waveform plots

    """
    return 10. * np.log10(np.sqrt(np.sum(origianl_waveform**2)) / np.sqrt(
        np.sum((origianl_waveform - target_waveform)**2)))

def compute_lsd(original_spectrogram, target_spectrogram):
    """ Compare lsd between the original and target audio

    The log-spectral distance (LSD), also referred to as log-spectral distortion, 
    is a distance measure (expressed in dB) between two spectra

    Args:
        param1 (list): origianl_spectrogram
        param2 (list): target_spectrogram

    Returns:
        float: compute lsd for spectrogram plots

    """
    original_log = np.log10(np.abs(original_spectrogram)**2)
    target_log = np.log10(np.abs(target_spectrogram)**2)
    original_target_squared = (original_log - target_log)**2
    target_lsd = np.mean(np.sqrt(np.mean(original_target_squared, axis=0)))
    
    return target_lsd

def plot_figures(original_spectrogram, 
                 noise_spectrogram, 
                 recovery_spectrogram,
                 original_waveform, 
                 noise_waveform, 
                 recovery_waveform,
                 original_bitrate, 
                 noise_bitrate, 
                 recovery_bitrate, 
                 save_path, 
                 fft_window_size):
    """ Plot spectorgram and waveform

    This function can compare difference between the original and recovery audio

    Args:
        param1 (list): original_spectrogram
        param2 (list): noise_spectrogram
        param3 (list): recovery_spectrogram
        param4 (list): original_waveform
        param5 (list): noise_waveform
        param6 (list): recovery_waveform
        param7 (list): original_bitrate
        param8 (list): noise_bitrate
        param9 (list): recovery_bitrate
        param10 (str): save_path
        param11 (int): fft_window_size

    Returns:
        float: compute SNR for waveform plots

    """
    max_frame = 100
    # cmap = 'nipy_spectral'
    cmap = 'rainbow_r'
    # cmap = 'gist_rainbow'
    # cmap = 'viridis'
    # cmap = 'inferno_r'
    # cmap = 'magma_r'
    # cmap = 'plasma_r'
    plt.figure(figsize=(20, 8))

    # Check input audio bit rate is equal or not
    if not (original_bitrate == noise_bitrate == recovery_bitrate):
        print('Warning: time axis on waveform plots will be meaningless')

    # Compute dB-scale magnitudes
    original_dB = librosa.amplitude_to_db(original_spectrogram, ref=np.max)
    noise_dB = librosa.amplitude_to_db(noise_spectrogram, ref=np.max)
    recovery_dB = librosa.amplitude_to_db(recovery_spectrogram, ref=np.max)

    # Compute lsd for spectrogram plots
    noise_lsd = compute_lsd(original_spectrogram, noise_spectrogram)
    recovery_lsd = compute_lsd(original_spectrogram, recovery_spectrogram)

    # Spectrogram plots
    # Original audio
    ax = plt.subplot(2, 3, 1)
    plt.title('Original spectrum (dB)')
    fig = display.specshow(original_dB,
                           sr=original_bitrate, 
                           y_axis='hz', 
                           x_axis='time',
                           hop_length=fft_window_size / 4, 
                           cmap=cmap,
                           edgecolors='face')
    fig.axes.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    
    # Noise audio
    ax = plt.subplot(2, 3, 2)
    plt.title('Noise spectrum (dB)')
    fig = display.specshow(noise_dB,
                           sr=noise_bitrate, 
                           y_axis='hz', 
                           x_axis='time',
                           hop_length=fft_window_size / 4,
                           cmap=cmap,
                           edgecolors='face')
    fig.axes.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    ax.text(0.05, 0.1, r'LSD={:.2}'.format(noise_lsd), color='red', 
            fontsize=12, transform=ax.transAxes, backgroundcolor='white')
    
    # Recovery audio
    ax = plt.subplot(2, 3, 3)
    plt.title('Recovery spectrum (dB)')
    fig = display.specshow(recovery_dB,
                           sr=recovery_bitrate, 
                           y_axis='hz', 
                           x_axis='time',
                           hop_length=fft_window_size / 4,
                           cmap=cmap,
                           edgecolors='face')
    fig.axes.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    ax.text(0.05, 0.1, r'LSD={:.2}'.format(recovery_lsd), color='red', 
            fontsize=12, transform=ax.transAxes, backgroundcolor='white')

    # Compute SNR for waveform plots
    noise_snr = compute_snr(original_waveform, noise_waveform)
    recovery_snr = compute_snr(original_waveform, recovery_waveform)

    # Waveform plots
    # Original audio
    ax = plt.subplot(2, 3, 4)
    original_time = np.arange(max_frame, dtype=np.float) / float(original_bitrate)
    plt.title('Original waveform (16 kbps)')
    fig = plt.plot(original_time, original_waveform[:max_frame])
    ax.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    # Noise audio
    ax = plt.subplot(2, 3, 5)
    noise_time = np.arange(max_frame, dtype=np.float) / float(noise_bitrate)
    plt.title('Noise waveform (16 kbps)')
    fig = plt.plot(noise_time, noise_waveform[:max_frame])
    ax.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    ax.text(0.05, 0.1, r'SNR={:.1f}'.format(noise_snr), color='red', 
            fontsize=12, transform=ax.transAxes, backgroundcolor='white')
    
    # Recovery audio
    ax = plt.subplot(2, 3, 6)
    recovery_time = np.arange(max_frame, dtype=np.float) / float(recovery_bitrate)
    plt.title('Recovery waveform (16 kbps)')
    fig = plt.plot(recovery_time, recovery_waveform[:max_frame])
    ax.set_xticklabels([])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    ax.text(0.05, 0.1, r'SNR={:.1f}'.format(recovery_snr), color='red', 
            fontsize=12, transform=ax.transAxes, backgroundcolor='white')

    # Save plot into target location
    plt.tight_layout()
    plt.savefig(save_path)

def read_audio_spectrum(x, fft_window_size):
    """
    Reads wav file and produces spectrum
    reference: 
        https://librosa.github.io/librosa/generated/librosa.core.stft.html
    """
    return librosa.core.stft(x, n_fft=fft_window_size)
    
def test_best_result(origianl_waveform):
    """
    Audio Super Resolution Using Neural Networks
    reference: 
        https://github.com/kuleshov/audio-super-res
    """
    origianl_waveform = origianl_waveform.flatten()
    recovery_waveform = []
    audio_length = len(origianl_waveform)
    noise = np.random.random_sample((audio_length,))
    noise_list = [x / 100 for x in noise]
    noise_count = 0
    
    for n in origianl_waveform:
        difference = n - noise_list[noise_count]
        recovery_waveform.append(difference)
        noise_count += 1
    
    return np.asarray(recovery_waveform)



