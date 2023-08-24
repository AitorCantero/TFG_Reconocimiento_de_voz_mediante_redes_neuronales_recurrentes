
import colorednoise as cn
import numpy as np
import librosa
import random
import ast
import os


def modulate_volume(audio, factor):
    """
    Modulates the volume of an audio waveform.

    Args:
        audio (np.ndarray): Input audio waveform.
        factor (float): Volume modulation factor.

    Returns:
        np.ndarray: Modulated audio waveform.
    """
    amplified_audio = audio * factor
    return amplified_audio


def time_stretch(audio, rate):
    """
    Applies time stretching to an audio waveform.

    Args:
        audio (np.ndarray): Input audio waveform.
        rate (float): Time stretching rate.

    Returns:
        np.ndarray: Stretched audio waveform.
    """
    stretched_audio = librosa.effects.time_stretch(y=audio, rate=rate)
    original_length = len(audio)

    if len(stretched_audio) > original_length:
        # Trim the stretched audio to match the original length
        excess_length = len(stretched_audio) - original_length
        trim_start = excess_length // 2
        trim_end = trim_start + original_length
        stretched_audio = stretched_audio[trim_start:trim_end]

    elif len(stretched_audio) < original_length:
        # Repeat the stretched audio to match the original length
        repetitions = original_length // len(stretched_audio)
        remainder = original_length % len(stretched_audio)
        stretched_audio = np.tile(stretched_audio, repetitions)
        stretched_audio = np.concatenate([stretched_audio, stretched_audio[:remainder]])

    return stretched_audio


def pitch_shift(audio, sample_rate, n_steps):
    """
    Applies pitch shifting to an audio waveform.

    Args:
        audio (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        n_steps (int): Number of steps for pitch shifting.

    Returns:
        np.ndarray: Pitch-shifted audio waveform.
    """
    shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=n_steps)
    return shifted_audio


def merge_noise(audio, sample_rate, noise_factor):
    """
    Merges noise with an audio waveform.

    Args:
        audio (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        noise_factor (float): Noise factor for blending.

    Returns:
        np.ndarray: Noisy audio waveform.
    """
    list_1 = [os.path.normpath(os.path.join('data/noise/free-sound/', path)) for path in os.listdir('data/noise/free-sound/')]
    list_2 = [os.path.normpath(os.path.join('data/noise/sound-bible/', path)) for path in os.listdir('data/noise/sound-bible/')]
    noises = list_1 + list_2

    random_noise, _ = librosa.load(random.choice(noises), sr=sample_rate)
    audio_length = len(audio)
    noise_length = len(random_noise)

    if audio_length < noise_length:
        random_noise = random_noise[:len(audio)]
    elif audio_length > noise_length:
        repetitions = audio_length // len(random_noise)
        remainder = audio_length % len(random_noise)
        random_noise = np.tile(random_noise, repetitions)
        random_noise = np.concatenate([random_noise, random_noise[:remainder]])

    noisy_audio = (1 - noise_factor) * audio + noise_factor * random_noise
    return noisy_audio


def colored_noise(audio, beta, noise_factor):
    """
    Adds colored noise to an audio waveform.

    Args:
        audio (np.ndarray): Input audio waveform.
        beta (float): Powerlaw exponent for colored noise generation.
        noise_factor (float): Noise factor for blending.

    Returns:
        np.ndarray: Noisy audio waveform.
    """
    noise = cn.powerlaw_psd_gaussian(beta, len(audio))
    noise = noise / np.max(noise)
    noisy_audio = (1 - noise_factor) * audio + noise_factor * noise
    return noisy_audio


def spec_augment(original_melspec, freq_masking_max_percentage, time_masking_max_percentage):
    """
    Applies SpecAugment to a mel spectrogram.

    Args:
        original_melspec (np.ndarray): Input mel spectrogram.
        freq_masking_max_percentage (float): Maximum percentage of frequencies to mask.
        time_masking_max_percentage (float): Maximum percentage of time frames to mask.

    Returns:
        np.ndarray: Augmented mel spectrogram.
    """
    augmented_melspec = original_melspec.copy()
    all_frames_num, all_freqs_num = augmented_melspec.shape

    # Frequency masking
    freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
    num_freqs_to_mask = int(freq_percentage * all_freqs_num)
    f0 = int(np.random.uniform(low=0.0, high=(all_freqs_num - num_freqs_to_mask)))
    
    augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

    # Time masking
    time_percentage = random.uniform(0.0, time_masking_max_percentage)
    num_frames_to_mask = int(time_percentage * all_frames_num)
    t0 = int(np.random.uniform(low=0.0, high=(all_frames_num - num_frames_to_mask)))
    
    augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0

    return augmented_melspec


def preprocess_audio_waveform(waveform, sample_rate, functions=[], parameters=[]):
    """
    Preprocesses an audio waveform using a series of waveform-based transformations.

    Args:
        waveform (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        functions (list): List of preprocessing function names.
        parameters (list): List of parameters corresponding to the preprocessing functions.

    Returns:
        np.ndarray: Preprocessed waveform.
    """
    preprocessed_waveform = waveform.copy()
    for f, p in zip(functions, parameters):
        if f == 'modulate_volume':
            preprocessed_waveform = modulate_volume(preprocessed_waveform, float(p))
        elif f == 'time_stretch':
            preprocessed_waveform = time_stretch(preprocessed_waveform, float(p))
        elif f == 'pitch_shift':
            preprocessed_waveform = pitch_shift(preprocessed_waveform, sample_rate, int(p))
        elif f == 'merge_noise':
            preprocessed_waveform = merge_noise(preprocessed_waveform, sample_rate, float(p))
        elif f == 'colored_noise':
            p_tuple = ast.literal_eval(p)
            preprocessed_waveform = colored_noise(preprocessed_waveform, p_tuple[0], p_tuple[1])

    preprocessed_waveform = preprocessed_waveform.reshape(-1, 100)
    return preprocessed_waveform


def preprocess_audio_spectrogram(waveform, sample_rate, functions=[], parameters=[]):
    """
    Preprocesses an audio waveform as a mel spectrogram using a series of transformations.

    Args:
        waveform (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        functions (list): List of preprocessing function names.
        parameters (list): List of parameters corresponding to the preprocessing functions.

    Returns:
        np.ndarray: Preprocessed mel spectrogram.
    """
    later_preprocess = []
    preprocessed_waveform = waveform.copy()
    for f, p in zip(functions, parameters):
        if f == 'time_stretch':
            preprocessed_waveform = time_stretch(preprocessed_waveform, float(p))
        elif f == 'pitch_shift':
            preprocessed_waveform = pitch_shift(preprocessed_waveform, sample_rate, int(p))
        elif f == 'merge_noise':
            preprocessed_waveform = merge_noise(preprocessed_waveform, sample_rate, float(p))
        elif f == 'colored_noise':
            p_tuple = ast.literal_eval(p)
            preprocessed_waveform = colored_noise(preprocessed_waveform, p_tuple[0], p_tuple[1])
        else:
            later_preprocess.append((f, p))
            
    spectrogram = librosa.feature.melspectrogram(y=preprocessed_waveform, sr=sample_rate, window=True)
    normalized_spectrogram = librosa.power_to_db(spectrogram)

    for f, p in later_preprocess:
        if f == 'spec_augment':
            p_tuple = ast.literal_eval(p)
            normalized_spectrogram = spec_augment(normalized_spectrogram, p_tuple[0], p_tuple[1])
    
    return normalized_spectrogram



def preprocess_audio_spectrogram_rot(waveform, sample_rate, functions=[], parameters=[]):
    """
    Preprocesses an audio waveform as a mel spectrogram using a series of transformations.

    Args:
        waveform (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate of the audio.
        functions (list): List of preprocessing function names.
        parameters (list): List of parameters corresponding to the preprocessing functions.

    Returns:
        np.ndarray: Preprocessed mel spectrogram (rotated 90º).
    """
    normalized_spectrogram = preprocess_audio_spectrogram(waveform, sample_rate, functions=[], parameters=[])
    normalized_spectrogram = np.rot90(normalized_spectrogram, k=3, axes=(1, 0))
    
    return normalized_spectrogram