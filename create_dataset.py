# Import necessary libraries
import torch
import torchaudio  # For audio processing
import pywt  # For wavelet transformations
from torch.utils.data import Dataset
import pandas as pd
import os

# Define a custom Dataset class for ship audio data
class ShipsEarDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, transformation, target_sample_rate, num_samples,decompose=False):
        """
        Initialize the dataset with the annotation file, audio directory,
        transformation, target sample rate, and number of samples.
        """
        # Read annotations from CSV file
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir  # Directory containing audio samples
        self.transformation = transformation  # Transformations to apply to audio
        self.target_sample_rate = target_sample_rate  # Desired sample rate
        self.num_samples = num_samples  # Target number of samples in each audio file
        self.decompose=decompose

    def __len__(self):
        # Return the number of audio samples in the dataset
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Retrieve an audio sample and its corresponding label by index.
        Applies necessary preprocessing steps including resampling, padding,
        and wavelet-based transformation.
        """
        # Get audio sample path and label
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        
        # Load audio sample
        signal, sr = torchaudio.load(audio_sample_path)
        
        # Apply preprocessing steps
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        # Apply wavelet-based preprocessing to the signal
        signal = self._apply_wavelet_transform(signal)
        
        # Apply additional transformations
        signal = self.transformation(signal)
        
        return signal, label  # Return processed signal and its label
    
    def _wavelet_transform(self, signal, wavelet='db1'):
        """
        Perform wavelet decomposition on the signal.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=None)
        return coeffs
    
    def _thresholding(self, coeffs, threshold):
        """
        Apply thresholding to wavelet coefficients for noise reduction.
        """
        return [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    def _reconstruct_signal(self, coeffs, wavelet='db1'):
        """
        Reconstruct the signal from wavelet coefficients.
        """
        return pywt.waverec(coeffs, wavelet)
    
    def _extract_signal(self, signal, wavelet='db1', threshold=0.3):
        """
        Perform wavelet decomposition, thresholding, and reconstruction.
        """
        coeffs = self._wavelet_transform(signal, wavelet)
        coeffs = self._thresholding(coeffs, threshold)
        return self._reconstruct_signal(coeffs, wavelet)
    
    def _apply_wavelet_transform(self, signal):
        """
        Apply wavelet transform to reduce noise and enhance the signal.
        """
        signal = signal.numpy()  # Convert tensor to numpy array
        if signal.ndim == 2:
            signal = signal[0]  # Flatten if extra dimension exists
        if self.decompose:
            signal = self._extract_signal(signal)  # Perform extraction
        spect = torch.tensor(signal, dtype=torch.float32)  # Convert back to tensor
        return spect.unsqueeze(0)  # Add channel dimension for compatibility
    
    def _cut_if_necessary(self, signal):
        """
        Truncate the signal if it exceeds the target number of samples.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Pad the signal with zeros if it's shorter than the target number of samples.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        """
        Resample the signal to the target sample rate if it's different.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        """
        Convert the signal to mono if it's stereo.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        """
        Get the file path for the audio sample based on index.
        """
        fold = f"{self.annotations.iloc[index, 1]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        """
        Retrieve the label for the audio sample based on index.
        """
        return self.annotations.iloc[index, 3]
