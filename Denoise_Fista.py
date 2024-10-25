# Import necessary libraries
import numpy as np          # For numerical computations
import pandas as pd         # For data manipulation
import pywt                 # For wavelet transformations
import os                   # For directory and file handling
import torchaudio           # For audio processing in PyTorch
import torch                # PyTorch for tensor operations
from Fista import fista     # Import FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) from a custom module

""" 
Function to compute the wavelet coefficients of an audio signal.
It performs wavelet decomposition on the input signal to break it down into multiple frequency bands.
"""
def Coeff_signal(source_signal_path, wavelet):
    # Load the audio signal and its sample rate
    signal, sr = torchaudio.load(source_signal_path)
    signal = signal.squeeze(0)  # Remove unnecessary dimensions
    signal = signal.numpy()     # Convert PyTorch tensor to numpy array for compatibility with pywt

    # Normalize the signal to ensure uniform scaling
    signal = signal / np.max(np.abs(signal))
    
    # Perform wavelet decomposition on the signal
    coeffs = pywt.wavedec(signal, wavelet, level=4)
    
    # Calculate the length of each wavelet coefficient
    coeffs_length = [len(coeff) for coeff in coeffs]
    
    # Flatten the coefficients into a single list for easier processing
    concat_coeffs = [ele for coeff in coeffs for ele in coeff]
    
    # Return flattened coefficients and their respective lengths
    return (concat_coeffs, coeffs_length)

"""
Function to reconstruct the denoised wavelet coefficients into their original structure.
This is based on the optimized coefficients after FISTA denoising.
"""
def coeff_denoised(coeffs_length, opt_coeffs):
    reconstruct_coeffs = []
    start = 0
    # Iterate through each coefficient length to partition optimized coefficients
    for length in coeffs_length:
        end = start + length
        reconstruct_coeffs.append(np.array(opt_coeffs[start:end]))
        start = end
    return reconstruct_coeffs

"""
Function to reconstruct an audio signal from the modified wavelet coefficients.
This function performs the inverse wavelet transformation to recreate the time-domain signal.
"""
def Reconstruct_audio(coeffs, wavelet):
    return pywt.waverec(coeffs, wavelet)

"""
Main function to denoise audio using FISTA optimization.
Processes all audio files in a specified directory, performs denoising using FISTA, and saves the output.
"""
def Denoise_Fista(lambda_, wavelet, audio_path, target_path):
    # Loop through each folder in the audio directory
    for folder in os.listdir(audio_path):
        # Set up paths for source and target directories for each folder
        source_audio_path = os.path.join(audio_path, folder)
        target_audio_path = os.path.join(target_path, folder)

        # Process each audio file in the current folder
        for file in os.listdir(source_audio_path):
            source_file_path = os.path.join(source_audio_path, file)  # Source file path
            target_file_path = os.path.join(target_audio_path, file)  # Target file path

            # Check if the denoised file already exists
            if os.path.exists(target_file_path):
                print(f"Denoised already completed for {target_file_path}")
                continue  # Skip to the next file if already denoised

            # Step 1: Extract coefficients from the source signal
            concat_coeffs, coeffs_length = Coeff_signal(source_file_path, wavelet)

            # Step 2: Apply FISTA optimization to the coefficients for denoising
            xfista, _, _ = fista(concat_coeffs, lambda_, 50)  # FISTA with max 50 iterations

            # Step 3: Reconstruct the denoised coefficients back into the original structure
            coeffs = coeff_denoised(coeffs_length, xfista)

            # Step 4: Reconstruct the audio signal from the denoised coefficients
            reconstruct_audio = Reconstruct_audio(coeffs, wavelet)
            denoised_audio = torch.from_numpy(reconstruct_audio).unsqueeze(0)  # Convert to tensor

            # Step 5: Convert to float32 tensor for saving and store the denoised audio
            denoised_audio = denoised_audio.to(torch.float32)
            torchaudio.save(target_file_path, denoised_audio, sample_rate=52734)  # Saving audio

            print(f"Saved FISTA-denoised file at {target_file_path}")
        
        print(f"Denoising completed for folder: {folder}")
    print("All denoising completed.")
