import unittest
import os
import pandas as pd
import librosa
import torch
import numpy as np

from audio import load_audio_file, pad_sequence, extract_features

# Sample data for testing
class TestAudioProcessing(unittest.TestCase):
    
    def test_load_audio_file(self):
        sample_rate = 22050
        # Create a temporary audio file
        signal = np.random.randn(sample_rate * 2)  # 2 seconds of random noise
        file_path = 'test.wav'
        librosa.output.write_wav(file_path, signal, sample_rate)
        
        loaded_signal, sr = load_audio_file(file_path)
        
        self.assertEqual(sr, sample_rate)
        self.assertTrue(np.array_equal(signal, loaded_signal))
        
        # Clean up
        os.remove(file_path)

    def test_pad_sequence(self):
        seq = np.random.randn(13, 150)  # Example MFCCs
        max_length = 174
        padded_seq = pad_sequence(seq, max_length)
        
        self.assertEqual(padded_seq.shape[1], max_length)
        self.assertTrue(np.array_equal(padded_seq[:, :150], seq))
        self.assertTrue(np.all(padded_seq[:, 150:] == 0))

    def test_extract_features(self):
        signal = np.random.randn(22050)  # 1 second of random noise
        sr = 22050
        n_mfcc = 13
        max_length = 174
        n_fft = 512
        
        features = extract_features(signal, sr, n_mfcc, max_length, n_fft)
        
        self.assertEqual(features.shape, (n_mfcc, max_length))
        self.assertIsInstance(features, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
