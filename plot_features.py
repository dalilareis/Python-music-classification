
import os
import sys
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
import librosa.display 
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt

def plot_chromas(name, feature):
	plt.figure(figsize=(10, 5))
	plt.subplot(2,2,1)
	librosa.display.specshow(feature[0], y_axis='chroma')
	plt.title(name[0])
	plt.colorbar()
	plt.subplot(2,2,2)
	librosa.display.specshow(feature[1], y_axis='chroma')
	plt.title(name[1])
	plt.colorbar()
	plt.subplot(2,2,3)
	librosa.display.specshow(feature[2], y_axis='chroma', x_axis='time')
	plt.title(name[2])
	plt.colorbar()
	plt.subplot(2,2,4)
	librosa.display.specshow(feature[3], y_axis='tonnetz', x_axis='time')
	plt.title('Tonal Centroids: %s' % name[3])
	plt.colorbar()
	plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.01, hspace=0.1)
	plt.tight_layout()
	plt.show()

def plot_spectrals(name, features):
	plt.figure(figsize=(10, 5))
	plt.subplot(2, 2, 1)
	plt.semilogy(features[0].T, label=name[0])
	plt.ylabel('Hz')
	plt.xticks([])
	plt.xlim([0, features[0].shape[-1]])
	plt.legend()
	plt.subplot(2, 2, 2)
	plt.semilogy(features[1].T, label=name[1])
	plt.ylabel('Hz')
	plt.xticks([])
	plt.xlim([0, features[1].shape[-1]])
	plt.legend()
	plt.subplot(2, 2, 3)
	plt.semilogy(features[2].T, label=name[2])
	plt.ylabel('Hz')
	plt.xticks([])
	plt.xlim([0, features[2].shape[-1]])
	plt.legend()
	plt.subplot(2, 2, 4)
	librosa.display.specshow(features[3], x_axis='time')
	plt.colorbar()
	plt.ylabel('Frequency bands')
	plt.title('Spectral contrast')
	plt.tight_layout()
	plt.show()	

def plot_bases(features):
	plt.figure()
	plt.subplot(3, 1, 1)
	librosa.display.specshow(librosa.amplitude_to_db(features[0], ref=np.max), y_axis='log')
	plt.colorbar(format='%+2.0f dB')
	plt.title('log Power Spectrogram (Short-Time Fourier Transform)')
	plt.subplot(3, 1, 2)
	librosa.display.specshow(librosa.amplitude_to_db(features[1], ref=np.max), y_axis='cqt_note')
	plt.colorbar(format='%+2.0f dB')
	plt.title('power Spectrogram (Constant-Q Transform)')
	plt.subplot(3, 1, 3)
	librosa.display.specshow(librosa.power_to_db(features[2], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel spectrogram from STFT (Mel-scaled)')
	plt.tight_layout()	
	plt.show()

def plot_others(name, features):
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.semilogy(features[0].T, label=name[0])
	plt.xticks([])
	plt.xlim([0, features[0].shape[-1]])
	plt.legend(loc='best')
	plt.subplot(2, 1, 2)
	librosa.display.specshow(features[1], x_axis='time')
	plt.colorbar()
	plt.title(name[1])
	plt.tight_layout()
	plt.show()


tids = utils.get_fs_tids('fma_small')

filepath = utils.get_audio_path('fma_small', tids[0])

x, sr = librosa.load(filepath, sr=None, mono=True, res_type='kaiser_fast')

cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
assert cqt.shape[0] == 7 * 12
assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)

stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
assert stft.shape[0] == 1 + 2048 // 2
assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
del x

chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
rmse = librosa.feature.rmse(S=stft)
spectral_centroid = librosa.feature.spectral_centroid(S=stft)
spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)
spectral_contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)
spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)

mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)

mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

names_chroma = ["chroma_cqt", "chroma_cens", "chroma_stft", "tonnetz"] 
chroma = [chroma_cqt, chroma_cens, chroma_stft, tonnetz]

names_spectral = ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_contrast"]
spectral = [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast]

names_others = ["RMS Energy", "MFCC"]
others = [rmse, mfcc]

bases = [stft, cqt, mel]

del stft, cqt, mel

if __name__ == "__main__":
	if sys.argv[1] == 'base':
		plot_bases(bases)
	elif sys.argv[1] == 'chroma':
		plot_chromas(names_chroma, chroma)
	elif sys.argv[1] == 'spectral':
		plot_spectrals(names_spectral, spectral)
	elif sys.argv[1] == 'others':
		plot_others(names_others, others)






