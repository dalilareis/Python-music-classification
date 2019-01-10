
import os
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
import ast
import sys
from ctypes import *
from ctypes.wintypes import *

#------------------------------------Auxiliary functions-------------------------------------------
def sched_getaffinity(pid):
    # Number of samples per 30s audio clip.
    NB_AUDIO_SAMPLES = 1321967
    SAMPLING_RATE = 44100

    #Use multiple cores for extraction (in windows, with function sched_getaffinity
    __all__ = ['sched_getaffinity', 'sched_setaffinity']

    kernel32 = WinDLL('kernel32')

    DWORD_PTR = WPARAM
    PDWORD_PTR = POINTER(DWORD_PTR)

    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = HANDLE

    OpenProcess = kernel32.OpenProcess
    OpenProcess.restype = HANDLE
    OpenProcess.argtypes = (DWORD, # dwDesiredAccess,_In_
                            BOOL,  # bInheritHandle,_In_
                            DWORD) # dwProcessId, _In_

    GetProcessAffinityMask = kernel32.GetProcessAffinityMask
    GetProcessAffinityMask.argtypes = (
        HANDLE,     # hProcess, _In_
        PDWORD_PTR, # lpProcessAffinityMask, _Out_
        PDWORD_PTR) # lpSystemAffinityMask, _Out_

    SetProcessAffinityMask = kernel32.SetProcessAffinityMask
    SetProcessAffinityMask.argtypes = (
        HANDLE,    # hProcess, _In_
        DWORD_PTR) # dwProcessAffinityMask, _In_

    PROCESS_SET_INFORMATION = 0x0200
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    if sys.getwindowsversion().major < 6:
        PROCESS_QUERY_LIMITED_INFORMATION = PROCESS_QUERY_INFORMATION
    if pid == 0:
        hProcess = GetCurrentProcess()
    else:
        hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not hProcess:
        raise WinError()
    lpProcessAffinityMask = DWORD_PTR()
    lpSystemAffinityMask = DWORD_PTR()
    if not GetProcessAffinityMask(hProcess, byref(lpProcessAffinityMask), byref(lpSystemAffinityMask)):
        raise WinError()
    mask = lpProcessAffinityMask.value
    return {c for c in range(sizeof(DWORD_PTR) * 8) if (1 << c) & mask}

def load(filepath):

    filename = os.path.basename(filepath)

    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                ('album', 'date_created'), ('album', 'date_released'),
                ('artist', 'date_created'), ('artist', 'active_year_begin'),
                ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            'category', categories=SUBSETS, ordered=True)

    COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                ('album', 'type'), ('album', 'information'),
                ('artist', 'bio')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks

def get_fs_tids(audio_dir):
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

#-----------------------------Extract Features and build csv file-----------------------------------

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()

def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        filepath = get_audio_path('fma_small', tid)

        x, sr = librosa.load(filepath, sr=None, mono=True, res_type='kaiser_fast')

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rmse(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))

    return features

def main():
    tracks = load('tracks.csv')   
    tids = get_fs_tids('fma_small')

    features = pd.DataFrame(index=tids, columns=columns(), dtype=np.float32)
    genres = tracks.loc[tids, ('track', 'genre_top')].values

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(len(sched_getaffinity(0)))

    words = [nb_workers]
    for nb_workers in words:                       
        print('Working with {} processes.'.format(nb_workers))

        pool = multiprocessing.Pool(nb_workers)
        it = pool.imap_unordered(compute_features, tids)

        for i, row in enumerate(tqdm(it, total=len(tids))):
            features.loc[row.name] = row

            if i % 1000 == 0:
                features['', '','genre'] = genres
                save(features, 10)

    features['', '','genre'] = genres
    save(features, 10)

def save(features, ndigits):
    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv('features.csv', float_format='%.{}e'.format(ndigits))

if __name__ == "__main__":
    main()
