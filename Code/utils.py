import pandas as pd
import os.path
import ast
import sys
from ctypes import *
from ctypes.wintypes import *



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

