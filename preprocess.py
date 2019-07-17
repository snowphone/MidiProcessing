#Original source: https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer
from enum import Enum
import errno
import json
import os
import glob
import shutil
from sys import argv, stderr

import numpy as np
import pretty_midi
from pypianoroll import Multitrack, Track

class Instrument(Enum):
	Piano = 0
	Guitar = 24 
	Bass = 32 
	Strings = 48


def isSuitable(path):
	"""Return True for qualified midi files and False for unwanted ones"""

	def _get_midi_info(pm):
			"""Return useful information from a pretty_midi.PrettyMIDI instance"""
			if pm.time_signature_changes:
				pm.time_signature_changes.sort(key=lambda x: x.time)
				first_beat_time = pm.time_signature_changes[0].time
			else:
				first_beat_time = pm.estimate_beat_start()

			tc_times, tempi = pm.get_tempo_changes()

			if len(pm.time_signature_changes) == 1:
				time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
											pm.time_signature_changes[0].denominator)
			else:
				time_sign = None

			velocities = len({note.velocity for instrument in pm.instruments
							for note in instrument.notes})

			midi_info = {
				'first_beat_time': first_beat_time,
				'num_time_signature_change': len(pm.time_signature_changes),
				'time_signature': time_sign,
				'tempo': tempi[0] if len(tc_times) == 1 else None,
				'velocities' : velocities
			}

			return midi_info

	try:
		pm = pretty_midi.PrettyMIDI(path)
		midi_info = _get_midi_info(pm)
	except:
		return False

	if midi_info['first_beat_time'] > 0.0:
		return False
	elif midi_info['num_time_signature_change'] > 1:
		return False
	elif midi_info['time_signature'] not in ['4/4']:
		return False
	else:
		return True



class Converter:

	beat_resolution = 4	#16th note is a minimum beat unit
	base_note = 4
	bars = 4

	def convertToMatrix(path):
		"""Return (piano, guitar, bass, string)-track piano-roll from a MIDI file"""

		multitrack = Multitrack(path, beat_resolution=Converter.beat_resolution, name=os.path.basename(path))

		#Merge into 4 tracks
		mergedTracks = Converter._merge(multitrack)
		
		#merged.save(os.path.join(converter_path, midi_name + '.npz'))
		mergedTracks.binarize()
		ret = mergedTracks.get_stacked_pianoroll()
		return ret


	def _merge(multitrack):
		"""Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
		four tracks (Bass, Guitar, Piano and Strings)"""
		category_list = {'Piano': [], 'Guitar': [], 'Bass': [], 'Strings': []}
		program_dict = {i.name: i.value for i in Instrument}
		#cf) program: 0 to 127


		for idx, track in enumerate(multitrack.tracks):
			if track.is_drum:
				pass
			elif track.program // 8 == 0:	#0 - 7
				category_list['Piano'].append(idx)
			elif track.program // 8 == 3:	#24 - 31
				category_list['Guitar'].append(idx)
			elif track.program // 8 == 4:	#32 - 39
				category_list['Bass'].append(idx)
			elif track.program // 8 == 5:	#40 - 47
				category_list['Strings'].append(idx)
			else:
				category_list['Strings'].append(idx)
				pass

		tracks = []
		for key, index_list in category_list.items():
			if index_list:
				merged = multitrack[index_list].get_merged_pianoroll()
				tracks.append(Track(pianoroll=merged, program=program_dict[key], is_drum=False, name=key))
			else:
				tracks.append(Track(pianoroll=None, program=program_dict[key], is_drum=False, name=key))

		return Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)

class Splitter:
	pitch_beg = 24
	pitch_end = 108
	input_len = Converter.beat_resolution * Converter.base_note * Converter.bars

	def split(pianoroll, last_bar_mode="fill"):
		pianoroll = Splitter._clampPitch(pianoroll)
		pianoroll = Splitter._pad(pianoroll, last_bar_mode)
		pianoroll = Splitter._adjustVelocity(pianoroll)
		pianoroll = Splitter._reshapeToInput(pianoroll)
		return pianoroll

	def _clampPitch(pianoroll):
		return pianoroll[:, Splitter.pitch_beg:Splitter.pitch_end, :]  #Shape: (time, pitches, programs)

	def _pad(pianoroll, last_bar_mode):
		pitches = Splitter.pitch_end - Splitter.pitch_beg 
		tracks = pianoroll.shape[-1]

		if int(pianoroll.shape[0] % Splitter.input_len) != 0:
			if last_bar_mode == "fill":
				bubble = np.zeros([Splitter.input_len - pianoroll.shape[0] % Splitter.input_len, pitches, tracks])
				pianoroll = np.concatenate((pianoroll, bubble), axis=0)
			elif last_bar_mode == "remove":
				pianoroll = np.delete(pianoroll,  np.s_[-int(pianoroll.shape[0] % Splitter.input_len):], axis=0)
		return pianoroll
	
	def _adjustVelocity(pianoroll):
		return pianoroll

	def _reshapeToInput(pianoroll):
		tracks = pianoroll.shape[-1]
		pitches = Splitter.pitch_end - Splitter.pitch_beg 
		input_list = pianoroll.reshape(-1, Splitter.input_len, pitches, tracks)
		return input_list


def get_midi_paths(root):
	"""Return a list of paths to MIDI files in `root` (recursively)"""
	return [os.path.join(dirpath, filename) 
			for dirpath, _, filenames in os.walk(root) 
			for filename in filenames 
			if filename.endswith(".mid")]

def saveMatrix(arr, basePath, name, test_ratio=0.1):
	if np.random.rand() < test_ratio:
		#test
		save_path = os.path.join(basePath, "test", name)
	else:
		save_path = os.path.join(basePath, "train", name)
		#train
	np.save(save_path, arr)
	return


def preprocess(MIDI_path):
	print("Genre: {}".format(MIDI_path))

	os.makedirs(os.path.join(MIDI_path, "test"), exist_ok=True)
	os.makedirs(os.path.join(MIDI_path, "train"), exist_ok=True)

	path_list = get_midi_paths(MIDI_path)
	
	print("Input: {} MIDI files".format(len(path_list)))

	cnt = 0
	for path in path_list:
		try:
			if not isSuitable(path):
				continue

			name = os.path.basename(path)

			cnt += 1
			pianoroll = Converter.convertToMatrix(path)
			splitted_input_list = Splitter.split(pianoroll, "remove")
			print("#{:3d}: Processing {} into {} matrices".format(cnt, name, len(splitted_input_list)))

			for i, splitted_input in enumerate(splitted_input_list):
				saveMatrix(splitted_input, MIDI_path, "{}_{}.npy".format(name, i))

		except Exception as e:
			print(e)
			continue
	print("{} MIDI files are converted from {} inputs".format(cnt, len(path_list)))
	return
			
def makeJCP_mixed(MIDI_path):
	path = os.path.join(MIDI_path, "../JCP_mixed/")
	os.makedirs(path, exist_ok=True)
	npy_list = glob.glob(os.path.join(MIDI_path, "train/*.npy"))
	for i in npy_list:
		shutil.copyfile(os.path.abspath(i), os.path.join(path, os.path.basename(i)))

if __name__ == "__main__":
	for MIDI_path in argv[1:]:
		preprocess(MIDI_path)
		makeJCP_mixed(MIDI_path)
	print("Done!")
