import numpy as np
import scipy 
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.signal import find_peaks

@dataclass
class Buffer:
    sr: float = 48000
    buffer: np.ndarray = np.zeros(int(sr))

    def create_buffer(self, seconds=0.0):
        self.buffer = np.zeros(int(ceil(self.sr * seconds)))

@dataclass
class OnsetDetection:
	#needs a signal which passes from transform_signal_to_spectral_flux to novelty and to find_peak
	def transform_signal_to_spectral_flux(self):
		pass

	def novelty(self):
		pass

	def find_peak(self):
		#returns np array
		pass 

	def peak_to_sample(self):
		pass

	def harmonic_percussive_source_separation(self):
		pass

	def spectral_flux(self)
		pass