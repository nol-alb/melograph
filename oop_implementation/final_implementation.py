# This is the main file for Melograph final project.
# Group Members:
# Noel Alben
# Rhythm Jain
# Thiago R Roque

from OnsetDetection.OnsetDetection import OnsetDetection
from PitchDetection.PitchDetection import PitchDetection
from Visualization.Visualization import Visualization

if __name__ == '__main__':

	onset_samples = OnsetDetection()
	midi_collection = PitchDetection()
	graph = Visualization()
	graph.make_diGraph()
	graph.save_graphXML()
	#write file to images/ directory


