import glob
import os
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Visualization:
	def make_diGraph(self, baseline_midi_notes, ):
		G = nx.DiGraph()
		G.add_nodes_from(baseline_midi_notes)
		G.add_edges_from(baselineTransitions)
		d = nx.degree(G)
		degrees = []
		for i in d:
    		degrees.append(i[1])
    	pos = nx.spring_layout(G, k=0.15, iterations=20)
		nx.draw(G, with_labels=True, font_weight='bold', node_size=[v * 100 for v in degrees])

	def return_baseline_transitions(self, baseline_midi_notes):
		baselineTransitions = []
		for i in range(len(baselinePitches)-1):
    		baselineTransitions.append((str(baselinePitches[i])+"",str(baselinePitches[i+1])))

    def save_graphXML(self, graph, path):
    	nx.write_graphml(graph, path)