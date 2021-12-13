from music21 import pitch as pt

path = 'test.npy'
with open('test.npy', 'rb') as f:  
	a = np.load(f)


for i in a:
	