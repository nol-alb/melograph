import os
import glob
import json
import subprocess
import numpy as np

import pandas as pd
import networkx as nx
from music21 import pitch
import matplotlib.pyplot as plt

def make_graph():
    path = '../Visualization/test.npy'
    with open(path, 'rb') as f1:
        a = np.load(f1)

    pitches =[] 
    for i in a:
        pitches.append(str(pitch.Pitch(i)))

    # print("pitsches:", pitches)

    with open("temp.txt", "w") as f2:
        for ele in pitches:
            f2.write(ele+"\n")
    os.system("context -n 2 temp.txt | sed 's/ /, /g' | infot -n > temp2.txt")

    path = "temp2.txt"

    df=pd.read_csv(path, sep='\t', header=None)
    df.columns=["transitions", "weights"]
    # display(df)
    tr_str=np.array(df["transitions"])
    tr=[]
    for j in range(len(tr_str)):
        tr.append((tr_str[j].split(", ")))

    print("weights:", type(df["weights"]))
    df["weights"][(df["weights"])<3]=3
    df["weights"][(df["weights"])>10]=10
    withcomma=[]
    for ind in range(len(tr)):
        withcomma.append((tr[ind][0].split(" ")))
    # print("withcomma:",withcomma)
    swaras, counts=np.unique(np.array(withcomma).flatten(), return_counts=True)
    # print(counts[7])
    counts[(counts*10)<30]=3
    counts=counts.tolist()


    # print("swaras: ",swaras)

    data = {"nodes": [], "edges": []}

    for i in range(len(swaras)):
        if i == 0:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'type':'rectangle', 'weight': counts[i]*10, 'color' : 'red' }})
        elif i == len(swaras)-1:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'type':'rectangle', 'weight': counts[i]*10, 'color' : 'blue' }})
        else:
            data["nodes"].append({ 'data': { 'id': swaras[i], 'name': swaras[i], 'weight': counts[i]*10, 'color': '#a38344' }})
        
    # print(len(tr))
    for j in range(len(tr)):
        if tr[j][0] in swaras and tr[j][1] in swaras:
            data["edges"].append({'data': { 'source': tr[j][0], 'target': tr[j][1], 'weight': int(df["weights"][j]) }})
    print(len(data["edges"]))
    with open("data.json", "w") as f3:
        json.dump(data, f3)

    f1.close()
    f2.close()
    f3.close()

make_graph()
