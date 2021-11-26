import json
import pandas as pd
json_file = "05_Jazz3-150-C_comp.jams.json"
with open(json_file) as f:
    data = json.load(f)
time = []
duration = []
value = []
for i in data["annotations"][7]['data']:
    time.append(i["time"])
    duration.append(i["duration"])
    value.append(i["value"])


df = pd.DataFrame(columns=['time', 'duration', 'value'])
df['time'] = time
df['duration'] = duration
df['value'] = value

df.to_excel("output.xlsx") 