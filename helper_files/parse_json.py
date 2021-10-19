import json
json_file = 'file.json'
with open(json_file) as f:
    data = json.load(f)
time = []
duration = []
value = []
for i in data["annotations"][7]['data']:
    time.append(i["time"])
    duration.append(i["duration"])
    value.append(i["value"])

