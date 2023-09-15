import json

def openFile(path):
    with open(path, 'r', encoding="utf8") as json_file:
        data = json.load(json_file)
    return data

def saveJsonFile(path, data):
    with open(path, 'w', encoding="utf8") as json_output:
        json.dump(data, json_output, indent=4)