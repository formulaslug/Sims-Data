import json
Magic:dict
Parameters:dict
with open('params.json', 'r') as file:
    params = json.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params
print("Parameters loaded...")
