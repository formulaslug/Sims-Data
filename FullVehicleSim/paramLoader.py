import json5
Magic:dict
Parameters:dict
with open('params.json5', 'r') as file:
    params = json5.load(file)
    Magic = params["Magic"]
    Parameters = params["Parameters"]
    del params
print("Parameters loaded...")
