from copy import copy, deepcopy
import json


class DataAccessor:
    def __init__(self, path: str):
        self.__path = path
        self.__config = self.__load_config()

    def __load_config(self):
        with open(self.__path, 'r') as file:
            config = file.read()
            return config

    def get_parameter(self, *args):
        part = json.loads(self.__config)
        for arg in args:
            part = part[arg]
        return part["value"]

    def get_config(self):
        return deepcopy(self.__config)

    def __repr__(self):
        return json.dumps(self.__config, sort_keys=True, indent=4)
