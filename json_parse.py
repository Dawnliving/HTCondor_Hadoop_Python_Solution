import json
from typing import Any

class DynamicJsonObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # Handle nested dictionaries
            if isinstance(value, dict):
                setattr(self, key, DynamicJsonObject(**value))
            # Handle lists of dictionaries
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                setattr(self, key, [DynamicJsonObject(**item) for item in value])
            else:
                setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


def create_nested_object_from_json(json_file: str) -> Any:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return DynamicJsonObject(**data)