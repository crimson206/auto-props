import yaml
from typing import Dict, List, Union, Any
from crimson.intelli_type import IntelliType


class Json_(IntelliType[Union[Dict[str, Any], List[Any], Any]]):
    """
    **Description**:
        A class representing a JSON-like data structure, typically involving nested 
        dictionaries (`dict`) and lists (`list`). The structure can be complex and 
        variable, commonly found in web data formats.

        For more information about JSON, visit:
        https://www.json.org/json-en.html

    **Annotation**:
        Union[Dict[str, Any], List[Any], Any]
    """


def convert_json_to_yaml_code_block(data: Json_) -> str:
    yaml_string = yaml.dump(data, default_flow_style=False, sort_keys=False)
    yaml_code_block = f"``` yaml\n{yaml_string}```"
    return yaml_code_block


def sort_dictionary_order(dictionary: Dict, new_order: List[str]) -> Dict:
    sorted_dict = {key: dictionary[key] for key in new_order}
    return sorted_dict
