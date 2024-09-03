from typing import List, Dict
from .base import generate_property, generate_setter, Property
from crimson.templator import format_indent, format_insert
import os


def generate_properties(properties: List[Dict]) -> str:
    properties_string = str()

    for arg in properties:
        properties_string += generate_property(arg)

    return properties_string


def generate_setters(properties: List[Dict]) -> str:
    setters = str()

    for arg in properties:
        setters += generate_setter(arg)

    return setters


def generate_fields(properties: List[Property]) -> str:
    """
    Generates field definitions for a class.

    Args:
        fields_info (List[Dict[str, Any]]): A list of dictionaries, each containing:
            - name (str): The name of the field.
            - type (type): The type of the field.
            - default (Any): The default value of the field.

    Returns:
        str: A string containing the field definitions.
    """
    fields_str = ""

    for field in properties:
        field_type = field["type"].__name__  # Get the type as a string
        field_name = field["name"]
        field_default = repr(
            field["default"]
        )  # Convert the default value to a string representation
        fields_str += f"self._{field_name}: {field_type} = {field_default}\n"

    return fields_str


def generate_model(cls_name, properties: List[Property]) -> str:
    args_model_t = r"""
class \[cls_name\]:
    def __init__(self):
        \{fields\}
    \{properties\}
    \{setters\}
"""

    fields = generate_fields(properties)
    properties_string = generate_properties(properties)
    setters = generate_setters(properties)
    args_model_t = format_insert(args_model_t, cls_name=cls_name)

    kwargs = {
        "fields": fields,
        "properties": properties_string,
        "setters": setters,
    }

    model_formatted = format_indent(args_model_t, **kwargs)

    return model_formatted


def generate_file(content: str, directory: str, filename: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    with open(file_path, "w") as file:
        file.write(content)

    print(f"File generated at: {file_path}")
