import argparse
from typing import List, Dict, Any


def get_registered_arguments(parser: argparse.ArgumentParser) -> List[Dict[str, Any]]:
    """
    Extracts all registered arguments from an ArgumentParser object.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the details of a registered argument.
    """
    arguments = []

    for action in parser._actions:
        # Skip the help action
        if action.dest == "help":
            continue

        # Gather the argument details
        argument_info = {
            "name": action.dest,
            "type": action.type if action.type else type(action.default),
            "default": action.default,
            "help": action.help,
            "choices": action.choices,
        }
        arguments.append(argument_info)

    return arguments
