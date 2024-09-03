from typing import Dict, Optional
from crimson.templator import format_insert, format_indent

from typing import TypedDict, Any


class Property(TypedDict, total=False):
    """
    data with

    name: str
    type: str
    default: Any
    description: str
    """

    name: str
    type: str
    default: Any
    description: str


def type_to_str(obj: Optional[type]):
    if isinstance(obj, type):
        return obj.__name__
    return obj


def generate_property(property: Property) -> str:
    property_t = r'''
@property
def \[name\](self) -> \[type\]:
    """
    \{description\}
    """
    return self._\[name\]
'''
    kwargs = property.copy()
    kwargs["type"] = type_to_str(kwargs["type"])

    property_t = format_insert(property_t, **kwargs)

    property_f = format_indent(property_t, **kwargs)

    return property_f


def generate_setter(property: Dict) -> str:
    property_t = r"""
@\[name\].setter
def \[name\](self, value: \[type\]):
    self._\[name\] = value
"""
    kwargs = property.copy()
    kwargs["type"] = type_to_str(kwargs["type"])

    property_f = format_insert(property_t, **kwargs)

    return property_f
