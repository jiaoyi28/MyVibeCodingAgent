from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaType(str, Enum):
    OBJECT = "object"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"


@dataclass
class ToolProperty:
    type: SchemaType | str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "description": self.description,
        }


@dataclass
class ToolParameters:
    type: SchemaType | str = SchemaType.OBJECT
    properties: dict[str, ToolProperty] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "properties": {
                name: prop.to_dict() for name, prop in self.properties.items()
            },
            "required": self.required,
        }


@dataclass
class Tool:
    name: str
    description: str
    parameters: ToolParameters
    type: str = "function"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }