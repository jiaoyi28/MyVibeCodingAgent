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
    properties: dict[str, "ToolProperty"] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    items: "ToolProperty | None" = None
    enum: list[Any] = field(default_factory=list)
    min_items: int | None = None
    max_items: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "description": self.description,
        }
        if self.properties:
            data["properties"] = {
                name: prop.to_dict() for name, prop in self.properties.items()
            }
        if self.required:
            data["required"] = self.required
        if self.items is not None:
            data["items"] = self.items.to_dict()
        if self.enum:
            data["enum"] = self.enum
        if self.min_items is not None:
            data["minItems"] = self.min_items
        if self.max_items is not None:
            data["maxItems"] = self.max_items
        return data


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
