from __future__ import annotations

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
    NULL = "null"


def _normalize_type(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_normalize_type(item) for item in value]
    return value


def _schema_to_dict(schema: Any) -> Any:
    if isinstance(schema, JsonSchema):
        return schema.to_dict()
    if isinstance(schema, dict):
        return {name: _schema_to_dict(value) for name, value in schema.items()}
    if isinstance(schema, list):
        return [_schema_to_dict(value) for value in schema]
    return schema


@dataclass
class JsonSchema:
    # Keep this model close to JSON Schema while remaining lightweight.
    type: SchemaType | str | list[SchemaType | str] | None = None
    description: str | None = None
    title: str | None = None
    properties: dict[str, "JsonSchema"] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    items: "JsonSchema | list[JsonSchema] | None" = None
    enum: list[Any] = field(default_factory=list)
    const: Any = None
    default: Any = None
    examples: list[Any] = field(default_factory=list)
    format: str | None = None
    pattern: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    min_items: int | None = None
    max_items: int | None = None
    additional_properties: bool | "JsonSchema" | None = None
    one_of: list["JsonSchema"] = field(default_factory=list)
    any_of: list["JsonSchema"] = field(default_factory=list)
    all_of: list["JsonSchema"] = field(default_factory=list)
    nullable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.type is not None:
            data["type"] = _normalize_type(self.type)
        if self.description is not None:
            data["description"] = self.description
        if self.title is not None:
            data["title"] = self.title
        if self.properties:
            data["properties"] = {
                name: prop.to_dict() for name, prop in self.properties.items()
            }
        if self.required:
            data["required"] = self.required
        if self.items is not None:
            data["items"] = _schema_to_dict(self.items)
        if self.enum:
            data["enum"] = self.enum
        if self.const is not None:
            data["const"] = self.const
        if self.default is not None:
            data["default"] = self.default
        if self.examples:
            data["examples"] = self.examples
        if self.format is not None:
            data["format"] = self.format
        if self.pattern is not None:
            data["pattern"] = self.pattern
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        if self.min_length is not None:
            data["minLength"] = self.min_length
        if self.max_length is not None:
            data["maxLength"] = self.max_length
        if self.min_items is not None:
            data["minItems"] = self.min_items
        if self.max_items is not None:
            data["maxItems"] = self.max_items
        if self.additional_properties is not None:
            data["additionalProperties"] = _schema_to_dict(self.additional_properties)
        if self.one_of:
            data["oneOf"] = [_schema_to_dict(schema) for schema in self.one_of]
        if self.any_of:
            data["anyOf"] = [_schema_to_dict(schema) for schema in self.any_of]
        if self.all_of:
            data["allOf"] = [_schema_to_dict(schema) for schema in self.all_of]
        if self.nullable is not None:
            data["nullable"] = self.nullable
        return data


@dataclass
class ToolProperty(JsonSchema):
    """Backward-compatible alias for nested tool parameter properties."""


@dataclass
class ToolParameters(JsonSchema):
    """Backward-compatible alias for tool root input schema."""

    type: SchemaType | str | list[SchemaType | str] | None = SchemaType.OBJECT


@dataclass
class Tool:
    name: str
    description: str
    parameters: JsonSchema
    type: str = "function"

    @property
    def input_schema(self) -> JsonSchema:
        return self.parameters

    def to_dict(self) -> dict[str, Any]:
        return self.to_openai_dict()

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }

    def to_input_schema_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.to_dict(),
        }
