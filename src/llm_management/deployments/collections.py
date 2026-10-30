"""
Normalize configuration arrays into one slug-keyed runtime collection.
"""

from __future__ import annotations

from typing import Annotated, Any, Protocol, TypeVar, get_args

from pydantic import GetCoreSchemaHandler, ValidatorFunctionWrapHandler
from pydantic_core import core_schema


class SluggedConfig(Protocol):
    """
    Identity shared by validated deployment and group configuration entries.
    """

    @property
    def slug(self) -> str:
        """
        Return the entry's configured identity.
        """
        ...


def validate_slug_map(
    value: Any, handler: ValidatorFunctionWrapHandler
) -> dict[str, SluggedConfig]:
    """
    Validate entries before indexing them, preserving order and rejecting duplicates.

    Array input uses temporary position keys so repeated slugs survive until the
    entry schemas have validated them. Mapping input must agree with each entry's
    slug. The handler validates the concrete model types declared on the field;
    this function is shared only by configuration collections with a slug field.
    """
    from_array = isinstance(value, list)
    if from_array:
        value = {str(index): entry for index, entry in enumerate(value)}
    entries: dict[str, SluggedConfig] = handler(value)
    indexed: dict[str, SluggedConfig] = {}
    for key, entry in entries.items():
        if not from_array and key != entry.slug:
            raise ValueError(
                f"Collection key {key!r} does not match entry slug {entry.slug!r}"
            )
        if entry.slug in indexed:
            raise ValueError(f"Deployment slugs must be unique: {entry.slug!r}")
        indexed[entry.slug] = entry
    return indexed


T = TypeVar("T", bound=SluggedConfig)


def serialize_slug_map(entries: dict[str, T]) -> list[T]:
    """
    Export the runtime index as an ordered configuration array.
    """
    return list(entries.values())


class SlugMapAnnotation:
    """
    Attach validation and serialization using the alias's resolved entry type.

    Pydantic does not substitute generic parameters embedded in WrapValidator's
    input type or PlainSerializer's return type. Generate those schemas here
    from dict[str, ConcreteConfig] so both schema modes retain typed entries,
    including discriminated unions.
    """

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Accept arrays or keyed objects and serialize both as typed arrays.
        """
        entry_type = get_args(source_type)[1]
        mapping_schema = handler(source_type)
        array_schema = handler.generate_schema(list[entry_type])
        return core_schema.no_info_wrap_validator_function(
            validate_slug_map,
            mapping_schema,
            json_schema_input_schema=core_schema.union_schema(
                [array_schema, mapping_schema]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_slug_map, return_schema=array_schema
            ),
        )


SlugMappedList = Annotated[dict[str, T], SlugMapAnnotation()]
