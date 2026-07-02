from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class Attribute(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    data_type: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None
    nullable: Optional[bool] = None


class Entity(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    attributes: List[Attribute]
    primary_key: Optional[List[str]] = None
    description: Optional[str] = None


class Relationship(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    from_entity: str
    to_entity: str
    from_cardinality: str = Field(..., description="e.g. 1, 0..1, 0..N, 1..N")
    to_cardinality: str = Field(..., description="e.g. 1, 0..1, 0..N, 1..N")
    fk_attribute: Optional[str] = None
    pk_attribute: Optional[str] = None
    relationship_type: Optional[str] = None


class ERDModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    entities: List[Entity]
    relationships: List[Relationship] = []
