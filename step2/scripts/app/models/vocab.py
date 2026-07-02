from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class Term(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    synonyms: List[str] = []
    source_attributes: List[str] = []


class Mapping(BaseModel):
    model_config = ConfigDict(extra="allow")

    attribute: str
    canonical_term: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    match_details: Optional[dict] = None
    logical_type: Optional[str] = None
    normalization_profile: Optional[str] = None
    semantic_role: Optional[str] = None
    value_domain: Optional[List[str]] = None



class EntityMapping(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_entity: str
    canonical_entity: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    match_details: Optional[dict] = None


class Vocabulary(BaseModel):
    model_config = ConfigDict(extra="allow")

    terms: List[Term]
    mappings: List[Mapping]
    entity_mappings: List[EntityMapping] = []


# --- Models for the new OntologyProposal JSON schema (requested by user) ---

class Refinement(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = None
    note: Optional[str] = None


class MapsTo(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str  # property or relation_role
    canonical_name: Optional[str] = None
    role: Optional[str] = None  # identifier or descriptive
    relation: Optional[str] = None
    target_concept: Optional[str] = None


class AttributeProposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    maps_to: MapsTo
    confidence: Optional[float] = None


class ConceptProposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    canonical_name: str
    ontological_category: str
    source_entity: str
    confidence: Optional[float] = None
    refinement: Optional[Refinement] = None
    attributes: List[AttributeProposal] = []


class RelationProposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    canonical_name: str
    domain: str
    range: str
    from_cardinality: str
    to_cardinality: str
    confidence: Optional[float] = None


class OntologyProposal(BaseModel):
    model_config = ConfigDict(extra="allow")
    concepts: List[ConceptProposal]
    relations: List[RelationProposal] = []
