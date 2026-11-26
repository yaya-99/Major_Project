"""Pydantic schemas of the Caption Reliability Inspector API (response schema version 2)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2


class OverallFlag(BaseModel):
    status: str
    badge: str
    limitation: str = ""


class TokenDetail(BaseModel):
    word: str
    status: str    # supported_by_detected_region | possible_hallucination | possible_misidentification | insufficient_visual_evidence | not_assessed
    claim_type: str = "other"
    similarity: Optional[float] = None          # cosine similarity (NOT a probability)
    category: Optional[str] = None
    rule: str = ""
    note: str = ""
    limitation: str = ""
    evidence_region_id: Optional[int] = None
    detector_label: Optional[str] = None
    detector_confidence: Optional[float] = None
    bbox: Optional[List[float]] = None          # DETECTOR box in original-image pixels (not attention)


class CaptionOutput(BaseModel):
    schema_version: int = SCHEMA_VERSION
    status: str = "ok"
    caption: str
    tokens: List[str]
    reliability: List[str]
    token_details: List[TokenDetail] = Field(default_factory=list)
    attention_map: List[List[float]] = Field(default_factory=list)   # cross-attention (model attention), grid
    heatmap_png: Optional[str] = None            # base64 PNG ATTENTION overlay
    image_size: Optional[List[int]] = None       # [width, height] the bbox coordinates refer to
    overall: OverallFlag
    source: str = "model"                        # "model" | "cached_example"
    has_token_projection: bool = False
    evidence_scope: str = "object-level only; attributes and relations are not assessed"
    model: Optional[str] = None


class ModelStatus(BaseModel):
    status: str                                  # ok | unavailable | error
    detail: str = ""


class ComparisonOutput(BaseModel):
    schema_version: int = SCHEMA_VERSION
    B1: Optional[CaptionOutput] = None
    B2: Optional[CaptionOutput] = None
    A: Optional[CaptionOutput] = None
    statuses: Dict[str, ModelStatus] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    status: str
    uptime_seconds: float = 0.0
    schema_version: int = SCHEMA_VERSION


class ReadyStatus(BaseModel):
    ready: bool
    models: Dict[str, str] = Field(default_factory=dict)     # loaded | available | unavailable
    grounding_loaded: bool = False
    detail: str = ""


class ModelInfo(BaseModel):
    name: str
    description: str
    checkpoint: Optional[str] = None
    available: bool
    loaded: bool


class SelectedModel(BaseModel):
    variant: Optional[str] = None
    state: str                                   # validated | unavailable
    detail: str = ""


class ExampleInfo(BaseModel):
    name: str
    url: str
    source: str = ""
    license: str = ""
    purpose: str = ""
    curated_fallback: bool = True


class ExamplesOut(BaseModel):
    images: List[ExampleInfo] = Field(default_factory=list)
    empty: bool = True
    disclosure: str = "Curated fallback images; results shown for them are not a live user upload."
    problems: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
