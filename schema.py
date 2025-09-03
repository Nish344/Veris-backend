# schema.py - Enhanced for Phase 2 Requirements

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional, Literal, Union, TypedDict, Annotated
import operator

from pydantic import BaseModel, Field, HttpUrl, field_validator
from langchain_core.messages import BaseMessage

# --- Enums for controlled vocabularies ---

class InvestigationStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ThreatLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MediaType(str, Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"

class SourceType(str, Enum):
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"
    WEB_PAGE = "web_page"
    NEWS = "news"
    BLOG = "blog"
    FORUM = "forum"
    SCREENSHOT = "screenshot"
    PROFILE_ANALYSIS = "profile_analysis" # Added for author profiling

# --- Core Data Models ---

class MediaItem(BaseModel):
    """Represents a single piece of media, like an image or video."""
    media_id: str = Field(default_factory=lambda: f"media_{uuid.uuid4().hex[:12]}")
    media_type: MediaType
    url: Union[HttpUrl, str]
    local_path: Optional[str] = None
    access_url: Optional[str] = None # URL to access via API, e.g., /media/{media_id}

class EvidenceItem(BaseModel):
    """A single, validated piece of evidence from a source."""
    evidence_id: str = Field(default_factory=lambda: f"evd_{uuid.uuid4().hex[:12]}")
    investigation_id: Optional[str] = None
    cycle_id: Optional[int] = None
    source_type: str
    url: Optional[Union[HttpUrl, str]] = None
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    author_id: str = "unknown"
    media: List[MediaItem] = Field(default_factory=list)
    mentioned_accounts: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None

class VerificationResult(BaseModel):
    """Analysis results for a single piece of evidence."""
    analysis_id: str = Field(default_factory=lambda: f"anl_{uuid.uuid4().hex[:12]}")
    evidence_id: str
    cycle_id: Optional[int] = None
    flag_reason: str
    trust_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class FinalConclusion(BaseModel):
    """The final, synthesized report for an investigation."""
    summary: str
    threat_level: ThreatLevel
    tags: List[str]

# --- Investigation and Cycle Management ---

class InvestigationCycle(BaseModel):
    """Represents one full loop of the Collector -> Verifier -> Refiner process."""
    cycle_id: int
    query: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    evidence_collected: List[EvidenceItem] = Field(default_factory=list)
    analysis_results: List[VerificationResult] = Field(default_factory=list)

class Investigation(BaseModel):
    """Top-level object representing a full investigation from start to finish."""
    investigation_id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:12]}")
    initial_query: str
    vip_targets: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: InvestigationStatus = InvestigationStatus.RUNNING
    investigation_cycles: List[InvestigationCycle] = Field(default_factory=list)
    final_conclusion: Optional[FinalConclusion] = None

# --- API and Search Models ---

class InvestigationStartResponse(BaseModel):
    """Response when a new investigation is successfully started."""
    investigation_id: str
    status: InvestigationStatus
    message: str = "Investigation started successfully"

class SearchQuery(BaseModel):
    """Structured search query with context."""
    query: str = Field(..., description="The search query")
    source_types: List[str] = Field(default_factory=list)
    max_results: int = Field(10, ge=1, le=50)
    time_range: Optional[str] = Field(None)

# --- Graph Analysis Models ---

class GraphNode(BaseModel):
    """Represents a node in the investigation network graph."""
    node_id: str
    node_type: Literal["author", "hashtag", "mention", "url", "media"]
    label: str
    properties: Dict[str, str] = Field(default_factory=dict)
    threat_score: float = Field(0.0, ge=0.0, le=1.0)

class GraphEdge(BaseModel):
    """Represents an edge in the investigation network graph."""
    source_id: str
    target_id: str
    relationship: str
    weight: float = Field(1.0, ge=0.0)
    evidence_ids: List[str] = Field(default_factory=list)

class InvestigationGraph(BaseModel):
    """Complete graph representation of an investigation."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
