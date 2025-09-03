# =============================================================================
# 1. ENHANCED STATE DEFINITION with Decision Tracking
# =============================================================================

from typing import List, TypedDict, Annotated, Optional, Dict, Any
import operator
from langchain_core.messages import BaseMessage
from schema import EvidenceItem, VerificationResult, InvestigationCycle

class CycleDecision(TypedDict):
    """Tracks the reasoning for each cycle's decisions."""
    cycle_id: int
    query: str
    decision_type: str  # "continue" or "conclude"
    reasoning: str
    timestamp: str
    evidence_count: int
    analysis_count: int

class AgentState(TypedDict):
    """The agent's working memory for an execution cycle."""
    # Core identifiers
    initial_query: str
    investigation_id: str
    cycle_id: int
    
    # The query for the current cycle
    current_query: str

    # Conversational memory (optional, but good practice)
    messages: Annotated[List[BaseMessage], operator.add]

    # Data generated IN THIS CYCLE
    new_evidence: Annotated[List[EvidenceItem], operator.add]
    new_analysis: Annotated[List[VerificationResult], operator.add]
    
    # Synthesized context for the refiner
    graph_context: Optional[Dict[str, Any]]

    # Full history from all past cycles
    investigation_cycles: List[InvestigationCycle]

    # Control flow for the next cycle
    next_query: Optional[str]

    # Final result from the reporter
    final_summary: Optional[Dict[str, Any]]
    
    # Tracking to prevent infinite loops
    investigated_queries: List[str]
    investigated_entities: List[str]
    max_cycles: int
    investigation_complete_reason: Optional[str]
    
    # NEW: Decision tracking for transparency
    cycle_decisions: List[CycleDecision]
    final_conclusion: Optional[Dict[str, Any]]
