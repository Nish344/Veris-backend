#!/usr/bin/env python3
# workflow/refiner.py
"""
Refiner node for the verifier pipeline.

- enhanced_refinement_node(state, llm): main async entry implementing the refiner logic.
- FakeLLM: a tiny deterministic "LLM" used as fallback for local testing (returns JSON).
- run_refiner_from_file(json_path, max_cycles=5): convenience runner for local tests.

Usage:
    python -m workflow.refiner /path/to/test_outputs_graph/evidence_..._verified.json

The refiner expects `state` to include keys such as:
    - initial_query
    - graph_context (graph dict with 'nodes' and 'edges')
    - cycle_id
    - max_cycles
    - investigated_queries (list)
    - investigated_entities (list)
    - current_query
    - new_evidence, new_analysis (optional lists)

The function returns a dict with "next_query" and "cycle_decisions" (and optionally "investigation_complete_reason").
"""
from __future__ import annotations
import json
import asyncio
import os
import re
from typing import Dict, Any, Optional
from datetime import datetime

# NOTE: this module references a CycleDecision structure in your codebase.
# If you have a dataclass/struct for CycleDecision, import it instead of the local helper below.
try:
    # project-local import; keep this if exists
    from state import CycleDecision  # type: ignore
except Exception:
    # Simple fallback dataclass for standalone testing / readability
    from dataclasses import dataclass, asdict

    @dataclass
    class CycleDecision:
        cycle_id: int
        query: Optional[str]
        decision_type: str
        reasoning: str
        timestamp: str
        evidence_count: int = 0
        analysis_count: int = 0

        def to_dict(self):
            return asdict(self)


# -------------------------
# Fallback / Test LLM
# -------------------------
class FakeLLM:
    """
    Deterministic small 'LLM' that inspects the graph and returns a JSON decision.
    Rules (mirrors refiner decision logic):
      - find the first AUTHOR node not in investigated_entities
        -> return continue with query "Investigate author: <author_id>"
      - otherwise return conclude with reasoning
    This is used only when a real llm isn't provided.
    """
    def __init__(self, graph_context: Dict[str, Any], investigated_entities: Optional[list] = None,
                 investigated_queries: Optional[list] = None, cycle_id: int = 0, max_cycles: int = 5):
        self.graph = graph_context or {}
        self.investigated_entities = set(e.lower() for e in (investigated_entities or []))
        self.investigated_queries = set(q.lower() for q in (investigated_queries or []))
        self.cycle_id = cycle_id
        self.max_cycles = max_cycles

    async def ainvoke(self, prompt: str) -> Any:
        """
        Return an object with a `.content` attribute (mimics a small async LLM response).
        """
        # Scan nodes for 'AUTHOR' not in investigated_entities
        authors = []
        for n in self.graph.get("nodes", []):
            if not isinstance(n, dict):
                continue
            if n.get("type") == "AUTHOR":
                authors.append(n.get("id"))

        next_author = None
        for a in authors:
            if a and a.lower() not in self.investigated_entities:
                next_author = a
                break

        if next_author:
            resp = {
                "decision": "continue",
                "next_query": f"Investigate author: {next_author}",
                "reasoning": f"Found uninvestigated AUTHOR node '{next_author}' in the graph. "
                             "Author accounts frequently provide leads (other posts, linked media, or accounts). "
                             "Investigating this author could reveal corroborating evidence or link to additional accounts."
            }
        else:
            resp = {
                "decision": "conclude",
                "next_query": None,
                "reasoning": "No unexplored AUTHOR nodes remain in the graph and no other unexplored high-priority entities detected. Concluding investigation."
            }

        class _Resp:
            def __init__(self, content: str):
                self.content = content

        return _Resp(content=json.dumps(resp))


# -------------------------
# Main refiner function
# -------------------------
async def enhanced_refinement_node(state: Dict[str, Any], llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Stateful refiner function.

    Args:
        state: dictionary with current investigation state (see docstring).
        llm: asynchronous LLM-like object with awaitable `.ainvoke(prompt)` -> returns object with `.content`.

    Returns:
        dict with keys: next_query, cycle_decisions, (maybe investigation_complete_reason)
    """
    # Extract state values
    initial_query = state.get("initial_query", "No initial query provided")
    graph_context = state.get("graph_context", {}) or {}
    cycle_id = int(state.get("cycle_id", 0))
    max_cycles = int(state.get("max_cycles", 5))
    investigated_queries = list(state.get("investigated_queries", []))
    investigated_entities = list(state.get("investigated_entities", []))
    cycle_decisions = list(state.get("cycle_decisions", []))
    evidence_count = len(state.get("new_evidence", []))
    analysis_count = len(state.get("new_analysis", []))

    # Early termination: max cycles
    if cycle_id >= max_cycles:
        reason = f"Terminating: reached max cycles ({max_cycles})."
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query"),
            decision_type="conclude",
            reasoning=reason,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        # update state
        if state.get("current_query"):
            investigated_queries.append(state["current_query"])
        state["current_query"] = None
        state["investigated_queries"] = investigated_queries
        state["cycle_decisions"] = cycle_decisions
        return {"next_query": None, "investigation_complete_reason": reason, "cycle_decisions": cycle_decisions}

    # No graph -> conclude
    if not graph_context or not graph_context.get("nodes"):
        reason = "Terminating: no relationship graph present."
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query"),
            decision_type="conclude",
            reasoning=reason,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        if state.get("current_query"):
            investigated_queries.append(state["current_query"])
        state["current_query"] = None
        state["investigated_queries"] = investigated_queries
        state["cycle_decisions"] = cycle_decisions
        return {"next_query": None, "investigation_complete_reason": reason, "cycle_decisions": cycle_decisions}

    # Summarize already investigated entities/queries (safe-lowercasing for comparisons)
    investigated_entities_set = set(e.lower() for e in investigated_entities)
    investigated_queries_set = set(q.lower() for q in investigated_queries)

    # Prepare prompt for an LLM (kept structured; LLM may be None and FakeLLM used)
    prompt = {
        "initial_query": initial_query,
        "investigated_summary": {
            "queries_investigated": investigated_queries,
            "entities_investigated": investigated_entities,
            "current_cycle": cycle_id,
            "max_cycles": max_cycles,
            "evidence_count": evidence_count,
            "analysis_count": analysis_count
        },
        "graph": graph_context
    }
    prompt_str = json.dumps(prompt, indent=2)

    # Choose LLM fallback if none provided
    if llm is None:
        llm = FakeLLM(graph_context, investigated_entities=investigated_entities, investigated_queries=investigated_queries, cycle_id=cycle_id, max_cycles=max_cycles)

    # Call LLM (async)
    try:
        response = await llm.ainvoke(prompt_str)
        content = getattr(response, "content", "") or response
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        content = content.strip()
        # Some LLMs may fence with ```json; remove those if present
        if content.startswith("```"):
            # strip any code fence markers
            content = re.sub(r"^```(?:json)?", "", content)
            content = content.rstrip("` \n")
        # Load JSON
        decision_data = json.loads(content)
    except Exception as e:
        # parsing/call error -> conclude as safety
        reason = f"Error invoking/parsing LLM response: {e}. Concluding for safety."
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query"),
            decision_type="conclude",
            reasoning=reason,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        if state.get("current_query"):
            investigated_queries.append(state["current_query"])
        state["current_query"] = None
        state["investigated_queries"] = investigated_queries
        state["cycle_decisions"] = cycle_decisions
        return {"next_query": None, "investigation_complete_reason": reason, "cycle_decisions": cycle_decisions}

    # Normalize decision fields
    decision_type = decision_data.get("decision", "conclude")
    next_query = decision_data.get("next_query")
    reasoning = decision_data.get("reasoning", "No reasoning provided.")

    # Loop prevention: if next_query duplicates investigated queries, or targets already-investigated entities -> conclude
    if next_query:
        normalized_q = str(next_query).lower().strip()
        if normalized_q in investigated_queries_set:
            # duplicate
            reasoning = f"Refiner proposed duplicate query '{next_query}' already in investigated_queries. Preventing loop and concluding."
            next_query = None
            decision_type = "conclude"
        else:
            # check for @mentions or "author:" pattern targeting already-investigated entities
            tokens = re.findall(r'(@\w+|\bauthor[: ]+\w+)', normalized_q)
            # Extract plain @name or author <name>
            targets = set()
            for t in tokens:
                t = t.replace("author", "").replace(":", "").strip()
                t = t.strip()
                if t:
                    targets.add(t)
            # if any target subset of investigated_entities -> likely redundant
            if targets and targets.issubset(investigated_entities_set):
                reasoning = f"Refiner proposed query targeting already-investigated entities {list(targets)}. Concluding."
                next_query = None
                decision_type = "conclude"

    # Record decision
    decision = CycleDecision(
        cycle_id=cycle_id,
        query=state.get("current_query"),
        decision_type=decision_type,
        reasoning=reasoning,
        timestamp=datetime.now().isoformat(),
        evidence_count=evidence_count,
        analysis_count=analysis_count
    )
    cycle_decisions.append(decision)

    # Update state: move current_query to investigated_queries, set new current_query accordingly
    if state.get("current_query"):
        investigated_queries.append(state["current_query"])
    state["investigated_queries"] = investigated_queries
    state["cycle_decisions"] = cycle_decisions

    if next_query:
        state["current_query"] = next_query
    else:
        state["current_query"] = None

    result = {"next_query": next_query, "cycle_decisions": cycle_decisions}
    if not next_query:
        result["investigation_complete_reason"] = reasoning
    return result


# -------------------------
# Convenience runner for CLI/testing
# -------------------------
def _build_initial_state_from_graph(graph_json: Dict[str, Any], max_cycles: int = 5) -> Dict[str, Any]:
    """
    Create a minimal state dict from a graph JSON (the verified files structure).
    """
    # Try to infer initial query from the filename or top-level fields (none present in verified files),
    # keep it generic if not present.
    return {
        "initial_query": "Check claims mentioning VerisTruth / VerisProject",
        "graph_context": graph_json.get("graph_context") if "graph_context" in graph_json else graph_json,
        "cycle_id": 0,
        "max_cycles": max_cycles,
        "investigated_queries": [],
        "investigated_entities": [],
        "current_query": None,
        "new_evidence": graph_json.get("original_evidence", []),
        "new_analysis": graph_json.get("new_analysis", []),
        "cycle_decisions": []
    }


async def run_refiner_from_file(json_path: str, max_cycles: int = 5, use_real_llm: bool = False):
    """
    Load a verified graph JSON file and run the refiner once (cycle_id == 0).
    Prints results to stdout.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    state = _build_initial_state_from_graph(data, max_cycles=max_cycles)
    # choose llm (None => FakeLLM inside function)
    llm = None
    # (If you want, you can pass in a real LLM object here instead of None.)

    result = await enhanced_refinement_node(state, llm=llm)
    # Pretty print decision(s)
    print("=== Refiner result ===")
    print(json.dumps({
        "state_summary": {
            "initial_query": state.get("initial_query"),
            "current_query": state.get("current_query"),
            "investigated_queries": state.get("investigated_queries"),
            "investigated_entities": state.get("investigated_entities"),
        },
        "refiner_result": {
            "next_query": result.get("next_query"),
            "investigation_complete_reason": result.get("investigation_complete_reason", ""),
            "cycle_decisions": [
                (c.to_dict() if hasattr(c, "to_dict") else c.__dict__) for c in result.get("cycle_decisions", [])
            ]
        }
    }, indent=2))
    return result


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m workflow.refiner /path/to/verified_graph.json")
        sys.exit(1)

    json_path = sys.argv[1]
    # run the async runner
    asyncio.run(run_refiner_from_file(json_path))
