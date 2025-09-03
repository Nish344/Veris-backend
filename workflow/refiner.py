# =============================================================================
# workflow/refiner.py - CORRECTED VERSION
# =============================================================================
#!/usr/bin/env python3
"""
workflow/refiner.py

The strategic brain of the agent. This node reviews the synthesized graph
context from the `graph_node` and decides the next step in the investigation:
either to conclude or to formulate a new, targeted query.

UPDATED: Now includes anti-loop logic and better termination conditions.
"""
import json
import asyncio
from typing import Dict, Any

# For the test harness
import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from workflow.state import CycleDecision

# Add project root to path to allow importing schema
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schema import Investigation, InvestigationCycle

load_dotenv()

async def enhanced_refinement_node(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Analyzes the relationship graph to decide the next investigative step.
    Now includes detailed decision logging and reasoning.
    """
    print("--- Running Enhanced Refinement Node ---")
    from datetime import datetime
    import json
    import re

    initial_query = state.get("initial_query", "No initial query found.")
    graph_context = state.get("graph_context")
    cycle_id = state.get("cycle_id", 0)
    max_cycles = state.get("max_cycles", 5)
    investigated_queries = state.get("investigated_queries", [])
    investigated_entities = state.get("investigated_entities", [])
    cycle_decisions = state.get("cycle_decisions", [])
    
    evidence_count = len(state.get("new_evidence", []))
    analysis_count = len(state.get("new_analysis", []))

    # Early termination checks with detailed reasoning
    if cycle_id >= max_cycles:
        reason = f"Investigation terminated: Maximum cycles ({max_cycles}) reached. Completed {cycle_id} investigation cycles."
        print(f"Refinement Node: {reason}")
        
        # Log the decision
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query", ""),
            decision_type="conclude",
            reasoning=reason,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        
        return {
            "next_query": None, 
            "investigation_complete_reason": reason,
            "cycle_decisions": cycle_decisions
        }

    if not graph_context or not graph_context.get("nodes"):
        reason = "Investigation terminated: No relationship graph available. Unable to identify further investigative leads."
        print(f"Refinement Node: {reason}")
        
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query", ""),
            decision_type="conclude",
            reasoning=reason,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        
        return {
            "next_query": None, 
            "investigation_complete_reason": reason,
            "cycle_decisions": cycle_decisions
        }

    # Build context about what we've already investigated
    investigated_summary = {
        "queries_investigated": investigated_queries,
        "entities_investigated": investigated_entities,
        "current_cycle": cycle_id,
        "max_cycles": max_cycles,
        "evidence_collected_this_cycle": evidence_count,
        "analysis_completed_this_cycle": analysis_count
    }

    graph_json = json.dumps(graph_context, indent=2)
    investigated_json = json.dumps(investigated_summary, indent=2)

    prompt = f"""You are a lead investigator for a VIP protection unit.
Mission: "{initial_query}"

Current Investigation Status:
{investigated_json}

Relationship Graph (all known entities and relationships):
{graph_json}

CRITICAL INSTRUCTIONS:
1. You have already investigated the queries and entities listed above. DO NOT repeat identical queries.
2. You are on cycle {cycle_id} of maximum {max_cycles} cycles.
3. Look for NEW entities in the graph that haven't been explored OR entities with LOW TRUST SCORES that need deeper investigation.
4. If all promising leads have been exhausted, conclude the investigation.

Decision Rules:
- PRIORITY 1: If there are unexplored entities in the graph (authors/media not in investigated_entities), focus on the most suspicious/important one.
- PRIORITY 2: If there are entities from investigated_entities that generated evidence with very low trust scores (below 0.3), consider deeper investigation with a different angle or specific focus.
- PRIORITY 3: If all entities have been thoroughly explored AND no low-trust patterns need follow-up, conclude.
- If you're near the maximum cycles ({max_cycles}), be more selective and only continue for critical findings.
- Consider the evidence quality and trust scores from this cycle's analysis.

For entities with low trust scores, focus on:
- Specific claims they made that were flagged as "Unverified Claim"
- Their posting patterns or timing
- Connections to other suspicious entities
- Technical details they mentioned that could be verified

Your task is to decide the single most critical next step based ONLY on the graph and investigation history.

Respond with a JSON object:
- If concluding: {{"decision": "conclude", "next_query": null, "reasoning": "Detailed explanation of why you are concluding, including what was investigated and what was found"}}
- If continuing: {{"decision": "continue", "next_query": "Specific query about an unexplored entity OR deeper investigation of a low-trust entity", "reasoning": "Detailed explanation of why this specific entity is critical to investigate and what you hope to discover"}}

Requirements for continuing:
- The query must target a specific entity or aspect that provides NEW investigative value
- The query must be sufficiently different from previous queries to avoid loops
- Provide clear reasoning why this specific investigation is critical
- Explain what threat or concern this investigation might resolve

Your response must be a single, valid JSON object with detailed reasoning.
"""
    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.strip("```json").strip("```")
        
        decision_data = json.loads(content)
        
        next_query = decision_data.get("next_query")
        reasoning = decision_data.get("reasoning", "No reasoning provided.")
        decision_type = decision_data.get("decision", "conclude")

        # Additional validation to prevent loops
        if next_query:
            normalized_new_query = next_query.lower().strip()
            if normalized_new_query in investigated_queries:
                reasoning = f"Refiner proposed duplicate query '{next_query}' which was already investigated. Concluding to prevent infinite loop."
                print(f"Loop prevention: {reasoning}")
                next_query = None
                decision_type = "conclude"
            else:
                # Check if query targets already investigated entities
                entities_in_new_query = set(re.findall(r'@\w+', next_query.lower()))
                if entities_in_new_query.issubset(set(investigated_entities)):
                    reasoning = f"Refiner proposed query targeting already investigated entities {list(entities_in_new_query)}. All relevant leads exhausted."
                    print(f"Entity exhaustion: {reasoning}")
                    next_query = None
                    decision_type = "conclude"

        # Log the decision
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query", ""),
            decision_type=decision_type,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)

        print(f"Refiner Decision: {decision_type.upper()}")
        print(f"Reasoning: {reasoning}")
        
        if next_query:
            print(f"Next Query: \"{next_query}\"")
        else:
            print("Investigation will conclude.")
            
        result = {
            "next_query": next_query,
            "cycle_decisions": cycle_decisions
        }
        
        if not next_query:
            result["investigation_complete_reason"] = reasoning
            
        return result
        
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        reasoning = f"Error parsing refiner decision (JSON error: {e}). Concluding investigation as a safety measure."
        print(f"Error in refinement: {reasoning}")
        
        decision = CycleDecision(
            cycle_id=cycle_id,
            query=state.get("current_query", ""),
            decision_type="conclude",
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            evidence_count=evidence_count,
            analysis_count=analysis_count
        )
        cycle_decisions.append(decision)
        
        return {
            "next_query": None,
            "investigation_complete_reason": reasoning,
            "cycle_decisions": cycle_decisions
        }
