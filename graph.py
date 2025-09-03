#!/usr/bin/env python3
"""
graph.py

This file defines and exposes the main LangGraph object for the investigative agent.
It is structured to be compatible with the `langgraph dev` command-line tool,
and can also be run directly to execute a full investigation.
"""
import asyncio
import functools
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Schemas and State ---
from schema import Investigation, InvestigationCycle, EvidenceItem, VerificationResult
from workflow.state import AgentState

# --- All Tools ---
from tools.playwright_tool import search_and_scrape_x, scrape_single_page
from tools.instagram_scrapper_tool import search_and_scrape_instagram
from tools.author_profiler_tool import get_author_profile

# --- Node Imports ---
from workflow.collector import collector_node
from workflow.verifier import verifier_node
from workflow.graph import graph_node as graph_builder_node # Renamed to avoid conflict
from workflow.refiner import enhanced_refinement_node
from workflow.reporter import final_reporter_node

load_dotenv()


# --- LLM and Tool Configuration ---
def create_llm_with_tools(tools: List[Any]) -> Any:
    """Binds the tools to the LLM for the collector's planner."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0
    ).bind_tools(tools)

def create_analysis_llm() -> Any:
    """Creates the LLM for analysis nodes, configured for JSON output."""
    # FIX: Pass generation_config directly to remove the warning
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        generation_config={"response_mime_type": "application/json"}
    )


# --- State Management Node ---
def enhanced_archive_and_prepare_for_next_cycle(state: AgentState) -> Dict[str, Any]:
    """
    Archives the completed cycle's data and returns ONLY the fields that need to be
    updated for the next loop. Now includes termination checks and decision tracking.
    """
    print("--- ARCHIVING AND PREPARING FOR NEXT CYCLE ---")
    from datetime import datetime
    import re
    
    # Check if we've hit the maximum cycles
    if state['cycle_id'] >= state.get('max_cycles', 5):
        reason = f"Maximum cycles ({state.get('max_cycles', 5)}) reached. Investigation terminated."
        print(reason)
        return {
            "next_query": None,
            "investigation_complete_reason": reason
        }
    
    # Check if the next query has already been investigated
    next_query = state.get('next_query')
    if next_query:
        investigated_queries = state.get('investigated_queries', [])
        normalized_query = next_query.lower().strip()
        
        if normalized_query in investigated_queries:
            reason = f"Query '{next_query}' already investigated. Preventing infinite loop."
            print(reason)
            return {
                "next_query": None,
                "investigation_complete_reason": reason
            }

    from schema import InvestigationCycle, EvidenceItem, VerificationResult
    
    completed_cycle = InvestigationCycle(
        cycle_id=state['cycle_id'],
        query=state['current_query'], 
        start_time=datetime.now(),
        end_time=datetime.now(),
        evidence_collected=[EvidenceItem(**e) if isinstance(e, dict) else e for e in state.get("new_evidence", [])],
        analysis_results=[VerificationResult(**a) if isinstance(a, dict) else a for a in state.get("new_analysis", [])]
    )
    
    investigation_cycles = state.get("investigation_cycles", [])
    investigation_cycles.append(completed_cycle)
    
    # Update tracking lists
    investigated_queries = list(set(state.get('investigated_queries', [])))
    current_query_normalized = state['current_query'].lower().strip()
    if current_query_normalized not in investigated_queries:
        investigated_queries.append(current_query_normalized)
    
    # Extract entities from the current query and evidence for tracking
    investigated_entities = list(set(state.get('investigated_entities', [])))
    
    # Add entities mentioned in current query
    entities_in_query = re.findall(r'@\w+', state['current_query'])
    for entity in entities_in_query:
        entity_normalized = entity.lower()
        if entity_normalized not in investigated_entities:
            investigated_entities.append(entity_normalized)
    
    # Add entities from evidence
    for evidence_item in state.get("new_evidence", []):
        if isinstance(evidence_item, dict):
            evidence_item = EvidenceItem(**evidence_item)
        if evidence_item.author_id and evidence_item.author_id != "unknown":
            author_normalized = evidence_item.author_id.lower()
            if author_normalized not in investigated_entities:
                investigated_entities.append(author_normalized)
        for mention in evidence_item.mentioned_accounts:
            mention_normalized = mention.lower()
            if mention_normalized not in investigated_entities:
                investigated_entities.append(mention_normalized)
    
    print(f"Cycle {state['cycle_id']} complete. Moving to cycle {state['cycle_id'] + 1}")
    print(f"Next query: {next_query}")
    
    return {
        "investigation_cycles": investigation_cycles,
        "cycle_id": state['cycle_id'] + 1,
        "current_query": state['next_query'],
        "new_evidence": [],
        "new_analysis": [],
        "graph_context": None,
        "next_query": None,
        "investigated_queries": investigated_queries,
        "investigated_entities": investigated_entities,
        "cycle_decisions": state.get("cycle_decisions", []),  # Preserve decision history
    }

def generate_final_conclusion(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a comprehensive final conclusion based on all investigation data.
    """
    from datetime import datetime
    
    cycle_decisions = state.get("cycle_decisions", [])
    investigation_cycles = state.get("investigation_cycles", [])
    complete_reason = state.get("investigation_complete_reason", "Investigation completed")
    
    # Analyze investigation results
    total_evidence = sum(len(cycle.evidence_collected) for cycle in investigation_cycles)
    total_analysis = sum(len(cycle.analysis_results) for cycle in investigation_cycles)
    
    # Calculate threat assessment from analysis results
    all_trust_scores = []
    for cycle in investigation_cycles:
        for analysis in cycle.analysis_results:
            if hasattr(analysis, 'trust_score'):
                all_trust_scores.append(analysis.trust_score)
            elif isinstance(analysis, dict):
                all_trust_scores.append(analysis.get('trust_score', 0.5))
    
    avg_trust_score = sum(all_trust_scores) / len(all_trust_scores) if all_trust_scores else 0.5
    
    # Determine threat level
    if avg_trust_score >= 0.7:
        threat_level = "LOW"
        threat_summary = "Evidence shows mostly trustworthy sources with limited concerning patterns."
    elif avg_trust_score >= 0.4:
        threat_level = "MODERATE" 
        threat_summary = "Mixed evidence quality with some concerning patterns requiring monitoring."
    else:
        threat_level = "HIGH"
        threat_summary = "Low trust scores indicate potential threats or misinformation patterns."
    
    final_conclusion = {
        "investigation_summary": {
            "initial_query": state.get("initial_query", ""),
            "total_cycles_completed": len(investigation_cycles),
            "total_evidence_collected": total_evidence,
            "total_analysis_completed": total_analysis,
            "entities_investigated": state.get("investigated_entities", []),
            "completion_reason": complete_reason,
            "completion_timestamp": datetime.now().isoformat()
        },
        "threat_assessment": {
            "threat_level": threat_level,
            "average_trust_score": round(avg_trust_score, 3),
            "summary": threat_summary
        },
        "investigation_timeline": [
            {
                "cycle": decision["cycle_id"],
                "query": decision["query"],
                "decision": decision["decision_type"],
                "reasoning": decision["reasoning"],
                "evidence_found": decision["evidence_count"],
                "timestamp": decision["timestamp"]
            }
            for decision in cycle_decisions
        ],
        "recommendations": [
            "Monitor identified entities for ongoing activity" if total_evidence > 0 else "No specific entities require monitoring",
            f"Trust level: {threat_level} - {'Immediate action recommended' if threat_level == 'HIGH' else 'Continued monitoring sufficient'}",
            f"Investigation depth: {'Comprehensive' if len(investigation_cycles) >= 3 else 'Limited'} - {len(investigation_cycles)} cycles completed"
        ]
    }
    
    return {"final_conclusion": final_conclusion}


# --- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
    """The router of the graph. Decides whether to continue or end."""
    if state.get("next_query"):
        print("--- DECISION: CONTINUE ---")
        return "continue"
    else:
        print("--- DECISION: CONCLUDE ---")
        return "end"

# --- Graph Definition ---

# 1. Instantiate tools and LLMs
tools = [search_and_scrape_x, scrape_single_page, search_and_scrape_instagram, get_author_profile]
tool_executor = ToolNode(tools)
llm_with_tools = create_llm_with_tools(tools)
analysis_llm = create_analysis_llm()

# 2. Create partials to bind LLMs/tools to the node functions
bound_collector_node = functools.partial(collector_node, tool_executor=tool_executor, llm_with_tools=llm_with_tools)
bound_verifier_node = functools.partial(verifier_node, llm=analysis_llm)
bound_refiner_node = functools.partial(enhanced_refinement_node, llm=analysis_llm)
bound_reporter_node = functools.partial(final_reporter_node, llm=analysis_llm)

# 3. Define the graph structure
workflow = StateGraph(AgentState)

workflow.add_node("collector", bound_collector_node)
workflow.add_node("verifier", bound_verifier_node)
workflow.add_node("graph_builder", graph_builder_node)
workflow.add_node("refiner", bound_refiner_node)
workflow.add_node("state_synchronizer", enhanced_archive_and_prepare_for_next_cycle)
workflow.add_node("reporter", bound_reporter_node)

workflow.set_entry_point("collector")
workflow.add_edge("collector", "verifier")
workflow.add_edge("verifier", "graph_builder")
workflow.add_edge("graph_builder", "refiner")

workflow.add_conditional_edges(
    "refiner",
    should_continue,
    {"continue": "state_synchronizer", "end": "reporter"}
)

workflow.add_edge("state_synchronizer", "collector")
workflow.add_edge("reporter", END)

# 4. Expose the compiled graph as a top-level variable named "graph"
graph = workflow.compile()


async def enhanced_run_investigation(query: str, max_cycles: int = 5):
    """
    Runs a full investigation with enhanced decision tracking and final conclusions.
    """
    print("--- Starting Enhanced Investigation ---")
    
    from datetime import datetime
    
    # Initialize state with decision tracking
    initial_state = {
        "initial_query": query,
        "current_query": query,
        "investigation_id": f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "cycle_id": 0,
        "messages": [],
        "new_evidence": [],
        "new_analysis": [],
        "graph_context": None,
        "investigation_cycles": [],
        "next_query": None,
        "final_summary": None,
        "investigated_queries": [],
        "investigated_entities": [],
        "max_cycles": max_cycles,
        "investigation_complete_reason": None,
        "cycle_decisions": [],  # Track all decision reasoning
        "final_conclusion": None
    }
    
    # Run the investigation (assuming 'graph' is your compiled workflow)
    # final_state = await graph.ainvoke(initial_state)
    
    # For demonstration, simulate the final state processing
    final_state = await graph.ainvoke(initial_state)
    
    # Generate comprehensive final conclusion
    conclusion_data = generate_final_conclusion(final_state)
    final_state.update(conclusion_data)
    
    # Save results
    import os, json
    output_dir = "investigation_reports"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{final_state['investigation_id']}.json")
    
    with open(file_path, 'w') as f:
        json.dump(final_state, f, indent=2, default=str)
    
    # Print comprehensive summary
    print(f"\n--- Investigation Complete ---")
    print(f"ID: {final_state['investigation_id']}")
    print(f"Reason: {final_state.get('investigation_complete_reason', 'Natural conclusion')}")
    print(f"Total Cycles: {len(final_state.get('investigation_cycles', []))}")
    print(f"Evidence Collected: {sum(len(cycle.evidence_collected) for cycle in final_state.get('investigation_cycles', []))}")
    print(f"Entities Investigated: {final_state.get('investigated_entities', [])}")
    
    if final_state.get('final_conclusion'):
        conclusion = final_state['final_conclusion']
        print(f"Threat Level: {conclusion['threat_assessment']['threat_level']}")
        print(f"Average Trust Score: {conclusion['threat_assessment']['average_trust_score']}")
    
    print(f"Detailed Report: {file_path}")
    return final_state


async def main():
    await enhanced_run_investigation(
        query="Investigate data integrity issues at veris truth  [X : @VerisTruth and #VerisProject]"
    )

if __name__ == "__main__":
    asyncio.run(main())

