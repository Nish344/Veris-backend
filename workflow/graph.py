#!/usr/bin/env python3
"""
workflow/graph.py

This file defines the `graph_node`, a dedicated step in the agent's workflow
responsible for synthesizing all collected evidence into a structured graph.
This graph represents the relationships between entities (authors, media) and
is used by the `refinement_node` for strategic decision-making.
"""
import json
import asyncio
from typing import List, Dict, Any, Set

# To run the test harness, we need access to the schema
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schema import EvidenceItem, MediaItem, MediaType, InvestigationCycle

# --- Graph Generation Utility ---

def generate_graph_from_evidence(all_evidence: List[EvidenceItem], all_analysis: List[Dict] = None) -> Dict[str, Any]:
    """
    Parses a list of EvidenceItem objects to create a graph of all known nodes
    and edges (relationships).
    """
    nodes: Set[tuple] = set()
    edges: List[Dict[str, Any]] = []

    for item in all_evidence:
        evidence_id = item.evidence_id
        nodes.add((evidence_id, "EVIDENCE"))
        
        author_id = item.author_id
        nodes.add((author_id, "AUTHOR"))
        edges.append({"source": author_id, "target": evidence_id, "label": "CREATED"})

        for mentioned in item.mentioned_accounts:
            nodes.add((mentioned, "AUTHOR"))
            edges.append({"source": evidence_id, "target": mentioned, "label": "MENTIONS"})

        for hashtag in item.hashtags:
            nodes.add((hashtag, "HASHTAG"))
            edges.append({"source": evidence_id, "target": hashtag, "label": "TAGS"})

        for media in item.media:
            # Ensure media is a MediaItem object to access attributes
            if isinstance(media, dict):
                media_obj = MediaItem(**media)
            else:
                media_obj = media
            
            # Use local_path as the media_id for the graph
            media_id = media_obj.local_path
            nodes.add((media_id, "MEDIA"))
            edges.append({"source": evidence_id, "target": media_id, "label": "INCLUDES"})

    if all_analysis:
        for anal in all_analysis:
            analysis_id = anal["analysis_id"]
            nodes.add((analysis_id, "ANALYSIS"))
            edge = {"source": analysis_id, "target": anal["evidence_id"], "label": "ANALYZES"}
            properties = {
                "trust_score": anal.get("trust_score"),
                "flag_reason": anal.get("flag_reason")
            }
            if properties:
                edge["properties"] = properties
            edges.append(edge)

    node_list = [{"id": node_id, "type": node_type} for node_id, node_type in nodes]
    return {"nodes": node_list, "edges": edges}


# --- Main Graph Node ---

async def graph_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregates all historical and new evidence to build a comprehensive
    relationship graph, which is then added to the agent's state.
    """
    print("--- Running Graph Node ---")

    all_evidence: List[EvidenceItem] = []
    
    # 1. Get historical evidence from previous cycles
    for cycle in state.get("investigation_cycles", []):
        all_evidence.extend(cycle.evidence_collected)
    
    # 2. Get new evidence from the current cycle (adjusted to use 'original_evidence' for verifier outputs)
    new_evidence_data = state.get("original_evidence", [])
    # Ensure items are EvidenceItem objects, not just dicts
    current_evidence = [EvidenceItem(**e) if isinstance(e, dict) else e for e in new_evidence_data]
    all_evidence.extend(current_evidence)

    # 3. Get new analysis if available
    all_analysis = state.get("new_analysis", [])

    if not all_evidence:
        print("Graph Node: No evidence to process.")
        return {"graph_context": {"nodes": [], "edges": []}}

    graph = generate_graph_from_evidence(all_evidence, all_analysis)
    
    print(f"Graph Node: Generated graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")

    return {"graph_context": graph}


# --- Standalone Test Harness (Batch Runner and State Saver) ---
if __name__ == '__main__':
    async def test_and_save_graph_states():
        """
        Loads all state files from the verifier's output, runs the graph_node
        on each, and saves the updated state to a new directory for the refiner.
        """
        VERIFIER_OUTPUT_DIR = "workflow/test_outputs_verifier"
        GRAPH_OUTPUT_DIR = "workflow/test_outputs_graph"
        os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
        
        if not os.path.exists(VERIFIER_OUTPUT_DIR):
            print(f"Error: Directory not found: {VERIFIER_OUTPUT_DIR}")
            return

        test_files = [f for f in os.listdir(VERIFIER_OUTPUT_DIR) if f.endswith(".json")]

        if not test_files:
            print(f"No test files found in {VERIFIER_OUTPUT_DIR}.")
            return

        print(f"--- Found {len(test_files)} test files to process ---")

        for test_file in test_files:
            print(f"\n=================================================")
            print(f"  Processing state file: {test_file}")
            print(f"=================================================")
            
            file_path = os.path.join(VERIFIER_OUTPUT_DIR, test_file)
            with open(file_path, 'r') as f:
                mock_state = json.load(f)

            # Simulate a "first cycle" scenario for each file
            mock_state["investigation_cycles"] = []
            
            # Run the graph_node with this specific state
            result = await graph_node(mock_state)

            # **UPDATE THE STATE** with the node's output
            mock_state.update(result)

            # **SAVE THE UPDATED STATE** to the new directory
            output_filename = test_file.replace("verifier", "graph")
            output_filepath = os.path.join(GRAPH_OUTPUT_DIR, output_filename)
            
            with open(output_filepath, 'w') as f:
                json.dump(mock_state, f, indent=2)

            print(f"  -> Saved updated state for refiner to: {output_filepath}")
            print(f"--- Processing for {test_file} complete ---")

    asyncio.run(test_and_save_graph_states())