#!/usr/bin/env python3
"""
workflow/collector.py

Handles planning, tool execution, and formatting using REAL tools.
Includes a test harness that simulates the full AgentState flow,
producing complete state files ready for the verifier node.
"""
import json
import asyncio
import os
import re
from datetime import datetime
from typing import List, Dict, Any
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# --- Project-Specific Imports ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schema import EvidenceItem, MediaItem, MediaType, SourceType

# --- Import All Tools ---
from tools.playwright_tool import search_and_scrape_x, scrape_single_page
from tools.instagram_scrapper_tool import search_and_scrape_instagram
from tools.author_profiler_tool import get_author_profile
from tools.reddit_tool import enhanced_reddit_search, reddit_subreddit_analysis
from tools.tavily_tool import tavily_search_tool
from tools.search_tool import browser_search
from tools.browser_tool import enhanced_screenshot, screenshot_with_interaction
from tools.reverse_image_search_tool import reverse_image_search
from tools.network_graph_analyzer import analyze_collected_evidence

load_dotenv()

# --- Main Collector Node ---

async def collector_node(state: Dict[str, Any], tool_executor: ToolNode, llm_with_tools: Any) -> Dict[str, Any]:
    print(state)
    """
    Handles planning, tool execution, and formatting using REAL tools.
    """
    print(f"--- Running Collector for query: \"{state['current_query']}\" ---")
    
    # Dynamically build tool descriptions for the enhanced prompt
    tool_descriptions = ""
    # The tools are attached to the LLM object after binding
    for tool in getattr(llm_with_tools, 'tools', []):
        tool_descriptions += f"- Tool Name: `{tool.name}`\n"
        tool_descriptions += f"  - Description: {tool.description}\n\n"

    planner_prompt = f"""You are a highly intelligent query dispatcher. Your sole purpose is to analyze a complex user request and map it to as many concrete data collection tool calls as possible to gather the most comprehensive raw data available.

Available Tools:
{tool_descriptions}
the tool listed as "playwright" is used scrape x also called twitter
User Request: "{state['current_query']}"

Follow these steps to make your decision:
1. Analyze the User Request to understand the core intent and identify all relevant entities (e.g., usernames like @user1 @user2, hashtags like #hashtag, platforms like Instagram or X/Twitter, media IDs, etc.).
2. If the request directly maps to a tool's description (e.g., "Get the profile for @user" maps to `get_author_profile`), select that tool with the appropriate arguments. If there are multiple similar entities (e.g., multiple usernames), invoke the tool multiple times—once for each entity.
3. If the request is more analytical or abstract (e.g., "Analyze interactions of @user1 and @user2" or "Find the origin of media_xyz"), extract all primary entities from the request (e.g., "@user1", "@user2", or "media_xyz"). Then, select all possible tools that could gather MORE raw information about each entity. For general topics, authors, or media IDs, `search_and_scrape_x` is a good default tool, but always consider and include other platforms and tools like Instagram, Reddit, web searches, etc., if they could provide additional data.
4. If the request involves multiple platforms (e.g., Instagram and X/Twitter) or multiple entities, select and invoke all relevant tools to cover every aspect comprehensively (e.g., use `search_and_scrape_instagram` for Instagram usernames, `search_and_scrape_x` for X queries, `enhanced_reddit_search` for Reddit, etc.). Maximize coverage by using every tool that could plausibly return useful data.
5. Your goal is to ALWAYS select and invoke as many tools as possible that can gather more raw data across all identified entities and platforms. Do not hold back—err on the side of using more tools to ensure thorough data collection. Do not attempt to answer the user's query directly. Your only job is to invoke the appropriate tool(s)—invoke multiple (or all relevant) in one response.

Based on this logic, decide on the tool(s) to call to move the investigation forward. Generate as many tool calls as needed to maximize data gathering, especially for multiple entities or platforms.
"""
    
    # 1. Plan - Use the LLM to decide which tool(s) to call
    plan_response: AIMessage = await llm_with_tools.ainvoke(planner_prompt)
    
    if not plan_response.tool_calls:
        print("Collector: Planner did not select any tools. Returning no new evidence.")
        return {"new_evidence": []}

    # 2. Execute - Manually run the selected tools to get raw EvidenceItem lists
    print(f"Collector: Planner selected {len(plan_response.tool_calls)} tool calls.")
    new_evidence = []
    for tool_call in plan_response.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        print(f"Executing tool '{tool_name}' with args: {args}")
        
        tool = tool_dict.get(tool_name)
        if tool:
            try:
                if hasattr(tool, 'ainvoke'):
                    result = await tool.ainvoke(args)
                else:
                    result = tool.invoke(args)
                
                if isinstance(result, list):
                    new_evidence.extend(result)
                else:
                    new_evidence.append(result)
            except Exception as e:
                print(f"Error executing tool '{tool_name}': {e}")
        else:
            print(f"Tool '{tool_name}' not found.")

    print(f"Collector: Collected {len(new_evidence)} new evidence items.")
    return {"new_evidence": new_evidence}

# Global tool list and dict for lookup
tools = [
    search_and_scrape_x,
    scrape_single_page,
    search_and_scrape_instagram,
    get_author_profile,
    enhanced_reddit_search,
    reddit_subreddit_analysis,
    tavily_search_tool,
    browser_search,
    enhanced_screenshot,
    screenshot_with_interaction,
    reverse_image_search,
    analyze_collected_evidence  # Included, but use cautiously as it requires evidence file
]
tool_dict = {t.name: t for t in tools}

# --- Test Harnesses ---

# Option 1: Standalone Test Runner
async def standalone_test_runner():
    print("\n--- Running Collector Standalone Test ---")
    
    tool_executor = ToolNode(tools)  # Still used if needed, but not for execution
    llm_with_tools = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0).bind_tools(tools)
    
    mock_state = {
        "initial_query": "Standalone Test",
        "current_query": "scrape instagram user @laal_bandar05 and #VerisProject on x"
    }
    
    result = await collector_node(mock_state, tool_executor, llm_with_tools)
    print("\nStandalone Test Output:")
    print(json.dumps([item.model_dump() for item in result.get("new_evidence", [])], indent=2, default=str))
    assert len(result.get("new_evidence", [])) > 0
    print("\n--- Standalone Test PASSED ---")


# Option 2: Test Runner for Refiner Loop
async def refiner_loop_test_runner():
    print("\n--- Running Collector Test on Refiner's Output ---")
    
    REFINER_OUTPUT_DIR = "workflow/test_outputs_refiner"
    if not os.path.exists(REFINER_OUTPUT_DIR):
        print(f"Error: Directory not found: {REFINER_OUTPUT_DIR}")
        return

    test_files = [f for f in os.listdir(REFINER_OUTPUT_DIR) if f.endswith(".json")]
    if not test_files:
        print(f"No test files found in {REFINER_OUTPUT_DIR}.")
        return

    tool_executor = ToolNode(tools)  # Still used if needed
    llm_with_tools = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0).bind_tools(tools)

    for test_file in test_files:
        print(f"\n=================================================")
        print(f"  Testing Collector with state from: {test_file}")
        print(f"=================================================")

        with open(os.path.join(REFINER_OUTPUT_DIR, test_file), 'r') as f:
            refiner_state = json.load(f)

        next_query = refiner_state.get("next_query")

        if not next_query:
            print("  -> Refiner decided to conclude. Skipping Collector.")
            continue
        
        # Prepare the state for the next cycle
        next_cycle_state = {
            "initial_query": refiner_state["initial_query"],
            "current_query": next_query, # Use the refiner's output as the new query
            "investigation_cycles": refiner_state.get("investigation_cycles", []),
            # In a real run, old evidence would be archived here. For the test, we start fresh.
            "new_evidence": [],
            "new_analysis": []
        }
        
        # Run the collector with the new state
        result = await collector_node(next_cycle_state, tool_executor, llm_with_tools)
        
        print(f"\n  Collector produced {len(result.get('new_evidence', []))} new evidence items.")
        assert "new_evidence" in result
        print(f"--- Collector Test for {test_file} PASSED ---")


if __name__ == '__main__':
    # You can run either or both tests
    asyncio.run(standalone_test_runner())
    # asyncio.run(refiner_loop_test_runner())