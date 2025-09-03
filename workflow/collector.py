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
# from tools.tavily_tool import tavily_search

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

    planner_prompt = f"""You are a highly intelligent query dispatcher. Your sole purpose is to analyze a complex user request and map it to a single, concrete data collection tool call.

Available Tools:
{tool_descriptions}
User Request: "{state['current_query']}"

Follow these steps to make your decision:
1.  Analyze the User Request to understand the core intent.
2.  If the request directly maps to a tool's description (e.g., "Get the profile for @user" maps to `get_author_profile`), select that tool and the appropriate arguments.
3.  If the request is more analytical or abstract (e.g., "Analyze interactions of @user" or "Find the origin of media_xyz"), your task is to first extract the primary entity from the request (e.g., "@user" or "media_xyz"). Then, select the best tool to gather MORE raw information about that entity. For general topics, authors, or media IDs, `search_and_scrape_x` is the best default tool.
4.  Your goal is to ALWAYS select a tool that can gather more raw data. Do not attempt to answer the user's query directly. Your only job is to invoke a tool.

Based on this logic, decide on the single tool to call to move the investigation forward.
"""
    
    # 1. Plan - Use the LLM to decide which tool to call
    plan_response: AIMessage = await llm_with_tools.ainvoke(planner_prompt)
    
    if not plan_response.tool_calls:
        print("Collector: Planner did not select a tool. Returning no new evidence.")
        return {"new_evidence": []}

    # 2. Execute - Run the selected tool
    print(f"Collector: Planner selected tool '{plan_response.tool_calls[0]['name']}' with args {plan_response.tool_calls[0]['args']}")
    
    # Pass the AIMessage from the planner inside a list to the ToolNode
    tool_output_messages: List[ToolMessage] = await tool_executor.ainvoke([plan_response])
    
    # 3. Format - Convert raw tool output into structured EvidenceItem objects
    raw_results_str = tool_output_messages[0].content
    tool_name = plan_response.tool_calls[0]['name'] # Get tool name from the original plan
    new_evidence = _format_tool_results(raw_results_str, tool_name)
    
    print(f"Collector: Formatted {len(new_evidence)} new evidence items.")
    return {"new_evidence": new_evidence}


# =============================================================================
# workflow/collector.py - CORRECTED _format_tool_results function
# =============================================================================

def _format_tool_results(results_str: str, tool_name: str):
    """
    Parses the raw JSON string from a tool into a list of EvidenceItem objects.
    CORRECTED: Now properly formats all fields according to EvidenceItem schema.
    """
    try:
        import json
        results = json.loads(results_str)
        if not isinstance(results, list):
            results = [results]
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse JSON from tool '{tool_name}'. Raw output: '{results_str[:200]}...'")
        return []

    # Updated source map to use lowercase values matching the SourceType enum in schema.py
    source_map = {
        "search_and_scrape_x": "twitter",
        "search_and_scrape_instagram": "instagram", 
        "scrape_single_page": "web_page",
        "get_author_profile": "twitter",
    }
    
    source_type = source_map.get(tool_name, "web_page")

    formatted_evidence = []
    for item in results:
        try:
            from schema import EvidenceItem, MediaItem
            from datetime import datetime
            
            # Handle slight variations in keys
            url = item.get("url") or item.get("post_url") or item.get("profile_url")
            
            # Format content based on tool type
            content = ""
            if tool_name == "get_author_profile":
                # For author profiles, create a readable summary but keep full data in raw_data
                profile_data = item
                username = profile_data.get("username", "unknown")
                display_name = profile_data.get("display_name", "")
                bio = profile_data.get("bio", "")
                followers = profile_data.get("followers_count", 0)
                following = profile_data.get("following_count", 0)
                posts_count = profile_data.get("posts_count", 0)
                
                content = f"Profile: @{username}"
                if display_name and display_name != f"@{username}":
                    content += f" ({display_name})"
                if bio:
                    content += f" - {bio}"
                content += f" | Followers: {followers}, Following: {following}, Posts: {posts_count}"
                
                # Include recent posts summary if available
                recent_posts = profile_data.get("recent_posts", [])
                if recent_posts:
                    content += f" | Recent activity: {len(recent_posts)} recent posts"
            else:
                # For other tools, use the text/content field
                content = item.get("content") or item.get("text") or ""

            # Media items need to be converted to MediaItem objects
            media_items = []
            for media_data in item.get("media", []):
                if isinstance(media_data, dict):
                    # Ensure media_data has required fields for MediaItem
                    media_dict = {
                        "media_type": media_data.get("media_type", "IMAGE"),
                        "url": media_data.get("url", ""),
                        **{k: v for k, v in media_data.items() if k in ["media_id", "local_path", "access_url"]}
                    }
                    media_items.append(MediaItem(**media_dict))
                else:
                    media_items.append(media_data)

            # Handle timestamp parsing more robustly
            timestamp = item.get("timestamp") or item.get("created_at") or item.get("account_created_at")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Handle different timestamp formats
                        timestamp = timestamp.replace('Z', '+00:00')
                        parsed_timestamp = datetime.fromisoformat(timestamp)
                    else:
                        parsed_timestamp = datetime.now()
                except ValueError:
                    parsed_timestamp = datetime.now()
            else:
                parsed_timestamp = datetime.now()

            # Extract mentioned accounts and hashtags
            mentioned_accounts = item.get("mentioned_accounts", [])
            hashtags = item.get("hashtags", [])
            
            # For profile tools, extract mentions/hashtags from recent posts
            if tool_name == "get_author_profile" and "recent_posts" in item:
                for post in item["recent_posts"]:
                    mentioned_accounts.extend(post.get("mentioned_accounts", []))
                    hashtags.extend(post.get("hashtags", []))
            
            # Remove duplicates while preserving order
            mentioned_accounts = list(dict.fromkeys(mentioned_accounts))
            hashtags = list(dict.fromkeys(hashtags))

            evidence = EvidenceItem(
                source_type=source_type,
                url=url,
                content=content or "",
                timestamp=parsed_timestamp,
                author_id=item.get("author_id") or item.get("username") or "unknown",
                media=media_items,
                mentioned_accounts=mentioned_accounts,
                hashtags=hashtags,
                raw_data=item  # Store the complete original data here
            )
            formatted_evidence.append(evidence)
            
        except Exception as e:
            print(f"Warning: Failed to format an item from '{tool_name}'. Error: {e}. Item: {item}")
            continue
            
    return formatted_evidence


# --- Test Harnesses ---

# Option 1: Standalone Test Runner
async def standalone_test_runner():
    print("\n--- Running Collector Standalone Test ---")
    
    tools = [search_and_scrape_x, scrape_single_page, get_author_profile]
    tool_executor = ToolNode(tools)
    llm_with_tools = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0).bind_tools(tools)
    
    mock_state = {
        "initial_query": "Standalone Test",
        "current_query": "scrape https://www.example.com"
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

    tools = [search_and_scrape_x, search_and_scrape_instagram, scrape_single_page, get_author_profile]
    tool_executor = ToolNode(tools)
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

