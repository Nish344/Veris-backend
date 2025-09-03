# tools/tavily_tool.py - Refactored
import os
from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_core.tools import tool
from schema import EvidenceItem, SourceType

load_dotenv()

# Ensure TAVILY_API_KEY is loaded
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is not set in the .env file.")

@tool
def tavily_search_tool(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Perform a search using Tavily API and return results as EvidenceItem objects.
    """
    tavily = TavilySearchResults(max_results=max_results)
    results = tavily.invoke({"query": query})
    
    evidence_items = []
    for result in results:
        evidence_items.append(EvidenceItem(
            source_type=SourceType.WEB_SEARCH,
            url=result.get("url", ""),
            content=result.get("content", "No content available"),
            timestamp=datetime.now(),
            author_id="unknown",
            raw_data=result
        ))
    
    return evidence_items

# You can test this tool directly by running this file
if __name__ == '__main__':
    print("--- Testing Tavily Search Tool ---")
    
    query = "What is the latest news on the VerisProject?"
    results = tavily_search_tool.invoke({"query": query, "max_results": 5})
    
    print(f"Query: {query}")
    print("Results:")
    for item in results:
        print(item.model_dump_json(indent=2))
        print("-" * 20)