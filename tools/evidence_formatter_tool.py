import json
import uuid
import logging
from typing import List, Dict, Any, Union
from datetime import datetime
from langchain.tools import tool
from schema import EvidenceItem

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PARSER FUNCTIONS ---

def _parse_playwright_scrape(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parses the output from the playwright_tool's scrape_single_page function."""
    if not isinstance(data, dict) or "url" not in data:
        return []
    
    return [
        {
            "evidence_id": f"web_{str(uuid.uuid4())[:8]}",
            "source_type": "web_page",
            "url": data.get("url", ""),
            "content": data.get("content", "No content available")[:2000], # Truncate for safety
            "timestamp": datetime.now().isoformat(),
            "author": data.get("author", "Unknown"),
            "screenshot_path": data.get("screenshot_path", "")
        }
    ]

def _parse_x_search_result(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parses the output from the new search_and_scrape_x tool."""
    if not isinstance(data, list):
        return []

    evidence_list = []
    for item in data:
        if "error" in item:
            continue
            
        # FIX: Corrected keys to match the output of search_and_scrape_x
        author = item.get("author", "Unknown")
        evidence_list.append({
            "evidence_id": f"x_{str(uuid.uuid4())[:8]}",
            "source_type": "twitter", # Use 'twitter' to match schema
            "url": item.get("url", ""),
            "content": item.get("content", "No content available"),
            "timestamp": datetime.now().isoformat(), # Note: We could parse tweet date if needed
            "author": author,
            "metadata": {"username": author} # Use the same author field for username metadata
        })
    return evidence_list
    

# --- MAIN FORMATTER TOOL ---

SOURCE_PARSERS = {
    "scrape_single_page": _parse_playwright_scrape,
    "search_and_scrape_x": _parse_x_search_result,
}

@tool
def format_raw_data_into_evidence(raw_data: Union[Dict[str, Any], List[Dict[str, Any]]], source: str) -> str:
    """
    Parses raw data from different scraper tools into a list of structured EvidenceItem objects.
    This tool is essential for standardizing collected data before saving.
    'source' must be one of the known scraper tool names.
    Returns a JSON string of the list of evidence items.
    """
    parser = SOURCE_PARSERS.get(source)
    if not parser:
        return json.dumps({"error": f"No parser available for source: {source}"})

    try:
        parsed_items = parser(raw_data)
        # Validate with Pydantic and convert back to dict for JSON serialization
        # This ensures the data conforms to the EvidenceItem schema.
        validated_evidence_models = [EvidenceItem(**item) for item in parsed_items]
        # Use Pydantic's recommended model_dump with mode='json' for safe serialization
        validated_evidence_dicts = [model.model_dump(mode='json') for model in validated_evidence_models]

        return json.dumps(validated_evidence_dicts)
    except Exception as e:
        logging.error(f"Failed to parse or validate data from source '{source}': {e}")
        return json.dumps({"error": f"Data from source '{source}' was malformed. Details: {e}"})
