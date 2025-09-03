#!/usr/bin/env python3
"""
workflow/verifier.py

A highly reliable verifier node that uses an XML-based extraction strategy
and includes a rule-based fallback to ensure the workflow never halts.
- Smartly truncates large evidence items (like author profiles) for efficiency.
- Test harness loads full state from collector, updates it, and saves it for the refiner.
"""

import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import ValidationError

# --- Project-Specific Imports ---
import sys
# Add the project root to the path to allow importing 'schema'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schema import VerificationResult, EvidenceItem, SourceType

load_dotenv()

# --- Configuration ---
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def _extract_json_from_xml(text: str) -> str:
    """Extracts a JSON string from within <json> tags."""
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for cases where the model forgets the XML tags but still returns JSON
    if text.strip().startswith(('{', '[')):
        return text.strip()
    raise ValueError("No <json> tag found in the LLM response.")

def _rule_based_score(item: EvidenceItem) -> VerificationResult:
    """Provides a fallback score based on keywords if the LLM fails."""
    text = (item.content or "").lower()
    score = 0.5
    reason = "Fallback Heuristic"
    
    if any(kw in text for kw in ["official", "press release", "report", "verified"]):
        score = 0.8
        reason = "Heuristic: Official Language"
    elif any(kw in text for kw in ["unverified", "rumor", "alleged", "speculation"]):
        score = 0.2
        reason = "Heuristic: Unverified Language"
    elif item.source_type == SourceType.TWITTER and "Full profile analysis" in item.content:
        score = 0.7
        reason = "Heuristic: Profile Analysis"
        
    return VerificationResult(
        evidence_id=item.evidence_id,
        trust_score=score,
        flag_reason=reason,
        reasoning=f"Rule-based fallback applied to evidence item {item.evidence_id}."
    )

def _summarize_author_profile(item: EvidenceItem) -> str:
    """Creates a concise summary of an author profile for the LLM."""
    try:
        json_str_match = re.search(r'(\{.*\})', item.content, re.DOTALL)
        if not json_str_match:
            return item.content[:500]
            
        profile_data = json.loads(json_str_match.group(1))
        
        summary = (
            f"Profile for @{profile_data.get('username')}: "
            f"Bio: '{profile_data.get('bio')}'. "
            f"Followers: {profile_data.get('followers_count')}. "
            f"Following: {profile_data.get('following_count')}. "
        )
        
        recent_posts = profile_data.get('recent_posts', [])
        if recent_posts:
            summary += "Recent post texts: "
            post_texts = [f"\"{post.get('text', '')}\"" for post in recent_posts]
            summary += "; ".join(post_texts)
            
        return summary[:1000] # Limit summary size
    except Exception as e:
        logger.warning(f"Could not summarize author profile {item.evidence_id}, using raw content. Error: {e}")
        return item.content[:500]


# --- Main Verifier Node ---
async def verifier_node(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Analyzes new evidence, assigning a trust score using a reliable XML-based method
    with a rule-based fallback for maximum resilience.
    """
    logger.info("--- Running Verifier Node ---")
    
    new_evidence: List[EvidenceItem] = [
        EvidenceItem(**e) if isinstance(e, dict) else e 
        for e in state.get("new_evidence", [])
    ]

    if not new_evidence:
        logger.info("No new evidence to verify.")
        return {"new_analysis": []}

    prompt_items = []
    for item in new_evidence:
        content_for_prompt = item.content
        # Smart truncation for large author profiles
        if item.source_type == SourceType.TWITTER and item.content.startswith("Full profile analysis"):
             content_for_prompt = _summarize_author_profile(item)
        else:
            # Truncate other content to a reasonable length
            content_for_prompt = (item.content or '')[:500]

        prompt_items.append(
            f"Evidence ID: {item.evidence_id}\n"
            f"Author: {item.author_id}\n"
            f"Source: {item.source_type}\n"
            f"Content: \"{content_for_prompt}\""
        )
    evidence_for_prompt = "\n---\n".join(prompt_items)


    prompt = f"""You are a senior threat analyst. Your task is to analyze the following batch of evidence.

Your response MUST be a single JSON object wrapped in <json> tags. The JSON object must have a key "analyses", which is a list of objects. Each object in the list must contain:
- "evidence_id": The exact ID of the evidence item.
- "reasoning": A brief, one-sentence explanation for your trust score.
- "trust_score": A float between 0.0 (untrustworthy) and 1.0 (trustworthy).
- "flag_reason": A short category (e.g., "Unverified Claim", "Opinion", "Factual Report", "Profile Analysis").

--- BATCH OF EVIDENCE ---
{evidence_for_prompt}
--- END OF BATCH ---
"""

    analysis_results: List[VerificationResult] = []
    try:
        logger.info(f"Calling LLM to verify {len(new_evidence)} items...")
        response = await llm.ainvoke(prompt)
        json_string = _extract_json_from_xml(response.content)
        parsed_data = json.loads(json_string)

        if "analyses" not in parsed_data or not isinstance(parsed_data["analyses"], list):
            raise ValueError("Parsed JSON is missing the 'analyses' list.")

        valid_evidence_ids = {item.evidence_id for item in new_evidence}
        for analysis_item in parsed_data["analyses"]:
            if analysis_item.get("evidence_id") in valid_evidence_ids:
                try:
                    result = VerificationResult(**analysis_item)
                    analysis_results.append(result)
                except ValidationError as ve:
                    logger.warning(f"Validation failed for item: {analysis_item}. Error: {ve}")
            else:
                logger.warning(f"LLM returned analysis for unknown evidence_id: {analysis_item.get('evidence_id')}")

    except Exception as e:
        logger.error(f"Failed to process LLM response: {e}. Applying rule-based fallbacks.")
        for item in new_evidence:
            analysis_results.append(_rule_based_score(item))

    logger.info(f"Successfully created {len(analysis_results)} verification results.")
    return {"new_analysis": [res.model_dump() for res in analysis_results]}


# --- Standalone Test Harness ---
if __name__ == '__main__':
    
    COLLECTOR_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
    VERIFIER_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs_verifier")
    os.makedirs(VERIFIER_OUTPUT_DIR, exist_ok=True)

    async def test_verifier_with_collector_states():
        # LLM instance for the test
        llm_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

        if not os.path.exists(COLLECTOR_OUTPUT_DIR):
            print(f"Directory not found: {COLLECTOR_OUTPUT_DIR}")
            return

        test_files = [f for f in os.listdir(COLLECTOR_OUTPUT_DIR) if f.startswith("state_after_collector_") and f.endswith(".json")]

        for test_file in test_files:
            print(f"\n--- Testing Verifier with state from: {test_file} ---")
            filepath = os.path.join(COLLECTOR_OUTPUT_DIR, test_file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    state_from_collector = json.load(f)
                
                if not state_from_collector.get("new_evidence"):
                    print("No new evidence in this state file. Skipping.")
                    continue
                
                # Run the verifier node with the loaded state and LLM
                verifier_output = await verifier_node(state_from_collector, llm_instance)
                
                state_from_collector["new_analysis"] = verifier_output.get("new_analysis", [])
                
                output_filename = test_file.replace("collector", "verifier")
                output_filepath = os.path.join(VERIFIER_OUTPUT_DIR, output_filename)

                with open(output_filepath, 'w') as f:
                    json.dump(state_from_collector, f, indent=2, default=str)
                print(f"--- Saved final state for Refiner to: {output_filepath} ---")

            except Exception as e:
                print(f"An error occurred while testing with {test_file}: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(test_verifier_with_collector_states())
