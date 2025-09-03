#!/usr/bin/env python3
"""
workflow/verifier.py

Enhanced verifier node with improved verification logic including:
- Content relevance analysis
- Cross-reference detection
- Error handling improvements
- Enhanced platform-specific scoring
- Metadata analysis
- Sentiment and reliability assessment
- Direct JSON array input support (matches your collector output format)
"""

import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import ValidationError

# --- Project-Specific Imports ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schema import VerificationResult, EvidenceItem, SourceType

load_dotenv()

# --- Configuration ---
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# --- Enhanced Helper Functions ---
def _extract_json_from_xml(text: str) -> str:
    """Extracts a JSON string from within <json> tags."""
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for cases where the model forgets the XML tags but still returns JSON
    if text.strip().startswith(('{', '[')):
        return text.strip()
    raise ValueError("No <json> tag found in the LLM response.")

def _analyze_content_quality(content: str) -> Dict[str, Any]:
    """Analyzes content for quality indicators."""
    if not content:
        return {"quality_score": 0.1, "indicators": ["empty_content"]}
    
    content_lower = content.lower()
    indicators = []
    quality_score = 0.5  # Base score
    
    # Positive indicators
    if any(word in content_lower for word in ["verified", "official", "confirmed", "reported", "announced"]):
        quality_score += 0.2
        indicators.append("authoritative_language")
    
    if any(word in content_lower for word in ["data", "analysis", "research", "study", "report"]):
        quality_score += 0.15
        indicators.append("data_oriented")
    
    if re.search(r'\d{4}', content):  # Contains years
        quality_score += 0.1
        indicators.append("temporal_context")
    
    if len(content.split()) > 50:  # Substantial content
        quality_score += 0.1
        indicators.append("detailed_content")
    
    # Negative indicators
    if any(word in content_lower for word in ["rumor", "alleged", "unverified", "speculation", "opinion"]):
        quality_score -= 0.2
        indicators.append("speculative_language")
    
    if "error occurred" in content_lower or "failed" in content_lower:
        quality_score -= 0.4
        indicators.append("error_content")
    
    if any(word in content_lower for word in ["seems like", "appears", "might", "possibly", "maybe"]):
        quality_score -= 0.1
        indicators.append("uncertain_language")
    
    return {
        "quality_score": max(0.0, min(1.0, quality_score)),
        "indicators": indicators
    }

def _analyze_cross_references(evidence_items: List[Dict]) -> Dict[str, Set[str]]:
    """Analyzes cross-references between evidence items."""
    cross_refs = defaultdict(set)
    
    # Extract common themes, entities, hashtags
    for item in evidence_items:
        evidence_id = item.get("evidence_id")
        content = item.get("content", "").lower()
        
        # Look for common entities or projects
        if "veris" in content:
            cross_refs["veris_project"].add(evidence_id)
        
        if "data" in content and any(word in content for word in ["leak", "breach", "integrity", "security"]):
            cross_refs["data_security"].add(evidence_id)
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', content)
        for tag in hashtags:
            cross_refs[f"hashtag_{tag}"].add(evidence_id)
        
        # Extract mentions
        mentions = re.findall(r'@\w+', content)
        for mention in mentions:
            cross_refs[f"mention_{mention}"].add(evidence_id)
    
    return cross_refs

def _enhanced_rule_based_score(item: Dict, cross_refs: Dict[str, Set[str]], quality_analysis: Dict) -> VerificationResult:
    """Enhanced rule-based scoring with improved logic."""
    evidence_id = item.get("evidence_id")
    content = item.get("content", "")
    source_type = item.get("source_type", "").lower()
    author_id = item.get("author_id", "")
    has_media = len(item.get("media", [])) > 0
    
    # Start with quality analysis score
    base_score = quality_analysis.get("quality_score", 0.5)
    reason_parts = []
    
    # Platform-specific adjustments
    if source_type == "twitter":
        if "error occurred" in content.lower():
            base_score = 0.1
            reason_parts.append("Twitter scraping error")
        elif has_media:
            base_score += 0.1
            reason_parts.append("Contains media")
        elif any(ref_set for ref_set in cross_refs.values() if evidence_id in ref_set and len(ref_set) > 1):
            base_score += 0.15
            reason_parts.append("Cross-referenced content")
    
    elif source_type == "instagram":
        if has_media:
            base_score += 0.1
            reason_parts.append("Visual evidence present")
        
        # Check for promotional vs informational content
        if any(word in content.lower() for word in ["ad", "sponsored", "promotion"]):
            base_score -= 0.2
            reason_parts.append("Promotional content")
        elif any(word in content.lower() for word in ["research", "science", "project", "development"]):
            base_score += 0.15
            reason_parts.append("Educational/informational content")
    
    # Cross-reference bonus
    cross_ref_count = sum(1 for ref_set in cross_refs.values() if evidence_id in ref_set and len(ref_set) > 1)
    if cross_ref_count > 0:
        base_score += min(0.2, cross_ref_count * 0.1)
        reason_parts.append(f"Referenced in {cross_ref_count} themes")
    
    # Author credibility (basic heuristics)
    if author_id and author_id != "unknown":
        if any(word in author_id.lower() for word in ["official", "news", "science", "research"]):
            base_score += 0.1
            reason_parts.append("Credible author identifier")
    
    # Content-specific adjustments
    if "veris" in content.lower() and "data" in content.lower():
        if any(word in content.lower() for word in ["leak", "breach", "integrity"]):
            base_score += 0.2  # Relevant to data security investigation
            reason_parts.append("Relevant to data security investigation")
    
    # Timestamp recency (if available)
    timestamp_str = item.get("timestamp", "")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now()
            days_old = (now - timestamp.replace(tzinfo=None)).days
            if days_old < 7:
                base_score += 0.1
                reason_parts.append("Recent content")
        except:
            pass
    
    # Final adjustments
    final_score = max(0.0, min(1.0, base_score))
    
    # Determine flag reason
    if final_score >= 0.8:
        flag_reason = "High Trust - Verified Information"
    elif final_score >= 0.6:
        flag_reason = "Moderate Trust - Credible Source"
    elif final_score >= 0.4:
        flag_reason = "Low Trust - Requires Verification"
    elif final_score >= 0.2:
        flag_reason = "Very Low Trust - Questionable Content"
    else:
        flag_reason = "Untrusted - Error or Invalid Content"
    
    reasoning = f"Enhanced analysis: {'; '.join(reason_parts) if reason_parts else 'Basic content evaluation'}"
    
    return VerificationResult(
        evidence_id=evidence_id,
        trust_score=final_score,
        flag_reason=flag_reason,
        reasoning=reasoning
    )

def _create_enhanced_prompt(evidence_items: List[Dict], cross_refs: Dict) -> str:
    """Creates an enhanced prompt with cross-reference information."""
    
    # Build cross-reference summary
    cross_ref_summary = []
    for theme, evidence_ids in cross_refs.items():
        if len(evidence_ids) > 1:
            cross_ref_summary.append(f"- {theme}: {', '.join(evidence_ids)}")
    
    cross_ref_text = "\n".join(cross_ref_summary) if cross_ref_summary else "No significant cross-references detected."
    
    # Build evidence summaries
    prompt_items = []
    for item in evidence_items:
        media_info = f" (Contains {len(item.get('media', []))} media items)" if item.get('media') else ""
        timestamp_info = f" (Posted: {item.get('timestamp', 'Unknown')})" if item.get('timestamp') else ""
        
        prompt_items.append(
            f"Evidence ID: {item.get('evidence_id')}\n"
            f"Platform: {item.get('source_type', 'Unknown')}\n"
            f"Author: {item.get('author_id', 'Unknown')}\n"
            f"Content: \"{item.get('content', '')[:800]}...\"{media_info}{timestamp_info}"
        )
    
    evidence_text = "\n---\n".join(prompt_items)
    
    return f"""You are an expert digital forensics analyst specializing in online information verification. Analyze the following evidence batch for trustworthiness and relevance to potential data security investigations.

CROSS-REFERENCE ANALYSIS:
{cross_ref_text}

VERIFICATION CRITERIA:
- Content authenticity and reliability
- Source credibility and platform characteristics  
- Presence of supporting media/documentation
- Cross-references with other evidence
- Relevance to data security concerns
- Temporal context and recency

Your response MUST be a JSON object wrapped in <json> tags with an "analyses" array. Each analysis must include:
- "evidence_id": The exact evidence ID
- "trust_score": Float 0.0-1.0 (0.0=completely unreliable, 1.0=highly trustworthy)
- "flag_reason": Brief category (e.g., "Verified Report", "Social Media Claim", "Technical Error", "Cross-Referenced Information")
- "reasoning": Detailed explanation of your assessment including any cross-reference considerations

EVIDENCE BATCH:
{evidence_text}"""

# --- Enhanced Main Verifier Node ---
async def verifier_node(evidence_data: List[Dict], llm: Any = None) -> Dict[str, Any]:
    """
    Enhanced verifier that analyzes evidence with improved logic.
    Takes direct JSON input format as provided.
    """
    logger.info("--- Running Enhanced Verifier Node ---")
    
    if not evidence_data:
        logger.info("No evidence to verify.")
        return {"new_analysis": []}

    # Analyze cross-references
    cross_refs = _analyze_cross_references(evidence_data)
    logger.info(f"Found cross-references in {len([k for k, v in cross_refs.items() if len(v) > 1])} themes")
    
    analysis_results: List[VerificationResult] = []
    
    if llm:
        try:
            # Use LLM with enhanced prompt
            prompt = _create_enhanced_prompt(evidence_data, cross_refs)
            logger.info(f"Calling LLM to verify {len(evidence_data)} items...")
            
            response = await llm.ainvoke(prompt)
            json_string = _extract_json_from_xml(response.content)
            parsed_data = json.loads(json_string)

            if "analyses" not in parsed_data or not isinstance(parsed_data["analyses"], list):
                raise ValueError("Parsed JSON is missing the 'analyses' list.")

            valid_evidence_ids = {item["evidence_id"] for item in evidence_data}
            for analysis_item in parsed_data["analyses"]:
                evidence_id = analysis_item.get("evidence_id")
                if evidence_id in valid_evidence_ids:
                    try:
                        result = VerificationResult(**analysis_item)
                        analysis_results.append(result)
                    except ValidationError as ve:
                        logger.warning(f"Validation failed for item: {analysis_item}. Error: {ve}")
                        # Fallback to enhanced rule-based scoring
                        item = next(item for item in evidence_data if item["evidence_id"] == evidence_id)
                        quality_analysis = _analyze_content_quality(item.get("content", ""))
                        analysis_results.append(_enhanced_rule_based_score(item, cross_refs, quality_analysis))
                else:
                    logger.warning(f"LLM returned analysis for unknown evidence_id: {evidence_id}")

        except Exception as e:
            logger.error(f"Failed to process LLM response: {e}. Using enhanced rule-based fallbacks.")
            llm = None  # Force fallback
    
    # If LLM failed or not provided, use enhanced rule-based scoring
    if not llm or len(analysis_results) == 0:
        logger.info("Using enhanced rule-based verification...")
        for item in evidence_data:
            quality_analysis = _analyze_content_quality(item.get("content", ""))
            result = _enhanced_rule_based_score(item, cross_refs, quality_analysis)
            analysis_results.append(result)

    logger.info(f"Successfully created {len(analysis_results)} verification results.")
    
    # Add summary statistics
    avg_trust = sum(r.trust_score for r in analysis_results) / len(analysis_results) if analysis_results else 0
    high_trust_count = sum(1 for r in analysis_results if r.trust_score >= 0.7)
    
    return {
        "new_analysis": [res.model_dump() for res in analysis_results],
        "verification_summary": {
            "total_items": len(analysis_results),
            "average_trust_score": round(avg_trust, 3),
            "high_trust_items": high_trust_count,
            "cross_reference_themes": len([k for k, v in cross_refs.items() if len(v) > 1])
        }
    }

# --- Test Harness (Handles Your Collector Output Format) ---
async def test_verifier_with_collector_outputs():
    """
    Test harness that handles your collector output format:
    1. Reads from evidence_list folder (direct JSON arrays)
    2. Processes with enhanced verifier  
    3. Saves to test_outputs_verifier folder
    """
    
    COLLECTOR_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../evidence_list")
    VERIFIER_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs_verifier")
    os.makedirs(VERIFIER_OUTPUT_DIR, exist_ok=True)

    # LLM instance for the test
    try:
        llm_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize LLM: {e}. Using rule-based verification only.")
        llm_instance = None

    if not os.path.exists(COLLECTOR_OUTPUT_DIR):
        print(f"Directory not found: {COLLECTOR_OUTPUT_DIR}")
        return

    test_files = [f for f in os.listdir(COLLECTOR_OUTPUT_DIR) 
                  if f.endswith(".json")]
    
    if not test_files:
        print(f"No evidence files found in {COLLECTOR_OUTPUT_DIR}")
        print("Looking for files matching pattern: evidence_scrape_*.json")
        return

    for test_file in test_files:
        print(f"\n--- Testing Enhanced Verifier with evidence from: {test_file} ---")
        filepath = os.path.join(COLLECTOR_OUTPUT_DIR, test_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                evidence_data = json.load(f)
            
            # Your collector saves evidence as a direct JSON array
            if not isinstance(evidence_data, list):
                print(f"Expected list format, got {type(evidence_data)}. Skipping.")
                continue
                
            if not evidence_data:
                print("No evidence items to process. Skipping.")
                continue
            
            print(f"Found {len(evidence_data)} evidence items to verify")
            
            # Run the enhanced verifier with evidence data
            verifier_output = await verifier_node(evidence_data, llm_instance)
            
            # Create output structure
            output_state = {
                "source_file": test_file,
                "original_evidence": evidence_data,
                "new_analysis": verifier_output.get("new_analysis", []),
                "verification_summary": verifier_output.get("verification_summary", {}),
                "processed_at": datetime.now().isoformat(),
                "verifier_version": "enhanced"
            }
            
            output_filename = test_file.replace("evidence_scrape", "verified_evidence").replace(".json", "_verified.json")
            output_filepath = os.path.join(VERIFIER_OUTPUT_DIR, output_filename)

            with open(output_filepath, 'w') as f:
                json.dump(output_state, f, indent=2, default=str)
            
            print(f"--- Enhanced verification complete ---")
            print(f"Analyzed {len(verifier_output.get('new_analysis', []))} evidence items")
            
            if verifier_output.get("verification_summary"):
                summary = verifier_output["verification_summary"]
                print(f"Average trust score: {summary.get('average_trust_score', 0):.3f}")
                print(f"High trust items: {summary.get('high_trust_items', 0)}")
                print(f"Cross-reference themes: {summary.get('cross_reference_themes', 0)}")
            
            print(f"Saved results to: {output_filepath}")

        except Exception as e:
            print(f"An error occurred while processing {test_file}: {e}")
            import traceback
            traceback.print_exc()

# --- Standalone Test Function for Manual JSON ---
async def test_with_json_input(json_data: List[Dict], use_llm: bool = False):
    """Test function that takes JSON data directly (for manual testing)."""
    
    llm_instance = None
    if use_llm:
        try:
            llm_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}. Using rule-based verification only.")
    
    # Run verification
    result = await verifier_node(json_data, llm_instance)
    
    print("\n=== VERIFICATION RESULTS ===")
    print(json.dumps(result, indent=2))
    
    return result

# --- Main execution ---
if __name__ == '__main__':
    # Run with your collector output format
    print("Enhanced Verifier - Processing collector outputs from evidence_list folder")
    print("=" * 70)
    
    asyncio.run(test_verifier_with_collector_outputs())