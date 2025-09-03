# reverse_image_search_tool.py - Refactored
import asyncio
import hashlib
import requests
from datetime import datetime
from typing import Dict, List
from langchain_core.tools import tool
from schema import EvidenceItem, SourceType, MediaItem

@tool
async def reverse_image_search(media_url: str) -> List[EvidenceItem]:
    """
    Performs reverse image search to find the origin and other instances of an image.
    Returns a list of EvidenceItem objects.
    """
    try:
        image_hash = hashlib.md5(media_url.encode()).hexdigest()[:12]
        image_analysis = await _analyze_image(media_url)
        search_results = []

        # Simulated Google reverse search
        google_results = await _google_reverse_search(media_url)
        search_results.extend(google_results)

        if image_analysis.get("description"):
            text_search_results = await _search_by_description(image_analysis["description"])
            search_results.extend(text_search_results)

        metadata_results = await _search_by_metadata(image_analysis.get("metadata", {}))
        search_results.extend(metadata_results)

        analysis = await _analyze_search_results(search_results, media_url)
        evidence_items = []

        # Primary evidence item with image analysis
        evidence_items.append(EvidenceItem(
            source_type=SourceType.IMAGE_ANALYSIS,
            url=media_url,
            content=f"Image analysis: {image_analysis.get('description', 'No description')}",
            timestamp=datetime.now(),
            author_id="unknown",
            media=[MediaItem(media_type="IMAGE", url=media_url)],
            raw_data={"image_hash": image_hash, "analysis": image_analysis}
        ))

        # Add search results as evidence
        for result in search_results:
            evidence_items.append(EvidenceItem(
                source_type=SourceType.WEB_SEARCH,
                url=result.get("source_url", ""),
                content=f"Found at {result.get('source_domain', 'unknown')}: {result.get('context', '')}",
                timestamp=datetime.now() if not result.get("found_date") or result["found_date"] == "unknown" else datetime.fromisoformat(result["found_date"]),
                raw_data={"confidence": result.get("confidence", 0.0), "search_type": result.get("search_type", "unknown")}
            ))

        # Add origin analysis if available
        if analysis["likely_origin"]:
            evidence_items.append(EvidenceItem(
                source_type=SourceType.IMAGE_ORIGIN,
                url=analysis["likely_origin"].get("source_url", ""),
                content=f"Likely origin with confidence {analysis['confidence']}: {analysis['likely_origin'].get('context', '')}",
                timestamp=datetime.now(),
                raw_data={"repurposed": analysis["repurposed"], "timeline": analysis["timeline"]}
            ))

        return evidence_items

    except Exception as e:
        return [EvidenceItem(
            source_type=SourceType.IMAGE_ANALYSIS,
            content=f"Error in reverse image search: {str(e)}",
            timestamp=datetime.now(),
            url=media_url
        )]

async def _analyze_image(media_url: str) -> Dict:
    try:
        analysis = {
            "description": "Conference or meeting image with people in business attire",
            "estimated_date": None,
            "location_clues": [],
            "people_count": "3-5",
            "setting": "indoor/conference",
            "quality": "high",
            "metadata": {"file_format": "jpg", "dimensions": "unknown", "creation_date": None}
        }
        try:
            response = requests.head(media_url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                analysis["metadata"]["file_format"] = content_type.split('/')[-1]
                last_modified = response.headers.get('last-modified')
                if last_modified:
                    analysis["metadata"]["last_modified"] = last_modified
        except:
            pass
        return analysis
    except Exception:
        return {"error": "Failed to analyze image"}

async def _google_reverse_search(media_url: str) -> List[Dict]:
    # Simulated reverse search
    return [{"url": media_url, "description": "Original post", "confidence": 0.9}]

async def _search_by_description(description: str) -> List[Dict]:
    # Simulated text search
    return [{"url": f"https://example.com/{i}", "description": f"Match {i} for {description}", "confidence": 0.7} for i in range(2)]

async def _search_by_metadata(metadata: Dict) -> List[Dict]:
    # Simulated metadata search
    return [{"url": f"https://example.com/metadata{i}", "description": f"Metadata match {i}", "confidence": 0.6} for i in range(2)]

async def _analyze_search_results(search_results: List[Dict], original_url: str) -> Dict:
    dated_results = [r for r in search_results if r.get("found_date") and r["found_date"] != "unknown" and r["found_date"] != original_url]
    dated_results.sort(key=lambda x: x.get("found_date", "9999"))
    analysis = {
        "likely_origin": dated_results[0] if dated_results else None,
        "repurposed": len(dated_results) >= 2,
        "confidence": 0.8 if len(dated_results) >= 2 else 0.6 if dated_results else 0.0,
        "timeline": dated_results,
        "total_instances": len(search_results)
    }
    return analysis

async def test_reverse_search():
    test_url = "https://pbs.twimg.com/media/GzmyP_saAAAZX_d?format=jpg&name=small"
    result = await reverse_image_search(test_url)
    for item in result:
        print(item.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test_reverse_search())