"""workflow/reporter.py

An upgraded Final Reporter node for your investigation pipeline.

Key features added:
- Robust JSON-only extraction from LLM output (handles code fences and extra text).
- Strict schema validation with graceful fallback and automatic coercion of threat level.
- Optional *structured output* via LangChain's `with_structured_output` when available.
- Automatic tag normalization (dedupe, lowercase, alnum/underscore) and size limits.
- Deterministic prompting with explicit JSON Schema embedded in the prompt.
- Retry with exponential backoff, configurable timeouts.
- Summarization/compaction of oversized `new_analysis` to keep prompts within budget.
- Threat heuristic computed from `trust_score`s to guide the LLM and provide a fallback.
- Synchronous and asynchronous APIs.
- Clear logging (no noisy prints) and typed interfaces.
- CLI runner and test harness (mock LLM + optional live run if env vars are set).

This file preserves your public function name `final_reporter_node` and
input/output contract so it can drop into your existing workflow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypedDict, Union, runtime_checkable

# --------------------------------------------------------------------------------------
# External schema (from your project). We import and fall back to local light models only
# for testing/typing if the import isn't available (won't be used if your schema exists).
# --------------------------------------------------------------------------------------
try:  # Use the project's canonical models if present
    from schema import FinalConclusion, ThreatLevel, VerificationResult  # type: ignore
except Exception:  # pragma: no cover - fallback for isolated runs
    from pydantic import BaseModel

    class ThreatLevel(str, Enum):  # minimal mirror
        NONE = "NONE"
        LOW = "LOW"
        MODERATE = "MODERATE"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"

    class FinalConclusion(BaseModel):
        summary: str
        threat_level: ThreatLevel
        tags: List[str]

    class VerificationResult(BaseModel):
        evidence_id: str
        trust_score: float
        flag_reason: str
        reasoning: str

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
LOGGER = logging.getLogger("workflow.reporter")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(os.getenv("REPORTER_LOG_LEVEL", "INFO").upper())


# --------------------------------------------------------------------------------------
# LLM protocol + helpers (tool-agnostic)
# --------------------------------------------------------------------------------------
@runtime_checkable
class AsyncChatModel(Protocol):
    """Minimal async chat protocol expected by this module.

    The object should implement `ainvoke(prompt: str) -> Any` and optionally
    `with_structured_output(schema) -> AsyncChatModel`.
    """

    async def ainvoke(self, prompt: str) -> Any:  # pragma: no cover - protocol
        ...

    def with_structured_output(self, schema: Any) -> "AsyncChatModel":  # optional
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
MAX_PROMPT_CHARS: int = int(os.getenv("REPORTER_MAX_PROMPT_CHARS", 60_000))
MAX_ANALYSIS_ITEMS: int = int(os.getenv("REPORTER_MAX_ANALYSIS_ITEMS", 200))
RETRY_ATTEMPTS: int = int(os.getenv("REPORTER_RETRY_ATTEMPTS", 3))
RETRY_BASE_DELAY: float = float(os.getenv("REPORTER_RETRY_BASE_DELAY", 0.8))
TAG_LIMIT: int = int(os.getenv("REPORTER_TAG_LIMIT", 18))
SUMMARY_MAX_WORDS: int = int(os.getenv("REPORTER_SUMMARY_MAX_WORDS", 130))
TIMEOUT_SECS: float = float(os.getenv("REPORTER_TIMEOUT_SECS", 90))


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
JSON_FENCE_RE = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
NON_JSON_PREFIX_RE = re.compile(r"^[^\[{]*([\[{].*[\]}]).*$", re.DOTALL)


def _extract_json_str(text: str) -> str:
    """Extract the tightest JSON object/array string from an LLM response.

    Handles code fences and stray prose. Falls back to input if it already
    looks like JSON.
    """
    text = text.strip()
    # Try fenced block first
    m = JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Try to isolate the first {...} or [...] segment
    if text and (text[0] in "[{"):
        return text
    m = NON_JSON_PREFIX_RE.match(text)
    if m:
        return m.group(1).strip()
    return text  # last resort


def _coerce_threat(value: str) -> ThreatLevel:
    try:
        return ThreatLevel(value.upper())
    except Exception:
        # Map common variants
        mapping = {
            "NO": ThreatLevel.NONE,
            "SAFE": ThreatLevel.LOW,
            "MEDIUM": ThreatLevel.MODERATE,
            "MOD": ThreatLevel.MODERATE,
            "SEVERE": ThreatLevel.HIGH,
            "CRIT": ThreatLevel.CRITICAL,
        }
        return mapping.get(value.upper().strip(), ThreatLevel.NONE)


def _normalize_tag(t: str) -> Optional[str]:
    t = (t or "").strip().lower()
    if not t:
        return None
    # Keep alnum and common separators
    t = re.sub(r"[^a-z0-9_\- ]+", "", t)
    t = re.sub(r"\s+", "_", t).strip("_")
    return t or None


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _compact_analysis(analysis: Sequence[VerificationResult], max_items: int) -> List[Dict[str, Any]]:
    """Compact VerificationResult items for prompt efficiency.

    Selects up to `max_items` items prioritizing extremes (lowest/highest trust)
    and keeps only the most informative fields.
    """
    items = list(analysis)
    if not items:
        return []

    # Sort by trust_score ascending and descending to capture extremes
    try:
        by_low = sorted(items, key=lambda x: float(getattr(x, "trust_score", 0.0)))
        by_high = list(reversed(by_low))
    except Exception:
        by_low, by_high = items, items

    # Take first K/2 from low, K/2 from high, plus center if space
    k = min(max_items, len(items))
    half = max(1, k // 2)
    picked = []
    seen_ids = set()

    for lst in (by_low[:half], by_high[:k - half]):
        for it in lst:
            eid = getattr(it, "evidence_id", None)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            picked.append(it)

    # Fallback: if still fewer than k, fill with remaining in original order
    if len(picked) < k:
        for it in items:
            eid = getattr(it, "evidence_id", None)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            picked.append(it)
            if len(picked) >= k:
                break

    # Keep only compact fields
    compacted: List[Dict[str, Any]] = []
    for it in picked:
        compacted.append(
            {
                "evidence_id": getattr(it, "evidence_id", ""),
                "trust_score": float(getattr(it, "trust_score", 0.0)),
                "flag_reason": getattr(it, "flag_reason", ""),
                "reasoning": _limit_words(getattr(it, "reasoning", ""), 60),
            }
        )
    return compacted


def _threat_heuristic(analysis: Sequence[VerificationResult]) -> ThreatLevel:
    """Compute a coarse threat heuristic from trust scores and flags.

    This is *advisory* and included in the prompt to guide the LLM; the LLM may
    override it, and we still validate the final field.
    """
    if not analysis:
        return ThreatLevel.NONE

    scores = []
    risk_bumps = 0
    for a in analysis:
        try:
            s = float(getattr(a, "trust_score", 0.0))
            scores.append(s)
            flag = (getattr(a, "flag_reason", "") or "").lower()
            if any(x in flag for x in ["threat", "malware", "violence", "incitement", "extrem", "harm"]):
                risk_bumps += 1
        except Exception:
            continue

    if not scores:
        return ThreatLevel.NONE

    avg = sum(scores) / len(scores)
    # Heuristic mapping (tweak thresholds as needed)
    if avg >= 0.8 or risk_bumps >= 3:
        return ThreatLevel.CRITICAL
    if avg >= 0.6 or risk_bumps >= 2:
        return ThreatLevel.HIGH
    if avg >= 0.4 or risk_bumps >= 1:
        return ThreatLevel.MODERATE
    if avg >= 0.2:
        return ThreatLevel.LOW
    return ThreatLevel.NONE


def _json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["summary", "threat_level", "tags"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "threat_level": {
                "type": "string",
                "enum": [e.value for e in ThreatLevel],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": TAG_LIMIT,
            },
        },
    }


def _render_prompt(initial_query: str, compacted_analysis: List[Dict[str, Any]], heuristic: ThreatLevel) -> str:
    schema_json = json.dumps(_json_schema(), indent=2)
    compact_str = json.dumps(compacted_analysis, indent=2, ensure_ascii=False)

    return (
        "You are the lead investigator. Synthesize all findings into a FINAL JSON object only.\n"
        "Follow the JSON Schema strictly and DO NOT add any additional keys or text.\n\n"
        f"Initial Query: {json.dumps(initial_query)}\n\n"
        f"Compacted Analysis (max {len(compacted_analysis)} items):\n{compact_str}\n\n"
        f"Advisory threat assessment (heuristic): {heuristic.value}\n\n"
        "Rules:\n"
        "- Output must be a single JSON object with keys: summary, threat_level, tags.\n"
        f"- summary: One paragraph, concise (≤ {SUMMARY_MAX_WORDS} words).\n"
        "- threat_level: One of [NONE, LOW, MODERATE, HIGH, CRITICAL].\n"
        f"- tags: ≤ {TAG_LIMIT} short keywords.\n"
        "- No markdown fences, no extra commentary.\n\n"
        f"JSON Schema:\n{schema_json}\n"
    )


async def _ainvoke_with_retries(llm: AsyncChatModel, prompt: str) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            LOGGER.debug("LLM attempt %d/%d", attempt, RETRY_ATTEMPTS)

            # If the LLM supports structured output, try to leverage it first
            llm_used = llm
            if hasattr(llm, "with_structured_output"):
                try:
                    # Use local model for schema if available
                    from pydantic import BaseModel

                    class _FCModel(BaseModel):
                        summary: str
                        threat_level: str
                        tags: List[str]

                    llm_used = llm.with_structured_output(_FCModel)
                    LOGGER.debug("Using structured output mode")
                except Exception:
                    llm_used = llm

            async def _call() -> Any:
                return await llm_used.ainvoke(prompt)

            resp: Any = await asyncio.wait_for(_call(), timeout=TIMEOUT_SECS)
            content = getattr(resp, "content", resp)
            if hasattr(content, "model_dump_json"):
                # Structured output pydantic BaseModel
                json_text = content.model_dump_json()
            elif isinstance(content, (dict, list)):
                json_text = json.dumps(content)
            else:
                json_text = str(content)

            return _extract_json_str(json_text)
        except Exception as e:
            last_err = e
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            LOGGER.warning("LLM call failed on attempt %d: %s", attempt, e)
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(delay)
    assert last_err is not None
    raise last_err


def _sanitize_and_validate(report_data: Dict[str, Any]) -> FinalConclusion:
    # Coerce/clean fields
    summary = _limit_words(str(report_data.get("summary", "")).strip(), SUMMARY_MAX_WORDS)
    t_raw = str(report_data.get("threat_level", "NONE")).strip()
    threat = _coerce_threat(t_raw)

    tags_in = report_data.get("tags", []) or []
    clean_tags: List[str] = []
    seen: set[str] = set()
    for t in tags_in:
        nt = _normalize_tag(str(t))
        if nt and nt not in seen:
            clean_tags.append(nt)
            seen.add(nt)
        if len(clean_tags) >= TAG_LIMIT:
            break

    if not summary:
        summary = "No summary provided."

    try:
        return FinalConclusion(summary=summary, threat_level=threat, tags=clean_tags)
    except Exception as e:
        LOGGER.error("Failed to build FinalConclusion using project schema: %s", e)
        # Fallback via lightweight reconstruction (should rarely happen)
        from pydantic import BaseModel

        class _TmpFC(BaseModel):
            summary: str
            threat_level: ThreatLevel
            tags: List[str]

        tmp = _TmpFC(summary=summary, threat_level=threat, tags=clean_tags)
        # Recreate FinalConclusion with minimal assumptions
        return FinalConclusion(**tmp.model_dump())  # type: ignore[arg-type]


def _truncate_if_needed(prompt: str) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    LOGGER.warning("Prompt length %d exceeds limit %d; truncating.", len(prompt), MAX_PROMPT_CHARS)
    return prompt[:MAX_PROMPT_CHARS]


# --------------------------------------------------------------------------------------
# Public API (async)
# --------------------------------------------------------------------------------------
async def final_reporter_node(state: Dict[str, Any], llm: AsyncChatModel) -> Dict[str, Any]:
    """Generates the final conclusion for the investigation.

    Parameters
    ----------
    state : dict
        Must contain keys:
          - 'initial_query': str
          - 'new_analysis': Sequence[VerificationResult]
    llm : AsyncChatModel
        LLM client implementing `ainvoke()`.

    Returns
    -------
    dict with key 'final_conclusion' mapping to a `FinalConclusion`-compatible dict.
    """
    LOGGER.info("--- Running Final Reporter ---")

    initial_query: str = state.get("initial_query", "") or ""
    all_analysis: Sequence[VerificationResult] = state.get("new_analysis", []) or []

    # Compact the analysis for the prompt
    compacted = _compact_analysis(all_analysis, MAX_ANALYSIS_ITEMS)
    heuristic = _threat_heuristic(all_analysis)

    prompt = _render_prompt(initial_query, compacted, heuristic)
    prompt = _truncate_if_needed(prompt)

    try:
        json_text = await _ainvoke_with_retries(llm, prompt)
        report_data = json.loads(json_text)
        conclusion = _sanitize_and_validate(report_data)
        result = {"final_conclusion": conclusion.model_dump() if hasattr(conclusion, "model_dump") else conclusion.dict()}
        LOGGER.debug("Final conclusion generated successfully")
        return result
    except Exception as e:
        LOGGER.error("Error generating final report: %s", e)
        return {
            "final_conclusion": {
                "summary": "Failed to generate report.",
                "threat_level": ThreatLevel.NONE.value,
                "tags": ["error"],
            }
        }


# --------------------------------------------------------------------------------------
# Optional sync wrapper
# --------------------------------------------------------------------------------------
def final_reporter_node_sync(state: Dict[str, Any], llm: AsyncChatModel) -> Dict[str, Any]:
    """Synchronous wrapper around `final_reporter_node` for convenience."""
    return asyncio.run(final_reporter_node(state, llm))


# --------------------------------------------------------------------------------------
# CLI and Test Harness
# --------------------------------------------------------------------------------------
class _MockLLM:
    """A tiny mock that returns well-formed JSON for tests."""

    async def ainvoke(self, prompt: str) -> Any:
        # Minimal behavior guided by heuristic hints inside the prompt
        threat = "LOW"
        if "Advisory threat assessment (heuristic): CRITICAL" in prompt:
            threat = "CRITICAL"
        elif "HIGH" in prompt:
            threat = "HIGH"
        elif "MODERATE" in prompt:
            threat = "MODERATE"

        payload = {
            "summary": "Based on compiled analyses, credible sources outweigh unverified claims; monitor but no immediate action required.",
            "threat_level": threat,
            "tags": ["rumor_check", "verification", "monitoring"],
        }
        return type("Resp", (), {"content": json.dumps(payload)})()


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def _live_llm_if_available(model: Optional[str] = None) -> Optional[AsyncChatModel]:
    """Try to instantiate a live Gemini LLM via LangChain if environment allows."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or os.getenv("REPORTER_LIVE_MODEL", "gemini-1.5-flash-latest")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            generation_config={"response_mime_type": "application/json"},
        )
        return llm  # type: ignore[return-value]
    except Exception as e:
        LOGGER.warning("Live LLM not available: %s", e)
        return None


def _example_state() -> Dict[str, Any]:
    # Mirrors your original unit-test state
    try:
        a1 = VerificationResult(
            evidence_id="evd_123",
            trust_score=0.8,
            flag_reason="Verified Claim",
            reasoning="Official press release.",
        )
        a2 = VerificationResult(
            evidence_id="evd_456",
            trust_score=0.2,
            flag_reason="Opinion",
            reasoning="Anonymous blog post with no sources.",
        )
    except Exception:
        # Fallback when using the minimal pydantic mirror
        a1 = VerificationResult.model_validate(
            {"evidence_id": "evd_123", "trust_score": 0.8, "flag_reason": "Verified Claim", "reasoning": "Official press release."}
        )
        a2 = VerificationResult.model_validate(
            {"evidence_id": "evd_456", "trust_score": 0.2, "flag_reason": "Opinion", "reasoning": "Anonymous blog post with no sources."}
        )

    return {"initial_query": "Assess the validity of recent rumors.", "new_analysis": [a1, a2]}


def _cli_usage() -> None:
    print(
        """
Final Reporter CLI
------------------
Usage:
  python -m workflow.reporter [--input state.json] [--output report.json] [--live] [--model MODEL]

Options:
  --input   Path to JSON state. If omitted, uses a built-in example.
  --output  Where to write the final_conclusion JSON. Prints to stdout if omitted.
  --live    Use a live Gemini model (requires GOOGLE_API_KEY). Falls back to mock if not available.
  --model   Override live model name (default: gemini-1.5-flash-latest).

Environment overrides:
  REPORTER_MAX_PROMPT_CHARS, REPORTER_MAX_ANALYSIS_ITEMS, REPORTER_RETRY_ATTEMPTS,
  REPORTER_RETRY_BASE_DELAY, REPORTER_TAG_LIMIT, REPORTER_SUMMARY_MAX_WORDS,
  REPORTER_TIMEOUT_SECS, REPORTER_LOG_LEVEL
"""
    )


async def _main_cli(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args, _ = parser.parse_known_args(argv)

    if args.input:
        state = _load_json_file(args.input)
    else:
        state = _example_state()

    llm: Optional[AsyncChatModel] = None
    if args.live:
        llm = await _live_llm_if_available(args.model)
    llm = llm or _MockLLM()

    result = await final_reporter_node(state, llm)  # type: ignore[arg-type]

    if args.output:
        _save_json_file(args.output, result)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # If run as a script, behave like a CLI
    try:
        raise SystemExit(asyncio.run(_main_cli(sys.argv[1:])))
    except KeyboardInterrupt:
        raise SystemExit(130)
