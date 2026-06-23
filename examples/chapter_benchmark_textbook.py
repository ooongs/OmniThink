import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from src.actions.article_generation import ArticleGenerationModule
from src.actions.outline_generation import OutlineGenerationModule
from src.actions.article_polish import ArticlePolishingModule
from src.dataclass.Article import Article
from src.tools.lm import OpenAIModel_dashscope
from src.tools.mindmap import MindMap
from src.tools.rm import SerperSearch, is_allowed_source


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RUN_LOG_PATH: Optional[Path] = None
RUN_ROWS_DIR: Optional[Path] = None
RUN_LOG_LOCK = threading.Lock()

FAST_SYSTEM_PROMPT = (
    "You support a textbook-generation baseline. Extract concise concepts from retrieved "
    "web snippets, prioritizing learning objectives and required knowledge units. Keep outputs "
    "factual, compact, and useful for later section writing."
)

TEXTBOOK_SYSTEM_PROMPT = (
    "You write university textbook chapters for an automated benchmark baseline. The output "
    "must be educational rather than encyclopedic: define key ideas, connect concepts, explain "
    "why they matter, and use examples only when they clarify the required learning objectives. "
    "Treat the supplied benchmark requirements as mandatory coverage constraints. Do not omit "
    "required knowledge units, but integrate them naturally into coherent prose. Use Markdown "
    "headings, keep the chapter structure stable, and cite retrieved source snippets with inline "
    "numeric citations when factual claims depend on them. Avoid writing a references section. "
    "Avoid meta-commentary about the benchmark, the prompt, the pipeline, or missing sources."
)


def iter_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Dict]:
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and emitted >= limit:
                break
            yield json.loads(line)
            emitted += 1


def emit_json(payload: Dict):
    line = json.dumps(payload, ensure_ascii=False, default=str)
    tqdm.write(line)
    with RUN_LOG_LOCK:
        if RUN_LOG_PATH is not None:
            RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if RUN_ROWS_DIR is not None and payload.get("id"):
            row_log_dir = RUN_ROWS_DIR / safe_filename(str(payload["id"]))
            row_log_dir.mkdir(parents=True, exist_ok=True)
            with (row_log_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def write_text_artifact(artifact_dir: Optional[Path], name: str, content: str):
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / name).write_text(content, encoding="utf-8")


def write_json_artifact(artifact_dir: Optional[Path], name: str, payload: Any):
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def get_row_artifact_dir(args, row: Dict) -> Optional[Path]:
    if not args.keep_intermediates:
        return None
    return args.rundir / "rows" / safe_filename(str(row["id"]))


def configure_run_logging(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.rundir is None:
        args.rundir = args.outputdir / "_runs" / timestamp
    args.rundir.mkdir(parents=True, exist_ok=True)

    global RUN_LOG_PATH, RUN_ROWS_DIR
    RUN_LOG_PATH = args.rundir / "events.jsonl"
    RUN_ROWS_DIR = args.rundir / "rows"
    RUN_ROWS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_artifact(args.rundir, "python_command.txt", " ".join(sys.argv) + "\n")
    write_json_artifact(args.rundir, "run_config.json", vars(args))
    emit_json({
        "stage": "run_logging",
        "run_dir": str(args.rundir),
        "event_log": str(RUN_LOG_PATH),
    })


def flatten_knowledge_units(section_block: Dict) -> List[str]:
    units = []
    for group in section_block.get("ku_groups", []):
        units.extend(group.get("knowledge_units", []))
    return units


def count_knowledge_units(row: Dict) -> int:
    return sum(len(flatten_knowledge_units(section)) for section in row.get("section_blocks", []))


def compute_budgets(row: Dict) -> Tuple[int, int]:
    section_count = len(row.get("section_blocks", []))
    knowledge_unit_count = count_knowledge_units(row)
    query_budget = 2 * section_count + math.ceil(knowledge_unit_count / 5)
    source_budget = min(3 * query_budget, 40)
    return query_budget, source_budget


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def compact_query(text: str, max_chars: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def build_seed_queries(row: Dict, query_budget: int) -> List[str]:
    chapter_title = row.get("chapter_title", "")
    prefix = compact_query(f"{chapter_title} textbook")
    queries = []
    for section in row.get("section_blocks", []):
        for objective in section.get("learning_objectives", [])[:2]:
            queries.append(compact_query(f"{prefix}: {objective}"))
        for unit_chunk in chunked(flatten_knowledge_units(section), 5):
            queries.append(compact_query(f"{prefix}: {'; '.join(unit_chunk)}"))
    return queries[:query_budget]


def normalized_leakage_key(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def source_to_leakage_text(source: Any) -> str:
    if isinstance(source, dict):
        parts = [
            source.get("url", ""),
            source.get("link", ""),
            source.get("title", ""),
            source.get("description", ""),
            source.get("snippet", ""),
        ]
        snippets = source.get("snippets") or []
        if isinstance(snippets, str):
            parts.append(snippets)
        else:
            parts.extend(snippets)
        return " ".join(str(part or "") for part in parts).lower()
    return str(source or "").lower()


def add_leakage_phrase(phrases: Dict[str, Dict[str, str]], value: Any, kind: str):
    value = str(value or "").strip()
    if len(value) < 8:
        return
    raw = value.lower()
    key = normalized_leakage_key(value)
    if not key or len(key) < 8:
        return
    phrases.setdefault(key, {"raw": raw, "key": key, "kind": kind, "value": value})


def get_source_page_files(row: Dict) -> List[str]:
    files = []
    metadata = row.get("metadata") or {}
    chapter = metadata.get("chapter") or {}
    files.extend(chapter.get("source_page_files") or [])
    files.extend(row.get("source_page_files") or [])
    return [str(path) for path in files if path]


def build_chapter_leakage_phrases(row: Dict) -> List[Dict[str, str]]:
    phrases: Dict[str, Dict[str, str]] = {}
    metadata = row.get("metadata") or {}
    for value, kind in [
        (row.get("book_title"), "book_title"),
        (row.get("book_slug"), "book_slug"),
        (row.get("id"), "dataset_id"),
        (metadata.get("id"), "metadata_id"),
    ]:
        add_leakage_phrase(phrases, value, kind)

    for source_page_file in get_source_page_files(row):
        source_path = Path(source_page_file)
        add_leakage_phrase(phrases, source_path.name, "source_page_file")
        add_leakage_phrase(phrases, source_path.stem, "source_page_stem")

    return list(phrases.values())


def match_leakage_phrase(text: str, phrases: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    raw_text = str(text or "").lower()
    key_text = normalized_leakage_key(raw_text)
    for phrase in phrases:
        raw = phrase.get("raw") or ""
        key = phrase.get("key") or ""
        if raw and raw in raw_text:
            return phrase
        if key and key in key_text:
            return phrase
    return None


def get_chapter_source_leakage_reason(row: Dict, source: Any) -> Optional[Dict[str, str]]:
    if not is_allowed_source(source):
        return {"reason": "static_leakage_filter", "kind": "static", "value": ""}
    phrase = match_leakage_phrase(source_to_leakage_text(source), build_chapter_leakage_phrases(row))
    if phrase:
        return {
            "reason": "chapter_leakage_phrase",
            "kind": phrase["kind"],
            "value": phrase["value"],
        }
    return None


def build_chapter_leakage_policy(row: Dict) -> Dict[str, Any]:
    phrases = build_chapter_leakage_phrases(row)

    def query_allowed(query: str) -> bool:
        return match_leakage_phrase(query, phrases) is None

    def source_allowed(source: Any) -> bool:
        return get_chapter_source_leakage_reason(row, source) is None

    return {
        "phrases": phrases,
        "query_allowed": query_allowed,
        "source_allowed": source_allowed,
    }


def collect_sources_from_mind_map_node(node) -> List[Dict[str, Any]]:
    if node is None:
        return []
    sources = []
    for info in getattr(node, "info", []) or []:
        if isinstance(info, dict):
            sources.append(info)
    children = getattr(node, "children", {}) or {}
    for child in children.values():
        sources.extend(collect_sources_from_mind_map_node(child))
    return sources


def collect_sources_from_mind_map_artifact(node_data: Dict) -> List[Dict[str, Any]]:
    sources = []
    for info in node_data.get("info") or []:
        if isinstance(info, dict):
            sources.append(info)
    for child in (node_data.get("children") or {}).values():
        if isinstance(child, dict):
            sources.extend(collect_sources_from_mind_map_artifact(child))
    return sources


def collect_sources_from_section_outputs(section_outputs: List[Dict]) -> List[Dict[str, Any]]:
    sources = []
    for section_output in section_outputs or []:
        for info in section_output.get("collected_info") or []:
            if isinstance(info, dict):
                sources.append(info)
    return sources


def collect_existing_provenance_sources(artifact_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], bool]:
    if artifact_dir is None:
        return [], False
    sources = []
    provenance_files = 0
    mind_map_path = artifact_dir / "mind_map.json"
    if mind_map_path.exists():
        provenance_files += 1
        try:
            sources.extend(collect_sources_from_mind_map_artifact(json.loads(mind_map_path.read_text(encoding="utf-8"))))
        except Exception:
            pass
    section_outputs_path = artifact_dir / "section_outputs.json"
    if section_outputs_path.exists():
        provenance_files += 1
        try:
            sources.extend(collect_sources_from_section_outputs(json.loads(section_outputs_path.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return sources, provenance_files > 0


def summarize_source_for_report(source: Dict[str, Any], reason: Dict[str, str]) -> Dict[str, Any]:
    return {
        "url": source.get("url", ""),
        "title": source.get("title", ""),
        "description": source.get("description", ""),
        **reason,
    }


def build_leakage_report(row: Dict,
                         sources: List[Dict[str, Any]],
                         retriever_report: Optional[Dict[str, Any]] = None,
                         provenance_available: bool = True,
                         strict: bool = True) -> Dict[str, Any]:
    leaked_sources = []
    seen = set()
    for source in sources:
        reason = get_chapter_source_leakage_reason(row, source)
        if not reason:
            continue
        key = (source.get("url", ""), reason.get("reason", ""), reason.get("value", ""))
        if key in seen:
            continue
        seen.add(key)
        leaked_sources.append(summarize_source_for_report(source, reason))

    retriever_report = retriever_report or {}
    unverifiable = strict and not provenance_available
    issues = []
    if leaked_sources:
        issues.append("accepted_source_leakage")
    if unverifiable:
        issues.append("leakage_unverifiable")

    return {
        "valid": not issues,
        "issues": issues,
        "unverifiable": unverifiable,
        "provenance_available": provenance_available,
        "accepted_source_count": len({source.get("url", "") for source in sources if source.get("url")}),
        "leaked_source_count": len(leaked_sources),
        "leaked_sources": leaked_sources[:20],
        "leakage_query_blocked_count": retriever_report.get("leakage_query_blocked_count", 0),
        "leakage_source_blocked_count": retriever_report.get("leakage_source_blocked_count", 0),
        "blocked_query_samples": retriever_report.get("blocked_query_samples", []),
        "blocked_source_samples": retriever_report.get("blocked_source_samples", []),
    }


def make_section_title(section: Dict, index: int) -> str:
    objectives = section.get("learning_objectives", [])
    if objectives:
        title = objectives[0].strip().rstrip(".")
    else:
        title = section.get("section_id", f"Section {index}")
    title = compact_query(title, max_chars=88)
    return f"Section {index}: {title}"


def get_chapter_word_budget(row: Dict) -> Tuple[int, int, int]:
    chapter_budget = row.get("length_budget", {}).get("chapter_budget", {})
    target_words = int(chapter_budget.get("target_words") or 0)
    word_range = chapter_budget.get("word_range") or []
    min_words = int(word_range[0]) if len(word_range) > 0 else int(target_words * 0.75)
    max_words = int(word_range[1]) if len(word_range) > 1 else int(target_words * 1.5)
    return target_words, min_words, max_words


def count_markdown_words(markdown: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)*\b", markdown))


def allocate_section_word_targets(row: Dict) -> Dict[str, int]:
    target_words, _, _ = get_chapter_word_budget(row)
    sections = row.get("section_blocks", [])
    if target_words <= 0 or not sections:
        return {}

    weights = []
    for section in sections:
        weight = len(section.get("learning_objectives", [])) + len(flatten_knowledge_units(section))
        weights.append(max(weight, 1))

    total_weight = sum(weights)
    raw_targets = [max(1, int(target_words * weight / total_weight)) for weight in weights]
    remainder = target_words - sum(raw_targets)
    idx = 0
    while remainder > 0:
        raw_targets[idx % len(raw_targets)] += 1
        remainder -= 1
        idx += 1

    return {
        make_section_title(section, idx): raw_targets[idx - 1]
        for idx, section in enumerate(sections, start=1)
    }


def build_draft_outline(row: Dict) -> str:
    outline_lines = []
    for idx, section in enumerate(row.get("section_blocks", []), start=1):
        outline_lines.append(f"# {make_section_title(section, idx)}")
    return "\n".join(outline_lines)


def normalize_title(title: str) -> str:
    title = title.strip().strip("*_")
    title = re.sub(r"\s+", " ", title)
    return title.lower()


def book_chapter_label(row: Dict) -> str:
    book_title = row.get("book_title", "")
    chapter_number = row.get("chapter_number", "")
    if book_title and chapter_number:
        return f"{book_title}, Chapter {chapter_number}"
    if chapter_number:
        return f"Chapter {chapter_number}"
    return book_title


def is_book_chapter_label(row: Dict, line: str) -> bool:
    stripped = line.strip()
    stripped = stripped.strip("*_")
    stripped = re.sub(r"^#+\s*", "", stripped).strip()
    label = book_chapter_label(row)
    if label and normalize_title(stripped) == normalize_title(label):
        return True
    return False


def is_chapter_heading_title(row: Dict, title: str) -> bool:
    normalized = normalize_title(title)
    candidates = [
        row.get("chapter_title", ""),
        book_chapter_label(row),
        row.get("book_title", ""),
    ]
    return any(candidate and normalized == normalize_title(candidate) for candidate in candidates)


def extract_heading_titles(markdown: str, level: int) -> List[str]:
    titles = []
    for line in markdown.splitlines():
        match = HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == level:
            titles.append(match.group(2).strip())
    return titles


def extract_first_level_section_titles(outline: str) -> List[str]:
    return extract_heading_titles(outline, level=1)


def shift_outline_heading_levels(outline: str, delta: int) -> str:
    shifted_lines = []
    for line in outline.splitlines():
        stripped = line.strip()
        match = HEADING_RE.match(stripped)
        if not match:
            continue
        new_level = min(max(len(match.group(1)) + delta, 1), 6)
        shifted_lines.append(f"{'#' * new_level} {match.group(2).strip()}")
    return "\n".join(shifted_lines)


def strip_chapter_heading_from_outline(row: Dict, outline: str) -> Tuple[str, bool]:
    lines = [line.strip() for line in outline.splitlines() if line.strip()]
    if not lines:
        return outline, False

    first_match = HEADING_RE.match(lines[0])
    if not first_match or len(first_match.group(1)) != 1:
        return outline, False
    if not is_chapter_heading_title(row, first_match.group(2)):
        return outline, False

    remaining = lines[1:]
    if not remaining:
        return "", True
    heading_levels = [
        len(match.group(1))
        for match in (HEADING_RE.match(line) for line in remaining)
        if match
    ]
    if heading_levels and min(heading_levels) > 1:
        return shift_outline_heading_levels("\n".join(remaining), delta=-1), True
    return "\n".join(remaining), True


def align_outline_to_source_sections(row: Dict, outline: str, draft_outline: str) -> Tuple[str, List[str]]:
    expected_count = len(row.get("section_blocks", []))
    normalized_outline, removed_chapter_heading = strip_chapter_heading_from_outline(row, outline)
    if removed_chapter_heading:
        emit_json({
            "stage": "outline_normalized",
            "id": row.get("id"),
            "reason": "removed_chapter_heading",
        })

    section_titles = extract_first_level_section_titles(normalized_outline)
    if len(section_titles) == expected_count:
        return normalized_outline, section_titles

    level_two_titles = extract_heading_titles(normalized_outline, level=2)
    if len(level_two_titles) == expected_count:
        promoted_outline = shift_outline_heading_levels(normalized_outline, delta=-1)
        emit_json({
            "stage": "outline_normalized",
            "id": row.get("id"),
            "reason": "promoted_second_level_sections",
            "expected": expected_count,
        })
        return promoted_outline, extract_first_level_section_titles(promoted_outline)

    emit_json({
        "stage": "outline_fallback",
        "id": row.get("id"),
        "reason": "first_level_section_count_mismatch",
        "expected": expected_count,
        "actual": len(section_titles),
    })
    fallback_titles = extract_first_level_section_titles(draft_outline)
    return draft_outline, fallback_titles


def build_outline_context(row: Dict) -> str:
    section_word_targets = allocate_section_word_targets(row)
    lines = [
        f"Book: {row.get('book_title', '')}",
        f"Chapter: {row.get('chapter_title', '')}",
        "Create one first-level outline section for each source section below, in the same order.",
        "",
    ]
    for idx, section in enumerate(row.get("section_blocks", []), start=1):
        source_title = make_section_title(section, idx)
        knowledge_units = flatten_knowledge_units(section)
        lines.append(f"Source section {idx}: {source_title}")
        if source_title in section_word_targets:
            lines.append(f"Target words: approximately {section_word_targets[source_title]}")
        lines.append("Learning objectives:")
        lines.extend(f"- {objective}" for objective in section.get("learning_objectives", []))
        lines.append("Required knowledge units:")
        lines.extend(f"- {unit}" for unit in knowledge_units)
        lines.append("")
    return "\n".join(lines)


def build_section_metadata(row: Dict, section_titles: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    section_metadata = {}
    section_queries = {}
    section_word_targets = allocate_section_word_targets(row)
    for idx, section in enumerate(row.get("section_blocks", []), start=1):
        if idx > len(section_titles):
            break
        section_title = section_titles[idx - 1]
        source_section_title = make_section_title(section, idx)
        knowledge_units = flatten_knowledge_units(section)
        metadata_lines = []
        if source_section_title in section_word_targets:
            metadata_lines.extend([
                f"Target length for this section: approximately {section_word_targets[source_section_title]} words.",
                "Use enough explanation, definitions, worked interpretation, and examples to meet this length while staying focused.",
                "",
            ])
        metadata_lines.append("Section-specific learning objectives:")
        metadata_lines.extend(f"- {objective}" for objective in section.get("learning_objectives", []))
        metadata_lines.append("")
        metadata_lines.append("Section-specific required knowledge units:")
        metadata_lines.extend(f"- {unit}" for unit in knowledge_units)
        section_metadata[section_title] = "\n".join(metadata_lines)
        section_queries[section_title] = section.get("learning_objectives", []) + knowledge_units
    return section_metadata, section_queries


def demote_headings(markdown: str, max_level: int = 4) -> str:
    def replace(match):
        hashes = match.group(1)
        return f"{'#' * min(len(hashes) + 1, max_level)}{match.group(2)}"

    return re.sub(r"^(#{1,6})(\s+)", replace, markdown, flags=re.MULTILINE)


def normalize_final_markdown(row: Dict, markdown: str) -> str:
    chapter_title = row.get("chapter_title", "Untitled Chapter")
    output_lines = []
    chapter_seen = False

    for line in markdown.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            output_lines.append("")
            continue
        if is_book_chapter_label(row, stripped):
            continue

        match = HEADING_RE.match(stripped)
        if not match:
            output_lines.append(line.rstrip())
            continue

        level = len(match.group(1))
        title = match.group(2).strip()
        if normalize_title(title) == normalize_title(chapter_title):
            if not chapter_seen:
                output_lines.append(f"# {chapter_title}")
                chapter_seen = True
            continue

        if not chapter_seen:
            output_lines.append(f"# {chapter_title}")
            output_lines.append("")
            chapter_seen = True

        if level == 1:
            level = 2
        elif level > 4:
            level = 4
        output_lines.append(f"{'#' * level} {title}")

    if not chapter_seen:
        output_lines = [f"# {chapter_title}", ""] + output_lines

    normalized = "\n".join(output_lines).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", normalized)


def format_textbook_markdown(row: Dict, article_text: str) -> str:
    chapter_title = row.get("chapter_title", "Untitled Chapter")
    markdown = f"# {chapter_title}\n\n{demote_headings(article_text).strip()}\n"
    return normalize_final_markdown(row, markdown)


def extract_markdown_headings(markdown: str) -> List[Dict[str, Any]]:
    headings = []
    for line_no, line in enumerate(markdown.splitlines(), start=1):
        match = HEADING_RE.match(line.strip())
        if match:
            headings.append({
                "line": line_no,
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
            })
    return headings


def validate_output_structure(row: Dict, markdown: str, outline: Optional[str] = None) -> Dict[str, Any]:
    expected_section_count = len(row.get("section_blocks", []))
    headings = extract_markdown_headings(markdown)
    h1_titles = [heading["title"] for heading in headings if heading["level"] == 1]
    h2_titles = [heading["title"] for heading in headings if heading["level"] == 2]
    forbidden_lines = [
        {"line": line_no, "text": line.strip()}
        for line_no, line in enumerate(markdown.splitlines(), start=1)
        if is_book_chapter_label(row, line)
    ]
    issues = []
    chapter_title = row.get("chapter_title", "Untitled Chapter")

    if not headings or headings[0]["level"] != 1 or normalize_title(headings[0]["title"]) != normalize_title(chapter_title):
        issues.append("first_heading_must_be_chapter_title")
    if len(h1_titles) != 1:
        issues.append("exactly_one_chapter_heading_required")
    if len(h2_titles) != expected_section_count:
        issues.append("section_count_mismatch")
    if any(heading["level"] > 4 for heading in headings):
        issues.append("heading_depth_exceeds_subsubsection")
    if forbidden_lines:
        issues.append("book_chapter_label_present")

    outline_section_count = None
    if outline is not None:
        outline_section_count = len(extract_first_level_section_titles(outline))
        if outline_section_count != expected_section_count:
            issues.append("aligned_outline_section_count_mismatch")

    return {
        "id": row.get("id"),
        "valid": not issues,
        "issues": issues,
        "chapter_title": chapter_title,
        "expected_section_count": expected_section_count,
        "actual_section_count": len(h2_titles),
        "section_titles": h2_titles,
        "h1_titles": h1_titles,
        "max_heading_level": max((heading["level"] for heading in headings), default=0),
        "forbidden_lines": forbidden_lines,
        "outline_section_count": outline_section_count,
    }


def build_requirements_summary(row: Dict, section_metadata: Dict[str, str], max_chars: int = 18000) -> str:
    lines = [
        f"Chapter title: {row.get('chapter_title', '')}",
        f"Book title: {row.get('book_title', '')}",
        "",
    ]
    for section_title, metadata in section_metadata.items():
        lines.append(f"## {section_title}")
        lines.append(metadata)
        lines.append("")
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rsplit("\n", 1)[0]


def enforce_length(markdown: str, row: Dict, strong_lm, args, section_metadata: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    target_words, min_words, max_words = get_chapter_word_budget(row)
    current_words = count_markdown_words(markdown)
    report = {
        "target_words": target_words,
        "min_words": min_words,
        "max_words": max_words,
        "word_count": current_words,
        "length_attempts": 0,
    }
    if not args.enforce_length or target_words <= 0 or min_words <= current_words <= max_words:
        return markdown, report

    requirements = build_requirements_summary(row, section_metadata)
    revised = markdown
    for attempt in range(1, args.max_length_attempts + 1):
        current_words = count_markdown_words(revised)
        action = "expand" if current_words < min_words else "condense"
        prompt = f"""
You are revising a generated university textbook chapter so it satisfies the benchmark length budget.

Length budget:
- Required range: {min_words} to {max_words} words
- Target: approximately {target_words} words
- Current word count: {current_words} words
- Required action: {action} the chapter to land inside the required range, preferably near the target.

Rules:
1. Return the complete revised Markdown chapter, including exactly one chapter title as "# {row.get('chapter_title', 'Untitled Chapter')}".
2. Preserve this heading structure: "# Chapter", "## Section", "### Subsection", and "#### Subsubsection" only.
3. Cover the learning objectives and required knowledge units.
4. If expanding, add explanation, definitions, intuition, examples, contrasts, and interpretation. Do not add filler.
5. If condensing, remove redundancy while preserving required coverage.
6. Keep existing inline citation numbers where they support nearby claims. Do not invent new citation numbers.
7. Do not add a book title line, chapter label line, References section, source list, benchmark commentary, or process commentary.

Benchmark requirements:
{requirements}

Current chapter Markdown:
{revised}
""".strip()
        revised = strong_lm(prompt)[0].strip()
        report["length_attempts"] = attempt
        report["word_count"] = count_markdown_words(revised)
        if min_words <= report["word_count"] <= max_words:
            break
    return revised + "\n", report


def build_lms(args):
    fast_lm = OpenAIModel_dashscope(
        model=args.fast_model,
        max_tokens=args.max_fast_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=FAST_SYSTEM_PROMPT,
        enable_cache=args.enable_cache,
    )
    strong_lm = OpenAIModel_dashscope(
        model=args.strong_model,
        max_tokens=args.max_strong_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=TEXTBOOK_SYSTEM_PROMPT,
        enable_cache=args.enable_cache,
    )
    return fast_lm, strong_lm


def process_row(row: Dict, args, fast_lm, strong_lm, output_dir: Path) -> Dict:
    query_budget, source_budget = compute_budgets(row)
    seed_queries = build_seed_queries(row, query_budget)
    draft_outline = build_draft_outline(row)
    outline_context = build_outline_context(row)
    leakage_policy = build_chapter_leakage_policy(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['id']}.md"
    artifact_dir = get_row_artifact_dir(args, row)
    write_json_artifact(artifact_dir, "input_row.json", row)
    write_json_artifact(artifact_dir, "seed_queries.json", seed_queries)
    write_json_artifact(artifact_dir, "leakage_policy.json", {"phrases": leakage_policy["phrases"]})
    write_text_artifact(artifact_dir, "draft_outline.md", draft_outline)
    write_text_artifact(artifact_dir, "outline_context.md", outline_context)
    emit_json({
        "stage": "start",
        "id": row["id"],
        "query_budget": query_budget,
        "source_budget": source_budget,
        "seed_queries": len(seed_queries),
        "target_words": get_chapter_word_budget(row)[0],
        "word_range": list(get_chapter_word_budget(row)[1:]),
    })

    retriever = SerperSearch(
        k=args.retrievernum,
        query_budget=query_budget,
        source_budget=source_budget,
        is_valid_query=leakage_policy["query_allowed"],
        is_valid_source=leakage_policy["source_allowed"],
    )
    mind_map = MindMap(
        retriever=retriever,
        gen_concept_lm=fast_lm,
        depth=args.depth,
        workers=args.threadnum,
    )
    emit_json({"stage": "build_map", "id": row["id"]})
    for _ in mind_map.build_map(row["chapter_title"], initial_queries=seed_queries):
        pass
    if artifact_dir is not None and mind_map.root is not None:
        mind_map.save_map(mind_map.root, str(artifact_dir / "mind_map.json"))
        write_text_artifact(artifact_dir, "mind_map_concepts.txt", mind_map.export_categories_and_concepts())

    emit_json({"stage": "generate_outline", "id": row["id"]})
    outline_generator = OutlineGenerationModule(strong_lm)
    outline = outline_generator.generate_outline(
        topic=row["chapter_title"],
        mindmap=mind_map,
        draft_outline=draft_outline,
        outline_context=outline_context,
        section_count=len(row.get("section_blocks", [])),
        max_outline_attempts=args.max_outline_attempts,
    )
    outline_generator_validation = getattr(outline_generator, "last_outline_validation", {})
    write_json_artifact(artifact_dir, "outline_generator_validation.json", outline_generator_validation)
    if outline_generator_validation:
        emit_json({
            "stage": "outline_generator_validate",
            "id": row["id"],
            "valid": outline_generator_validation.get("valid"),
            "expected_section_count": outline_generator_validation.get("expected_section_count"),
            "actual_section_count": outline_generator_validation.get("actual_section_count"),
            "attempt_count": outline_generator_validation.get("attempt_count"),
            "issues": outline_generator_validation.get("issues", []),
        })
        if args.strict_outline and not outline_generator_validation.get("valid"):
            raise ValueError(
                f"Outline generator validation failed for {row['id']}: "
                f"{outline_generator_validation.get('issues', [])}"
            )
    write_text_artifact(artifact_dir, "raw_outline.md", outline)
    outline, section_titles = align_outline_to_source_sections(row, outline, draft_outline)
    write_text_artifact(artifact_dir, "aligned_outline.md", outline)
    section_metadata, section_queries = build_section_metadata(row, section_titles)
    write_json_artifact(artifact_dir, "section_metadata.json", section_metadata)
    write_json_artifact(artifact_dir, "section_queries.json", section_queries)

    article_with_outline = Article.from_outline_str(topic=row["chapter_title"], outline_str=outline)
    write_json_artifact(artifact_dir, "article_outline_tree.json", article_with_outline.get_outline_tree())
    article_generator = ArticleGenerationModule(
        retriever=retriever,
        article_gen_lm=strong_lm,
        retrieve_top_k=args.retrieve_top_k,
        max_thread_num=args.threadnum,
        agent_name="WriteTextbookSection",
        section_metadata=section_metadata,
        section_queries=section_queries,
    )
    emit_json({"stage": "generate_article", "id": row["id"]})
    article = article_generator.generate_article(
        topic=row["chapter_title"],
        mindmap=mind_map,
        article_with_outline=article_with_outline,
        language_style={"style": "university textbook", "language_type": args.language},
    )
    write_json_artifact(artifact_dir, "section_outputs.json", article_generator.last_section_outputs)
    write_text_artifact(artifact_dir, "article_raw.md", article.to_string())
    markdown = format_textbook_markdown(row, article.to_string())
    write_text_artifact(artifact_dir, "formatted_before_polish.md", markdown)
    output_path.write_text(markdown, encoding="utf-8")
    if not args.skip_polish:
        emit_json({"stage": "polish_article", "id": row["id"]})
        polisher = ArticlePolishingModule(article_gen_lm=strong_lm, article_polish_lm=strong_lm)
        article = polisher.polish_article(topic=row["chapter_title"], draft_article=article)
        write_text_artifact(artifact_dir, "article_polished_raw.md", article.to_string())
        markdown = format_textbook_markdown(row, article.to_string())
        write_text_artifact(artifact_dir, "formatted_after_polish.md", markdown)
        output_path.write_text(markdown, encoding="utf-8")

    emit_json({"stage": "enforce_length", "id": row["id"]})
    markdown, length_report = enforce_length(markdown, row, strong_lm, args, section_metadata)
    markdown = normalize_final_markdown(row, markdown)
    length_report["word_count"] = count_markdown_words(markdown)
    write_text_artifact(artifact_dir, "final.md", markdown)
    output_path.write_text(markdown, encoding="utf-8")

    validation_report = validate_output_structure(row, markdown, outline=outline)
    write_json_artifact(artifact_dir, "validation_report.json", validation_report)
    emit_json({
        "stage": "validate_output",
        "id": row["id"],
        "valid": validation_report["valid"],
        "expected_section_count": validation_report["expected_section_count"],
        "actual_section_count": validation_report["actual_section_count"],
        "issues": validation_report["issues"],
    })
    if args.strict_structure and not validation_report["valid"]:
        raise ValueError(f"Output structure validation failed for {row['id']}: {validation_report['issues']}")

    budget_report = retriever.get_budget_report()
    leakage_sources = collect_sources_from_mind_map_node(mind_map.root)
    leakage_sources.extend(collect_sources_from_section_outputs(article_generator.last_section_outputs))
    retriever_leakage_report = {
        **budget_report,
        **retriever.get_leakage_report(),
    }
    leakage_report = build_leakage_report(
        row,
        leakage_sources,
        retriever_report=retriever_leakage_report,
        provenance_available=True,
        strict=args.strict_leakage,
    )
    write_json_artifact(artifact_dir, "leakage_report.json", leakage_report)
    emit_json({
        "stage": "validate_leakage",
        "id": row["id"],
        "valid": leakage_report["valid"],
        "issues": leakage_report["issues"],
        "leakage_query_blocked_count": leakage_report["leakage_query_blocked_count"],
        "leakage_source_blocked_count": leakage_report["leakage_source_blocked_count"],
        "leaked_source_count": leakage_report["leaked_source_count"],
    })
    if args.strict_leakage and not leakage_report["valid"]:
        raise ValueError(f"Output leakage validation failed for {row['id']}: {leakage_report['issues']}")

    write_json_artifact(artifact_dir, "budget_report.json", budget_report)
    return {
        "id": row["id"],
        "path": str(output_path),
        "query_budget": query_budget,
        "source_budget": source_budget,
        "structure_valid": validation_report["valid"],
        "leakage_valid": leakage_report["valid"],
        "leakage_query_blocked_count": leakage_report["leakage_query_blocked_count"],
        "leakage_source_blocked_count": leakage_report["leakage_source_blocked_count"],
        "leaked_source_count": leakage_report["leaked_source_count"],
        "expected_section_count": validation_report["expected_section_count"],
        "actual_section_count": validation_report["actual_section_count"],
        **length_report,
        **budget_report,
    }


def skip_report(row: Dict, output_path: Path, args=None) -> Dict:
    markdown = output_path.read_text(encoding="utf-8")
    target_words, min_words, max_words = get_chapter_word_budget(row)
    validation_report = validate_output_structure(row, markdown)
    artifact_dir = get_row_artifact_dir(args, row) if args is not None else None
    sources, provenance_available = collect_existing_provenance_sources(artifact_dir)
    strict_leakage = bool(getattr(args, "strict_leakage", True)) if args is not None else True
    leakage_report = build_leakage_report(
        row,
        sources,
        provenance_available=provenance_available,
        strict=strict_leakage,
    )
    write_json_artifact(artifact_dir, "validation_report.json", validation_report)
    write_json_artifact(artifact_dir, "leakage_report.json", leakage_report)
    return {
        "id": row["id"],
        "path": str(output_path),
        "skipped": True,
        "reason": "output_exists",
        "target_words": target_words,
        "min_words": min_words,
        "max_words": max_words,
        "word_count": count_markdown_words(markdown),
        "structure_valid": validation_report["valid"],
        "expected_section_count": validation_report["expected_section_count"],
        "actual_section_count": validation_report["actual_section_count"],
        "structure_issues": validation_report["issues"],
        "leakage_valid": leakage_report["valid"],
        "leakage_issues": leakage_report["issues"],
        "leakage_query_blocked_count": leakage_report["leakage_query_blocked_count"],
        "leakage_source_blocked_count": leakage_report["leakage_source_blocked_count"],
        "leaked_source_count": leakage_report["leaked_source_count"],
    }


def existing_output_is_valid(row: Dict, output_path: Path, args) -> bool:
    markdown = output_path.read_text(encoding="utf-8")
    validation_report = validate_output_structure(row, markdown)
    artifact_dir = get_row_artifact_dir(args, row)
    sources, provenance_available = collect_existing_provenance_sources(artifact_dir)
    leakage_report = build_leakage_report(
        row,
        sources,
        provenance_available=provenance_available,
        strict=args.strict_leakage,
    )
    write_json_artifact(artifact_dir, "validation_report.json", validation_report)
    write_json_artifact(artifact_dir, "leakage_report.json", leakage_report)
    if validation_report["valid"] and leakage_report["valid"]:
        return True
    if not leakage_report["valid"]:
        emit_json({
            "stage": "resume_regenerate",
            "id": row["id"],
            "reason": "existing_output_leakage_invalid",
            "issues": leakage_report["issues"],
            "leaked_source_count": leakage_report["leaked_source_count"],
            "unverifiable": leakage_report["unverifiable"],
        })
        return False
    emit_json({
        "stage": "resume_regenerate",
        "id": row["id"],
        "reason": "existing_output_structure_invalid",
        "issues": validation_report["issues"],
        "expected_section_count": validation_report["expected_section_count"],
        "actual_section_count": validation_report["actual_section_count"],
    })
    return False


def process_row_with_fresh_lms(row: Dict, args) -> Dict:
    output_path = args.outputdir / f"{row['id']}.md"
    if args.resume and output_path.exists() and existing_output_is_valid(row, output_path, args):
        return skip_report(row, output_path, args)
    fast_lm, strong_lm = build_lms(args)
    return process_row(row, args, fast_lm, strong_lm, args.outputdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("chapter_benchmark_final_outline_blind.jsonl"))
    parser.add_argument("--outputdir", type=Path, default=Path("results/chapter_benchmark_textbooks"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fast-model", type=str, default="qwen3.6-flash")
    parser.add_argument("--strong-model", type=str, default="qwen3.7-plus")
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--retrievernum", type=int, default=5)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--threadnum", type=int, default=3)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-fast-tokens", type=int, default=1500)
    parser.add_argument("--max-strong-tokens", type=int, default=9000)
    parser.add_argument("--enforce-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length-attempts", type=int, default=2)
    parser.add_argument("--max-outline-attempts", type=int, default=2,
                        help="Maximum outline generation/repair attempts before validation fails.")
    parser.add_argument("--skip-polish", action="store_true")
    parser.add_argument("--chapter-workers", type=int, default=1,
                        help="Number of JSONL chapters to process concurrently.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows whose output Markdown file already exists.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop all processing when any chapter fails.")
    parser.add_argument("--rundir", type=Path, default=None,
                        help="Directory for run logs and intermediate artifacts.")
    parser.add_argument("--keep-intermediates", action=argparse.BooleanOptionalAction, default=True,
                        help="Persist row inputs, outlines, raw drafts, section outputs, and validation reports.")
    parser.add_argument("--strict-structure", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail a chapter when final Markdown does not match the dataset heading structure.")
    parser.add_argument("--strict-outline", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail a chapter when the outline generator does not produce the expected section count.")
    parser.add_argument("--strict-leakage", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail or regenerate outputs when source provenance is missing or contains leakage.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show tqdm progress for completed chapters.")
    args = parser.parse_args()

    load_dotenv()
    configure_run_logging(args)
    rows = list(iter_jsonl(args.input, limit=args.limit))
    emit_json({
        "stage": "loaded_rows",
        "count": len(rows),
        "chapter_workers": args.chapter_workers,
        "section_threads_per_chapter": args.threadnum,
        "resume": args.resume,
    })
    if args.chapter_workers <= 1:
        fast_lm, strong_lm = build_lms(args)
        row_iter = tqdm(rows, total=len(rows), desc="chapters", unit="chapter", disable=not args.progress)
        for row in row_iter:
            row_iter.set_postfix_str(row["id"][:48])
            output_path = args.outputdir / f"{row['id']}.md"
            if args.resume and output_path.exists() and existing_output_is_valid(row, output_path, args):
                emit_json(skip_report(row, output_path, args))
                continue
            report = process_row(row, args, fast_lm, strong_lm, args.outputdir)
            emit_json(report)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.chapter_workers) as executor:
        future_to_row = {
            executor.submit(process_row_with_fresh_lms, row, args): row
            for row in rows
        }
        futures = concurrent.futures.as_completed(future_to_row)
        future_iter = tqdm(
            futures,
            total=len(future_to_row),
            desc="chapters",
            unit="chapter",
            disable=not args.progress,
        )
        for future in future_iter:
            row = future_to_row[future]
            future_iter.set_postfix_str(row["id"][:48])
            try:
                emit_json(future.result())
            except Exception as exc:
                error_report = {
                    "id": row.get("id"),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                emit_json(error_report)
                if args.fail_fast:
                    raise


if __name__ == "__main__":
    main()
