import dspy
import re
from src.tools.mindmap import MindMap
from src.utils.ArticleTextProcessing import ArticleTextProcessing
from typing import Dict, List, Optional, Tuple, Union

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _normalize_title(title: str) -> str:
    title = title.strip().strip("*_")
    title = re.sub(r"\s+", " ", title)
    return title.lower()


def _extract_heading_titles(outline: str, level: int) -> List[str]:
    titles = []
    for line in outline.splitlines():
        match = HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == level:
            titles.append(match.group(2).strip())
    return titles


def _shift_heading_levels(outline: str, delta: int) -> str:
    shifted_lines = []
    for line in outline.splitlines():
        match = HEADING_RE.match(line.strip())
        if not match:
            continue
        new_level = min(max(len(match.group(1)) + delta, 1), 6)
        shifted_lines.append(f"{'#' * new_level} {match.group(2).strip()}")
    return "\n".join(shifted_lines)


def _strip_topic_heading(topic: str, outline: str) -> Tuple[str, bool]:
    lines = [line.strip() for line in outline.splitlines() if line.strip()]
    if not lines:
        return outline, False
    first_match = HEADING_RE.match(lines[0])
    if not first_match or len(first_match.group(1)) != 1:
        return outline, False
    if _normalize_title(first_match.group(2)) != _normalize_title(topic):
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
        return _shift_heading_levels("\n".join(remaining), delta=-1), True
    return "\n".join(remaining), True


def validate_textbook_outline(topic: str, outline: str, expected_section_count: Optional[int]) -> Tuple[str, Dict]:
    cleaned_outline = ArticleTextProcessing.clean_up_outline(outline)
    normalized_outline, removed_topic_heading = _strip_topic_heading(topic, cleaned_outline)
    normalized_outline = ArticleTextProcessing.clean_up_outline(normalized_outline)

    first_level_titles = _extract_heading_titles(normalized_outline, level=1)
    promoted_second_level = False
    if expected_section_count is not None and len(first_level_titles) != expected_section_count:
        second_level_titles = _extract_heading_titles(normalized_outline, level=2)
        if len(second_level_titles) == expected_section_count:
            normalized_outline = _shift_heading_levels(normalized_outline, delta=-1)
            normalized_outline = ArticleTextProcessing.clean_up_outline(normalized_outline)
            first_level_titles = _extract_heading_titles(normalized_outline, level=1)
            promoted_second_level = True

    issues = []
    if expected_section_count is not None and len(first_level_titles) != expected_section_count:
        issues.append("section_count_mismatch")
    if any(_normalize_title(title) == _normalize_title(topic) for title in first_level_titles):
        issues.append("chapter_title_present_as_section")

    report = {
        "valid": not issues,
        "issues": issues,
        "expected_section_count": expected_section_count,
        "actual_section_count": len(first_level_titles),
        "section_titles": first_level_titles,
        "removed_topic_heading": removed_topic_heading,
        "promoted_second_level_sections": promoted_second_level,
    }
    return normalized_outline, report

class OutlineGenerationModule():

    def __init__(self,
                 outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(self,
                         topic: str,
                         mindmap: MindMap,
                         draft_outline: Optional[str] = None,
                         outline_context: Optional[str] = None,
                         section_count: Optional[int] = None,
                         max_outline_attempts: int = 2,
                         ):

        concepts = mindmap.export_categories_and_concepts()
        result = self.write_outline(
            topic=topic,
            concepts=concepts,
            draft_outline=draft_outline,
            outline_context=outline_context,
            section_count=section_count,
            max_outline_attempts=max_outline_attempts,
        )
        self.last_outline_validation = self.write_outline.last_outline_validation

        return result


class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.polish_page_outline = dspy.Predict(PolishPageOutline)
        self.textbook_outline = dspy.Predict(WriteTextbookOutline)
        self.repair_textbook_outline = dspy.Predict(RepairTextbookOutline)
        self.engine = engine
        self.last_outline_validation = {}

    def forward(self,
                topic: str,
                concepts: str,
                draft_outline: Optional[str] = None,
                outline_context: Optional[str] = None,
                section_count: Optional[int] = None,
                max_outline_attempts: int = 2):
        
        with dspy.settings.context(lm=self.engine):
            if draft_outline or outline_context:
                expected_count = _parse_expected_count(section_count)
                outline = self.textbook_outline(
                    topic=topic,
                    draft_outline=draft_outline or "",
                    outline_context=outline_context or "",
                    concepts=concepts,
                    section_count=str(expected_count or ""),
                ).outline
                outline, validation = validate_textbook_outline(topic, outline, expected_count)
                attempts = [{
                    "attempt": 1,
                    "action": "initial_generation",
                    "outline": outline,
                    **validation,
                }]

                max_outline_attempts = max(int(max_outline_attempts or 1), 1)
                for attempt in range(2, max_outline_attempts + 1):
                    if validation["valid"]:
                        break
                    outline = self.repair_textbook_outline(
                        topic=topic,
                        draft_outline=draft_outline or "",
                        outline_context=outline_context or "",
                        concepts=concepts,
                        section_count=str(expected_count or ""),
                        current_outline=outline,
                        actual_section_count=str(validation["actual_section_count"]),
                        validation_issues=", ".join(validation["issues"]),
                    ).outline
                    outline, validation = validate_textbook_outline(topic, outline, expected_count)
                    attempts.append({
                        "attempt": attempt,
                        "action": "repair_generation",
                        "outline": outline,
                        **validation,
                    })

                self.last_outline_validation = {
                    **validation,
                    "attempts": attempts,
                    "attempt_count": len(attempts),
                }
            else:
                outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_page_outline(topic=topic).outline)
                outline = ArticleTextProcessing.clean_up_outline(
                    self.polish_page_outline(draft=outline, concepts=concepts).outline)
                self.last_outline_validation = {}

        return outline


def _parse_expected_count(section_count: Optional[int]) -> Optional[int]:
    if section_count is None:
        return None
    try:
        parsed = int(section_count)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class WriteTextbookOutline(dspy.Signature):
    """
    Write a university textbook chapter outline from benchmark section requirements and retrieved concepts.
    Here is the format of your writing:
    1. Use "#" Title" for first-level section titles, "##" Title" for subsections, and "###" Title" only if needed.
    2. Output only the Markdown outline. Do not include prose, notes, references, or the chapter title itself.
    3. Create exactly the requested number of first-level sections when section_count is provided. Count only lines beginning with "# " as sections.
    4. Preserve the order and coverage of the draft/source sections, but improve section titles and add useful textbook subsections.
    5. Do not write the chapter title as a heading. The first heading must be a section heading.
    6. The outline should support a university textbook chapter, not a Wikipedia article.
    """

    topic = dspy.InputField(prefix="Chapter title: ", format=str)
    draft_outline = dspy.InputField(prefix="Draft/source outline:\n", format=str)
    outline_context = dspy.InputField(prefix="Benchmark section requirements:\n", format=str)
    concepts = dspy.InputField(prefix="Retrieved concepts to consider:\n", format=str)
    section_count = dspy.InputField(prefix="Required number of first-level sections: ", format=str)
    outline = dspy.OutputField(prefix="Write the textbook chapter outline:\n", format=str)


class RepairTextbookOutline(dspy.Signature):
    """
    Repair a university textbook chapter outline so its section count exactly matches the benchmark.
    Required output format:
    1. Output only Markdown headings.
    2. Use "# " only for top-level section titles, "## " for subsections, and "### " for subsubsections.
    3. The number of "# " headings must be exactly the requested section_count.
    4. Do not include the chapter title, prose, notes, references, or a sources section.
    5. Preserve the order and coverage of the draft/source sections.
    """

    topic = dspy.InputField(prefix="Chapter title to exclude from outline headings: ", format=str)
    draft_outline = dspy.InputField(prefix="Draft/source outline:\n", format=str)
    outline_context = dspy.InputField(prefix="Benchmark section requirements:\n", format=str)
    concepts = dspy.InputField(prefix="Retrieved concepts to consider:\n", format=str)
    section_count = dspy.InputField(prefix="Required number of # section headings: ", format=str)
    current_outline = dspy.InputField(prefix="Current invalid outline:\n", format=str)
    actual_section_count = dspy.InputField(prefix="Current number of # section headings: ", format=str)
    validation_issues = dspy.InputField(prefix="Validation issues to fix: ", format=str)
    outline = dspy.OutputField(prefix="Write the corrected textbook chapter outline:\n", format=str)


class PolishPageOutline(dspy.Signature):
    """
    Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the concept learned from an information-seeking to make it more informative.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    draft = dspy.InputField(prefix="Current outline:\n ", format=str)
    concepts = dspy.InputField(prefix="The information you learned from the conversation:\n", format=str)
    outline = dspy.OutputField(prefix='Write the page outline:\n', format=str)


class WritePageOutline(dspy.Signature):
    """
    Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)
