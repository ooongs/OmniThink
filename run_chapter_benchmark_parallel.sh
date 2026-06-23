#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT="${INPUT:-chapter_benchmark_final_outline_blind.jsonl}"
OUTPUTDIR="${OUTPUTDIR:-results/chapter_benchmark_textbooks}"
RUN_ROOT="${RUN_ROOT:-$OUTPUTDIR/_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUNDIR="${RUNDIR:-$RUN_ROOT/$RUN_ID}"
CHAPTER_WORKERS="${CHAPTER_WORKERS:-2}"
SECTION_THREADS="${SECTION_THREADS:-1}"
FAST_MODEL="${FAST_MODEL:-qwen3.6-flash}"
STRONG_MODEL="${STRONG_MODEL:-qwen3.7-plus}"
RETRIEVER_NUM="${RETRIEVER_NUM:-5}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-3}"
DEPTH="${DEPTH:-1}"
MAX_OUTLINE_ATTEMPTS="${MAX_OUTLINE_ATTEMPTS:-2}"

mkdir -p "$RUNDIR"

cmd=(
  python -m examples.chapter_benchmark_textbook
  --input "$INPUT"
  --outputdir "$OUTPUTDIR"
  --rundir "$RUNDIR"
  --fast-model "$FAST_MODEL"
  --strong-model "$STRONG_MODEL"
  --depth "$DEPTH"
  --retrievernum "$RETRIEVER_NUM"
  --retrieve-top-k "$RETRIEVE_TOP_K"
  --max-outline-attempts "$MAX_OUTLINE_ATTEMPTS"
  --threadnum "$SECTION_THREADS"
  --chapter-workers "$CHAPTER_WORKERS"
  --enable-cache
  --skip-polish
  --resume
)

if [[ -n "${LIMIT:-}" ]]; then
  cmd+=(--limit "$LIMIT")
fi

printf '%q ' "${cmd[@]}" "$@" > "$RUNDIR/command.txt"
printf '\n' >> "$RUNDIR/command.txt"

"${cmd[@]}" "$@" 2>&1 | tee "$RUNDIR/run.terminal.log"
exit "${PIPESTATUS[0]}"
