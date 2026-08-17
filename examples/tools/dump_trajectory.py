#!/usr/bin/env python3
"""Decode a materialized Gateway trajectory into readable segments and TSV.

The AgentFramework writes ``trajectory.json`` and ``trajectory.npz`` under
``<log_dir>/step_<step>/<session_id>/``. The JSON contains session/reward
metadata; the NPZ contains exact token ids, response masks, and optional
rollout log-probabilities.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np


def _resolve_inputs(entry: str) -> tuple[Path, Path]:
    path = Path(entry).expanduser()
    if path.is_dir():
        return path / "trajectory.npz", path / "trajectory.json"
    if path.suffix == ".npz":
        return path, path.with_suffix(".json")
    if path.name == "trajectory.json":
        return path.with_suffix(".npz"), path
    raise ValueError("expected a session directory, trajectory.npz, or trajectory.json")


def _load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[warn] transformers is unavailable; printing token ids", file=sys.stderr)
        return None
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:
        print(
            f"[warn] could not load local tokenizer {name!r} ({type(exc).__name__}); printing token ids",
            file=sys.stderr,
        )
        return None


def _mask_runs(mask: Sequence[int]) -> Iterator[tuple[int, int, int]]:
    start = 0
    while start < len(mask):
        value = int(mask[start])
        end = start + 1
        while end < len(mask) and int(mask[end]) == value:
            end += 1
        yield value, start, end
        start = end


def _decode(ids: Sequence[int], tokenizer) -> str:
    if tokenizer is None:
        return " ".join(f"<{token_id}>" for token_id in ids)
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def _prompt_window(ids: list[int], max_tokens: int | None) -> tuple[list[int], list[int], int]:
    if max_tokens is None or len(ids) <= max_tokens:
        return ids, [], 0
    head_len = max_tokens // 2
    tail_len = max_tokens - head_len
    return ids[:head_len], ids[-tail_len:] if tail_len else [], len(ids) - max_tokens


def _trajectory_labels(data: np.lib.npyio.NpzFile) -> list[str]:
    labels = {key.split("_", 1)[0] for key in data.files if key.startswith("traj")}
    return sorted(labels, key=lambda label: int(label[4:]) if label[4:].isdigit() else label)


def _validate_trajectory(data: np.lib.npyio.NpzFile, label: str) -> None:
    required = [f"{label}_prompt_ids", f"{label}_response_ids", f"{label}_response_mask"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{label} is missing arrays: {missing}")
    response_len = len(data[f"{label}_response_ids"])
    mask_len = len(data[f"{label}_response_mask"])
    if response_len != mask_len:
        raise ValueError(f"{label} response_ids={response_len}, response_mask={mask_len}")
    logprobs_key = f"{label}_response_logprobs"
    if logprobs_key in data.files and len(data[logprobs_key]) != response_len:
        raise ValueError(f"{label} response_logprobs is not aligned with response_ids")


def _write_summary(
    stream: TextIO,
    *,
    data: np.lib.npyio.NpzFile,
    metadata: dict[str, Any] | None,
    tokenizer,
    max_segments: int | None,
    max_prompt_tokens: int | None,
) -> None:
    if metadata:
        print(f"# session_id: {metadata.get('session_id')}", file=stream)
        print(f"# num_trajectories: {metadata.get('num_trajectories')}", file=stream)
        for index, trajectory in enumerate(metadata.get("trajectories", [])):
            print(
                f"# traj{index}: turns={trajectory.get('num_turns')} "
                f"prompt={trajectory.get('prompt_len')} response={trajectory.get('response_len')} "
                f"model_tokens={trajectory.get('model_token_count')} "
                f"reward={trajectory.get('reward_score')} "
                f"materialization={trajectory.get('materialization_reason')}",
                file=stream,
            )
            print(f"# traj{index} reward_info: {trajectory.get('reward_info')}", file=stream)
        print(file=stream)

    for label in _trajectory_labels(data):
        _validate_trajectory(data, label)
        prompt_ids = data[f"{label}_prompt_ids"].astype(np.int64).tolist()
        response_ids = data[f"{label}_response_ids"].astype(np.int64).tolist()
        response_mask = data[f"{label}_response_mask"].astype(np.int8).tolist()
        logprobs_key = f"{label}_response_logprobs"
        logprobs = data[logprobs_key].astype(np.float64).tolist() if logprobs_key in data.files else None
        runs = list(_mask_runs(response_mask))

        print(
            f"===== {label}: prompt={len(prompt_ids)} response={len(response_ids)} "
            f"model_tokens={sum(response_mask)} segments={len(runs)} "
            f"logprobs={'yes' if logprobs is not None else 'no'} =====",
            file=stream,
        )
        head, tail, omitted = _prompt_window(prompt_ids, max_prompt_tokens)
        print("\n----- PROMPT -----", file=stream)
        print(_decode(head, tokenizer), file=stream)
        if omitted:
            print(f"\n[... {omitted} prompt tokens omitted ...]\n", file=stream)
            print(_decode(tail, tokenizer), file=stream)

        for segment_index, (mask_value, start, end) in enumerate(runs, start=1):
            if max_segments is not None and segment_index > max_segments:
                print(f"\n[... {len(runs) - max_segments} response segments omitted ...]", file=stream)
                break
            role = "MODEL" if mask_value == 1 else "CONTEXT"
            print(f"\n--- SEGMENT-{segment_index} [{role}] tokens {start}:{end} (len={end - start}) ---", file=stream)
            text = _decode(response_ids[start:end], tokenizer).rstrip()
            print(text if text else "<empty>", file=stream)
            if mask_value == 1 and logprobs is not None:
                span = logprobs[start:end]
                print(
                    f"[logprobs] n={len(span)} min={min(span):.3f} "
                    f"mean={sum(span) / len(span):.3f} max={max(span):.3f}",
                    file=stream,
                )
        print(file=stream)


def _write_tsv(path: Path, data: np.lib.npyio.NpzFile, metadata: dict[str, Any] | None, tokenizer) -> None:
    rewards = {
        f"traj{index}": trajectory.get("reward_score")
        for index, trajectory in enumerate((metadata or {}).get("trajectories", []))
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("traj\tpos\trole\ttoken_id\ttext\tmask\tmodel_segment\tlogp\tis_reward_pos\treward\n")
        for label in _trajectory_labels(data):
            _validate_trajectory(data, label)
            response_ids = data[f"{label}_response_ids"].astype(np.int64).tolist()
            response_mask = data[f"{label}_response_mask"].astype(np.int8).tolist()
            logprobs_key = f"{label}_response_logprobs"
            logprobs = data[logprobs_key].astype(np.float64).tolist() if logprobs_key in data.files else None
            model_segment = 0
            for pos, token_id in enumerate(response_ids):
                if response_mask[pos] == 1 and (pos == 0 or response_mask[pos - 1] == 0):
                    model_segment += 1
                role = f"MODEL-{model_segment}" if response_mask[pos] == 1 else "CONTEXT"
                token_text = _decode([token_id], tokenizer).replace("\n", "\\n").replace("\t", "\\t")
                logp = "" if logprobs is None else f"{logprobs[pos]:.6f}"
                is_reward_pos = int(pos == len(response_ids) - 1 and rewards.get(label) is not None)
                reward = rewards.get(label) if is_reward_pos else ""
                stream.write(
                    f"{label}\t{pos}\t{role}\t{token_id}\t{token_text}\t{response_mask[pos]}\t"
                    f"{model_segment}\t{logp}\t{is_reward_pos}\t{reward}\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", help="session directory, trajectory.npz, or trajectory.json")
    parser.add_argument("--out", help="output directory for trajectory.txt and tokens.tsv; omit for stdout only")
    parser.add_argument("--tokenizer", help="local Hugging Face tokenizer name or path")
    parser.add_argument("--max-turns", type=int, dest="max_segments", help="maximum response segments to print")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1000,
        help="prompt head+tail token budget; use --no-prompt-window for the full prompt",
    )
    parser.add_argument("--no-prompt-window", action="store_true", help="print the complete prompt")
    args = parser.parse_args()

    if args.max_segments is not None and args.max_segments <= 0:
        parser.error("--max-turns must be greater than zero")
    if args.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be greater than zero")

    try:
        npz_path, json_path = _resolve_inputs(args.entry)
    except ValueError as exc:
        parser.error(str(exc))
    if not npz_path.is_file():
        parser.error(f"missing {npz_path}")

    metadata = None
    if json_path.is_file():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    tokenizer = _load_tokenizer(args.tokenizer)
    max_prompt_tokens = None if args.no_prompt_window else args.max_prompt_tokens

    with np.load(npz_path, allow_pickle=False) as data:
        summary = io.StringIO()
        _write_summary(
            summary,
            data=data,
            metadata=metadata,
            tokenizer=tokenizer,
            max_segments=args.max_segments,
            max_prompt_tokens=max_prompt_tokens,
        )
        rendered = summary.getvalue()
        print(rendered, end="")
        if args.out:
            output_dir = Path(args.out).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "trajectory.txt").write_text(rendered, encoding="utf-8")
            _write_tsv(output_dir / "tokens.tsv", data, metadata, tokenizer)
            print(f"# wrote {output_dir / 'trajectory.txt'}")
            print(f"# wrote {output_dir / 'tokens.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
