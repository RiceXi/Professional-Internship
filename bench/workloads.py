"""Inference traces and conversation datasets as simulator workload.

The simulator advances in ticks, one tick per decode iteration, so a real
trace has to be projected onto that clock. ``tick_seconds`` states the
assumption explicitly (default 0.05 s per decode step); it changes the
arrival pattern but not the comparison between memory strategies.

The contiguous baseline reserves ``maximum`` slots per request. Traces do not
declare a maximum length, so ``max_tokens`` stands for the server-side
``max_model_len``; longer requests are dropped and counted instead of being
silently truncated.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory.simulation import Request  # noqa: E402

AZURE_COLUMNS = ('TIMESTAMP', 'ContextTokens', 'GeneratedTokens')
CJK_RANGES = ((0x3000, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))
PROMPT_ROLES = ('human', 'user', 'system')
OUTPUT_ROLES = ('gpt', 'assistant', 'chatgpt')


@dataclass(frozen=True)
class Geometry:
    """Physical pool the trace was measured with, so a replay can reuse it."""

    frames: int
    block_size: int
    host_pages: int


@dataclass(frozen=True)
class LoadedTrace:
    """Requests plus the provenance needed to interpret and replay them."""

    kind: str
    source: str
    requests: list[Request] = field(default_factory=list)
    dropped_too_long: int = 0
    note: str = ''
    geometry: Geometry | None = None

    def __len__(self) -> int:
        return len(self.requests)


def estimate_tokens(text: str) -> int:
    """Approximate token count for text-only sources.

    CJK characters are counted as one token each, other characters as a
    quarter token each. Callers that need exact counts pass a tokenizer.
    """
    cjk = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES))
    return max(1, cjk + round((len(text) - cjk) / 4))


def parse_timestamp(value: str) -> datetime:
    """Accept ISO timestamps with any number of fractional digits (Python 3.10)."""
    value = value.strip()
    if '.' in value:
        head, fraction = value.split('.', 1)
        value = f'{head}.{fraction[:6]}'
    return datetime.fromisoformat(value)


def _validate(tick_seconds: float, max_tokens: int, limit: int | None) -> None:
    if tick_seconds <= 0:
        raise ValueError('tick_seconds must be positive')
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')


def _poisson_arrivals(rate: float, count: int, seed: int) -> list[int]:
    if rate <= 0:
        raise ValueError('arrival_rate must be positive')
    rng = random.Random(seed)
    clock = 0.0
    arrivals = []
    for _ in range(count):
        clock += rng.expovariate(rate)
        arrivals.append(int(clock))
    return arrivals


def load_azure_csv(path: str | Path, *, tick_seconds: float = 0.05, max_tokens: int = 2048,
                   limit: int | None = None) -> LoadedTrace:
    """Load the Azure LLM inference trace: real arrivals and real token counts."""
    _validate(tick_seconds, max_tokens, limit)
    path = Path(path)
    requests: list[Request] = []
    dropped = 0
    first: datetime | None = None
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in AZURE_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f'{path}: missing columns {missing}')
        for row in reader:
            stamp = parse_timestamp(row['TIMESTAMP'])
            if first is None:
                first = stamp
            length = int(row['ContextTokens']) + int(row['GeneratedTokens'])
            if length <= 0:
                raise ValueError(f'{path}: non-positive sequence length')
            if length > max_tokens:
                dropped += 1
                continue
            arrival = int(round((stamp - first).total_seconds() / tick_seconds))
            requests.append(Request(len(requests), arrival, length, max_tokens))
            if limit is not None and len(requests) == limit:
                break
    if not requests:
        raise ValueError(f'{path}: no request fits max_tokens={max_tokens}')
    return LoadedTrace(
        kind='azure-llm-trace',
        source=str(path),
        requests=requests,
        dropped_too_long=dropped,
        note=f'real arrivals and lengths; tick={tick_seconds}s per decode step',
    )


def load_sharegpt(path: str | Path, *, tick_seconds: float = 0.05, max_tokens: int = 2048,
                  arrival_rate: float = 1.0, limit: int | None = None, seed: int = 42,
                  count_tokens=None) -> LoadedTrace:
    """Load ShareGPT-style conversations; arrivals are a synthetic Poisson stream."""
    _validate(tick_seconds, max_tokens, limit)
    counter = count_tokens or estimate_tokens
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    conversations = payload['conversations'] if isinstance(payload, dict) else payload
    if not isinstance(conversations, list):
        raise ValueError(f'{path}: expected a list of conversations')

    pairs: list[tuple[int, int]] = []
    for item in conversations:
        turns = item.get('conversations') if isinstance(item, dict) else None
        if not isinstance(turns, list):
            raise ValueError(f'{path}: conversation entry without "conversations"')
        prompt = ''.join(str(t.get('value', '')) for t in turns
                         if str(t.get('from', '')).lower() in PROMPT_ROLES)
        output = ''.join(str(t.get('value', '')) for t in turns
                         if str(t.get('from', '')).lower() in OUTPUT_ROLES)
        if not prompt:
            raise ValueError(f'{path}: conversation without prompt turns')
        pairs.append((counter(prompt), counter(output) if output else 0))

    kept = [(prompt, output) for prompt, output in pairs if prompt + output <= max_tokens]
    dropped = len(pairs) - len(kept)
    if limit is not None:
        kept = kept[:limit]
    if not kept:
        raise ValueError(f'{path}: no conversation fits max_tokens={max_tokens}')
    arrivals = _poisson_arrivals(arrival_rate, len(kept), seed)
    requests = [Request(i, arrivals[i], prompt + output, max_tokens)
                for i, (prompt, output) in enumerate(kept)]
    return LoadedTrace(
        kind='sharegpt',
        source=str(path),
        requests=requests,
        dropped_too_long=dropped,
        note=f'text lengths estimated at {counter.__name__}; arrivals Poisson(rate={arrival_rate}/tick, seed={seed})',
    )


def load_longbench(path: str | Path, *, tick_seconds: float = 0.05, max_tokens: int = 32768,
                   output_tokens: int = 128, arrival_rate: float = 1.0,
                   limit: int | None = None, seed: int = 42, count_tokens=None) -> LoadedTrace:
    """Load LongBench jsonl; prompts are real, output length is assumed."""
    _validate(tick_seconds, max_tokens, limit)
    if output_tokens < 0:
        raise ValueError('output_tokens must be non-negative')
    counter = count_tokens or estimate_tokens
    path = Path(path)
    pairs: list[tuple[int, int]] = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if 'context' not in item and 'input' not in item:
            raise ValueError(f'{path}:{line_no}: expected a "context" or "input" field')
        prompt = f"{item.get('context', '')}\n{item.get('input', '')}".strip()
        pairs.append((counter(prompt), output_tokens))

    kept = [(prompt, output) for prompt, output in pairs if prompt + output <= max_tokens]
    dropped = len(pairs) - len(kept)
    if limit is not None:
        kept = kept[:limit]
    if not kept:
        raise ValueError(f'{path}: no sample fits max_tokens={max_tokens}')
    arrivals = _poisson_arrivals(arrival_rate, len(kept), seed)
    requests = [Request(i, arrivals[i], prompt + output, max_tokens)
                for i, (prompt, output) in enumerate(kept)]
    return LoadedTrace(
        kind='longbench',
        source=str(path),
        requests=requests,
        dropped_too_long=dropped,
        note=f'prompt lengths estimated at {counter.__name__}; output assumed {output_tokens} tokens',
    )


def load_lengths(path: str | Path) -> LoadedTrace:
    """Replay a committed length distribution without the original text."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    entries = payload.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{path}: expected a non-empty "entries" list')
    raw_geometry = payload.get('geometry')
    geometry = Geometry(int(raw_geometry['frames']), int(raw_geometry['block_size']),
                        int(raw_geometry['host_pages'])) if raw_geometry else None
    requests = [Request(i, int(e['arrival']), int(e['tokens']), int(e['maximum']))
                for i, e in enumerate(entries)]
    return LoadedTrace(
        kind=payload.get('kind', 'lengths'),
        source=f"{path} (from {payload.get('origin', 'unknown')})",
        requests=requests,
        note=str(payload.get('note', '')),
        geometry=geometry,
    )


def dump_lengths(trace: LoadedTrace, path: str | Path, *, origin: str | None = None,
                 geometry: Geometry | None = None) -> Path:
    """Write the derived arrival/length distribution; contains no source text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'kind': trace.kind,
        'origin': origin or trace.source,
        'note': trace.note,
        'dropped_too_long': trace.dropped_too_long,
        'geometry': (asdict(geometry) if geometry else None),
        'entries': [{'arrival': r.arrival, 'tokens': r.tokens, 'maximum': r.maximum}
                    for r in trace.requests],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


LOADERS = {
    'azure': load_azure_csv,
    'sharegpt': load_sharegpt,
    'longbench': load_longbench,
    'lengths': load_lengths,
}
