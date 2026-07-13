"""Real PyTorch benchmark suite for llm.cpp.

Measures the things an ML/AI engineer usually asks for:
model metadata, tokenizer/data throughput, forward latency, training-step
latency, autoregressive generation latency, memory, and JSON/CSV output.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
ENGINE_INFERENCE = ROOT / "engine" / "inference.py"
DEFAULT_DATA = ROOT / "engine" / "input.txt"
DEFAULT_OUT = ROOT / "benchmark" / "results"


def load_engine_module():
    spec = importlib.util.spec_from_file_location(
        "quadtrix_engine_inference", ENGINE_INFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ENGINE_INFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_ms(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    return {
        "avg_ms": mean,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p90_ms": percentile(samples, 0.90),
        "p95_ms": percentile(samples, 0.95),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def sync(torch: Any, device: Any) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed_samples(
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
    runs: int,
    warmup: int,
) -> tuple[list[float], Any]:
    last = None
    for _ in range(warmup):
        last = fn()
    sync(torch, device)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        last = fn()
        sync(torch, device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, last


def cuda_memory(torch: Any, device: Any) -> dict[str, float]:
    if not str(device).startswith("cuda"):
        return {}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def process_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        return None


@dataclass
class BenchRow:
    suite: str
    name: str
    backend: str
    batch_size: int = 0
    sequence_length: int = 0
    tokens: int = 0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    std_ms: float = 0.0
    tokens_per_sec: float = 0.0
    samples: int = 0
    loss: float | None = None
    memory_mb: float | None = None
    notes: str = ""


class QuadtrixPythonBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = load_engine_module()
        self.torch = __import__("torch")
        self.torch.manual_seed(args.seed)
        self.device = self.engine.device
        self.rows: list[BenchRow] = []

        if str(self.device).startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)

        self.model = self._make_model()
        self.model.eval()

    def _make_model(self):
        checkpoint = Path(
            self.args.checkpoint) if self.args.checkpoint else self.engine.default_checkpoint_path()
        if checkpoint.exists() and not self.args.random_weights:
            return self.engine.load_model(checkpoint)

        model = self.engine.GPTLanguageModel().to(self.device)
        model.eval()
        return model

    def _record(self, row: BenchRow) -> None:
        self.rows.append(row)
        print(
            f"{row.suite:<14} {row.name:<24} "
            f"avg={row.avg_ms:9.3f} ms  "
            f"p95={row.p95_ms:9.3f} ms  "
            f"tok/s={row.tokens_per_sec:10.1f}"
        )

    def run(self) -> dict[str, Any]:
        print("Benchmark")
        print(f"Device: {self.device}")
        print(f"Runs: {self.args.runs}, warmup: {self.args.warmup}")

        self.bench_tokenizer_and_data()
        self.bench_primitives()
        self.bench_forward()
        self.bench_training_step()
        self.bench_generation()

        return self.save()

    def bench_tokenizer_and_data(self) -> None:
        data_path = Path(self.args.data)
        text = data_path.read_text(
            encoding="utf-8") if data_path.exists() else "llm.cpp-benchmark text. " * 512
        if self.args.max_data_chars and len(text) > self.args.max_data_chars:
            text = text[: self.args.max_data_chars]
        tokenizer = self.engine.tokenizer

        samples, encoded = timed_samples(
            self.torch,
            self.device,
            lambda: tokenizer.encode(text),
            self.args.runs,
            self.args.warmup,
        )
        stats = summarize_ms(samples)
        self._record(
            BenchRow(
                suite="data",
                name="tokenizer_encode",
                backend="python",
                tokens=len(encoded),
                tokens_per_sec=len(encoded) / (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )

        tensor = self.torch.tensor(encoded, dtype=self.torch.long)
        max_block = min(self.engine.block_size, max(2, len(tensor) - 2))
        batch_size = min(self.args.batch_size, max(
            1, len(tensor) - max_block - 1))

        def make_batch():
            ix = self.torch.randint(len(tensor) - max_block - 1, (batch_size,))
            x = self.torch.stack([tensor[i: i + max_block]
                                 for i in ix]).to(self.device)
            y = self.torch.stack([tensor[i + 1: i + max_block + 1]
                                 for i in ix]).to(self.device)
            return x, y

        samples, _ = timed_samples(
            self.torch, self.device, make_batch, self.args.runs, self.args.warmup)
        stats = summarize_ms(samples)
        self._record(
            BenchRow(
                suite="data",
                name="batch_sample_to_device",
                backend="python",
                batch_size=batch_size,
                sequence_length=max_block,
                tokens=batch_size * max_block,
                tokens_per_sec=(batch_size * max_block) /
                (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )

    def bench_primitives(self) -> None:
        torch = self.torch
        cases = [
            ("matmul_3d", (1, 16, self.engine.n_embd),
             (self.engine.n_embd, self.engine.n_embd)),
            ("matmul_3d", (self.args.batch_size, self.engine.block_size,
             self.engine.n_embd), (self.engine.n_embd, self.engine.n_embd)),
            ("attention_scores", (self.args.batch_size,
             self.engine.block_size, self.engine.n_embd), None),
        ]
        for name, x_shape, w_shape in cases:
            x = torch.randn(*x_shape, device=self.device)
            if name == "attention_scores":
                def fn(): return torch.softmax(
                    (x @ x.transpose(-2, -1)) / math.sqrt(x_shape[-1]), dim=-1)
            else:
                w = torch.randn(*w_shape, device=self.device)
                def fn(): return x @ w
            samples, _ = timed_samples(
                torch, self.device, fn, self.args.runs, self.args.warmup)
            stats = summarize_ms(samples)
            tokens = x_shape[0] * x_shape[1]
            self._record(
                BenchRow(
                    suite="primitive",
                    name=f"{name}_{x_shape[0]}x{x_shape[1]}",
                    backend="python",
                    batch_size=x_shape[0],
                    sequence_length=x_shape[1],
                    tokens=tokens,
                    tokens_per_sec=tokens / (stats["avg_ms"] / 1000.0),
                    samples=len(samples),
                    memory_mb=process_rss_mb(),
                    **stats,
                )
            )

    def bench_forward(self) -> None:
        torch = self.torch
        cases = [(1, 8), (1, self.engine.block_size),
                 (self.args.batch_size, self.engine.block_size)]
        self.model.eval()
        for batch_size, seq_len in cases:
            idx = torch.randint(self.engine.vocab_size,
                                (batch_size, seq_len), device=self.device)
            targets = torch.randint(
                self.engine.vocab_size, (batch_size, seq_len), device=self.device)

            @torch.no_grad()
            def fn():
                return self.model(idx, targets)

            samples, last = timed_samples(
                torch, self.device, fn, self.args.runs, self.args.warmup)
            stats = summarize_ms(samples)
            loss = float(last[1].item())
            tokens = batch_size * seq_len
            self._record(
                BenchRow(
                    suite="forward",
                    name=f"batch{batch_size}_seq{seq_len}",
                    backend="python",
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    tokens=tokens,
                    tokens_per_sec=tokens / (stats["avg_ms"] / 1000.0),
                    samples=len(samples),
                    loss=loss,
                    memory_mb=process_rss_mb(),
                    **stats,
                )
            )

    def bench_training_step(self) -> None:
        torch = self.torch
        model = self.engine.GPTLanguageModel().to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.learning_rate)
        batch_size = self.args.batch_size
        seq_len = self.engine.block_size
        idx = torch.randint(self.engine.vocab_size,
                            (batch_size, seq_len), device=self.device)
        targets = torch.randint(self.engine.vocab_size,
                                (batch_size, seq_len), device=self.device)

        def fn():
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()
            return loss.detach()

        samples, loss = timed_samples(
            torch, self.device, fn, self.args.train_steps, self.args.warmup)
        stats = summarize_ms(samples)
        tokens = batch_size * seq_len
        self._record(
            BenchRow(
                suite="training",
                name=f"adamw_step_b{batch_size}_s{seq_len}",
                backend="python",
                batch_size=batch_size,
                sequence_length=seq_len,
                tokens=tokens,
                tokens_per_sec=tokens / (stats["avg_ms"] / 1000.0),
                samples=len(samples),
                loss=float(loss.item()),
                memory_mb=process_rss_mb(),
                **stats,
            )
        )
        del model
        gc.collect()

    def bench_generation(self) -> None:
        torch = self.torch
        tokenizer = self.engine.tokenizer
        prompts = [
            ("empty", ""),
            ("short", "The future of local AI is"),
            ("long", "llama.cpp is a compact transformer benchmark that measures " * 4),
        ]
        self.model.eval()
        for label, prompt in prompts:
            encoded = tokenizer.encode(prompt) or [0]
            encoded = encoded[-self.engine.block_size:]
            idx = torch.tensor([encoded], dtype=torch.long, device=self.device)

            @torch.no_grad()
            def fn():
                return self.model.generate(idx, self.args.generate_tokens, temperature=1.0, top_k=self.args.top_k)

            samples, _ = timed_samples(
                torch, self.device, fn, self.args.runs, self.args.warmup)
            stats = summarize_ms(samples)
            self._record(
                BenchRow(
                    suite="generation",
                    name=label,
                    backend="python",
                    batch_size=1,
                    sequence_length=len(encoded),
                    tokens=self.args.generate_tokens,
                    tokens_per_sec=self.args.generate_tokens /
                        (stats["avg_ms"] / 1000.0),
                    samples=len(samples),
                    memory_mb=process_rss_mb(),
                    **stats,
                )
            )

    def save(self) -> dict[str, Any]:
        out_dir = Path(self.args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_params = sum(p.numel() for p in self.model.parameters())
        result = {
            "schema_version": 1,
            "timestamp": now_iso(),
            "backend": "python",
            "system": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": self.torch.__version__,
                "device": str(self.device),
                "cuda": getattr(self.torch.version, "cuda", None),
                "rss_mb": process_rss_mb(),
                **cuda_memory(self.torch, self.device),
            },
            "model": {
                "vocab_size": self.engine.vocab_size,
                "block_size": self.engine.block_size,
                "n_embd": self.engine.n_embd,
                "n_head": self.engine.n_head,
                "n_layer": self.engine.n_layer,
                "dropout": self.engine.dropout,
                "parameters": n_params,
                "parameter_mb_fp32": n_params * 4 / (1024**2),
            },
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()},
            "results": [asdict(row) for row in self.rows],
        }
        json_path = out_dir / "python_benchmark.json"
        csv_path = out_dir / "python_benchmark.csv"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(asdict(self.rows[0]).keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        print(f"Saved {json_path}")
        print(f"Saved {csv_path}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--generate-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-data-chars", type=int, default=1_000_000)
    parser.add_argument("--random-weights", action="store_true",
                        help="Do not load a checkpoint even if one exists.")
    parser.add_argument("--quick", action="store_true",
                        help="Short run for smoke tests.")
    args = parser.parse_args()
    if args.quick:
        args.runs = 2
        args.warmup = 1
        args.train_steps = 1
        args.generate_tokens = 4
        args.max_data_chars = min(args.max_data_chars, 50_000)
    return args


def main() -> int:
    try:
        benchmark = QuadtrixPythonBenchmark(parse_args())
        benchmark.run()
        return 0
    except ImportError as exc:
        print(f"Missing Python benchmark dependency: {exc}", file=sys.stderr)
        print("Install the engine requirements, including torch and tiktoken.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
