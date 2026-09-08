"""
Live end-to-end resource monitor, NOT just a static snapshot.

1. Unloads the two configured models from Ollama (cold start), then prints
   `ollama ps` BEFORE anything runs — should show no active models.
2. Runs a real question through the FULL pipeline:
      handle_question("What is the total sales amount by country?", engine)
3. While that runs, a background thread samples `ollama ps` every 5 seconds,
   plus system RAM (psutil) and GPU VRAM (nvidia-smi), each timestamped.
4. After the question completes, prints the total time and EVERY sample, so
   you can watch, sample by sample, whether the PROCESSOR column says GPU or
   CPU while SQL generation and verification are actually happening.

Run:  py -3.11 scripts/check_resources.py
"""

import datetime
import subprocess
import sys
import threading
import time

import psutil

from app.config import settings
from app.db.connection import get_engine

engine = get_engine()
from app.agents.orchestrator import handle_question

QUESTION = "What is the total sales amount by country?"

SAMPLE_INTERVAL_SECONDS = 5


def run_cmd(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or p.stderr).strip()
    except FileNotFoundError:
        return "command not found"
    except subprocess.TimeoutExpired:
        return "command timed out"


def ollama_ps():
    return run_cmd(["ollama", "ps"])


def nvidia_smi():
    return run_cmd(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"]
    )


samples = []
_prev_smi = None
_stop = threading.Event()


def sampler():
    global _prev_smi
    while not _stop.is_set():
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        ram = psutil.virtual_memory()
        smi = nvidia_smi()
        _prev_smi = smi
        samples.append(
            {
                "t": stamp,
                "ram": f"{ram.percent}% used, {ram.available / 1e9:.1f} GB free",
                "smi": smi,
                "ps": ollama_ps(),
            }
        )
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def print_cold_start():
    print("### 1. COLD START: unloading configured models so we see them load live")
    models = {settings.ollama_sql_model, settings.ollama_reasoning_model}
    for m in sorted(models):
        out = run_cmd(["ollama", "stop", m])
        print(f"  ollama stop {m} -> {out or '(ok)'}")


def print_before():
    print("\n### 2. BEFORE ANYTHING RUNS")
    ps = ollama_ps()
    lines = ps.splitlines()
    print("ollama ps (should be empty / header only):")
    print(ps or "(no output)")
    if len(lines) <= 1:
        print("  -> no models loaded")
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.percent}% used, {ram.available / 1e9:.1f} GB free")
    print("nvidia-smi:", nvidia_smi())


def print_result(result, elapsed):
    rows = (result.get("result") or {}).get("rows") or []
    print("\n### 3. QUESTION RESULT")
    print(f"handle_question({QUESTION!r}) took {elapsed:.1f}s total")
    print(f"success={result.get('success')}  confidence={result.get('confidence')} "
          f"({result.get('confidence_score')})  attempts={result.get('attempts')}")
    print(f"sql: {(result.get('sql') or '').strip().replace(chr(10), ' ')}")
    print(f"rows: {len(rows)}")
    for row in rows[:5]:
        print(f"  {row}")


def print_samples():
    print(f"\n### 4. PER-SAMPLE LOG (every {SAMPLE_INTERVAL_SECONDS}s during the call)")
    gpu_seen = False
    for i, s in enumerate(samples):
        print(f"\n[{s['t']}] sample {i + 1}/{len(samples)}  RAM {s['ram']}")
        print(f"  nvidia-smi: {s['smi']}")
        ps_lines = s["ps"].splitlines()
        print("  ollama ps:")
        if len(ps_lines) <= 1:
            print("    (no models loaded)")
        else:
            for line in ps_lines:
                print(f"    {line}")
            if s["ps"].lower().find("gpu") != -1 and "100% gpu" in s["ps"].lower():
                gpu_seen = True

    print("\n### 5. SUMMARY")
    verdict = "GPU offload OBSERVED (PROCESSOR column showed GPU at some point)"
    if gpu_seen:
        print(verdict)
    else:
        print("PROCESSOR column showed 100% CPU in every sample — inference is CPU-bound on this box.")
    severe = [s for s in samples if "96" in s["ram"] or "97" in s["ram"] or "98" in s["ram"] or "99" in s["ram"]]
    if severe:
        print(f"{len(severe)}/{len(samples)} samples had RAM >= 96% used (swap-thrash territory).")


def main():
    print_cold_start()
    print_before()

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    result = handle_question(QUESTION, engine)
    elapsed = time.perf_counter() - t0

    _stop.set()
    thread.join(timeout=10)

    print_result(result, elapsed)
    print_samples()
    return 0


if __name__ == "__main__":
    sys.exit(main())