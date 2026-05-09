#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@dataclass
class TaskStats:
    index: int
    total: int
    domain: str
    task_id: str
    status: str
    outcome: str
    raw_result: str
    steps: int
    client_error: bool
    started_at: str
    ended_at: str
    elapsed_sec: float
    example_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    gui_root = repo_root / "GUI-Docker-Env"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-all-meta-path",
        type=Path,
        default=gui_root / "evaluation_examples" / "test_all.json",
    )
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--base-url", type=str, default=os.environ.get("OSWORLD_BASE_URL", "http://127.0.0.1:50003"))
    parser.add_argument("--token", type=str, default=os.environ.get("OSWORLD_TOKEN", "dart"))
    parser.add_argument("--result-dir", type=Path, default=gui_root / f"results_debug_client_error_{stamp}")
    parser.add_argument("--test-config-base-dir", type=Path, default=gui_root / "evaluation_examples")
    parser.add_argument("--model", type=str, default="ui_tars_1.5")
    parser.add_argument("--model-type", type=str, default="qwen25vl")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-trajectory-length", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action-space", type=str, default="pyautogui")
    parser.add_argument("--observation-type", type=str, default="screenshot")
    parser.add_argument("--os-type", type=str, default="Ubuntu")
    parser.add_argument("--infer-mode", type=str, default="qwen25vl_normal")
    parser.add_argument("--prompt-style", type=str, default="qwen25vl_normal")
    parser.add_argument("--input-swap", action="store_true")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--max-pixels", type=float, default=16384 * 28 * 28)
    parser.add_argument("--min-pixels", type=float, default=100 * 28 * 28)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--history-n", type=int, default=5)
    parser.add_argument("--callusr-tolerance", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--stop-token", type=str, default=None)
    parser.add_argument("--enable-proxy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_run_uitars(gui_root: Path):
    spec = importlib.util.spec_from_file_location("run_uitars", gui_root / "uitars_run" / "run_uitars.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_tasks(meta_path: Path, domain: str, task_limit: int) -> list[tuple[str, str]]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if domain != "all":
        data = {domain: data[domain]}
    tasks = [(task_domain, task_id) for task_domain in sorted(data) for task_id in data[task_domain]]
    if task_limit > 0:
        tasks = tasks[:task_limit]
    return tasks


def load_traj_records(traj_path: Path) -> list[dict[str, Any]]:
    if not traj_path.exists():
        return []
    records = []
    for line in traj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def has_client_error(records: list[dict[str, Any]]) -> bool:
    return any(
        record.get("action") == "DONE"
        and "client error" in str(record.get("response", "")).lower()
        for record in records
    )


def parse_outcome(raw_result: str) -> str:
    text = raw_result.strip()
    if not text:
        return "missing"
    try:
        value = float(text)
    except ValueError:
        return f"score={text}"
    if value == 1.0:
        return "success"
    if value == 0.0:
        return "fail"
    return f"score={text}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_task_stats(jsonl_path: Path, csv_path: Path, stats: TaskStats) -> None:
    row = asdict(stats)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def example_dir_for(args: argparse.Namespace, domain: str, task_id: str) -> Path:
    return (
        Path(args.result_dir)
        / args.action_space
        / args.observation_type
        / args.model
        / domain
        / task_id
    )


def build_run_args(run_uitars, cli_args: argparse.Namespace) -> argparse.Namespace:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["run_uitars.py"]
        args = run_uitars.config()
    finally:
        sys.argv = original_argv
    args.base_url = cli_args.base_url
    args.token = cli_args.token
    args.max_workers = 1
    args.domain = cli_args.domain
    args.test_all_meta_path = str(cli_args.test_all_meta_path)
    args.test_config_base_dir = str(cli_args.test_config_base_dir)
    args.result_dir = str(cli_args.result_dir)
    args.model = cli_args.model
    args.model_type = cli_args.model_type
    args.max_steps = cli_args.max_steps
    args.max_trajectory_length = cli_args.max_trajectory_length
    args.headless = cli_args.headless
    args.action_space = cli_args.action_space
    args.observation_type = cli_args.observation_type
    args.os_type = cli_args.os_type
    args.infer_mode = cli_args.infer_mode
    args.prompt_style = cli_args.prompt_style
    args.input_swap = cli_args.input_swap
    args.language = cli_args.language
    args.max_pixels = cli_args.max_pixels
    args.min_pixels = cli_args.min_pixels
    args.temperature = cli_args.temperature
    args.top_p = cli_args.top_p
    args.top_k = cli_args.top_k
    args.history_n = cli_args.history_n
    args.callusr_tolerance = cli_args.callusr_tolerance
    args.max_tokens = cli_args.max_tokens
    args.stop_token = cli_args.stop_token
    args.enable_proxy = cli_args.enable_proxy
    args.overwrite = cli_args.overwrite
    return args


def main() -> int:
    cli_args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    gui_root = repo_root / "GUI-Docker-Env"
    cli_args.result_dir = cli_args.result_dir.resolve()
    meta_path = cli_args.test_all_meta_path.resolve()
    config_base_dir = cli_args.test_config_base_dir.resolve()
    session_dir = cli_args.result_dir / "_debug_osworld"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_log_path = session_dir / "session.log"
    task_stats_jsonl = session_dir / "task_stats.jsonl"
    task_stats_csv = session_dir / "task_stats.csv"
    summary_path = session_dir / "summary.json"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with session_log_path.open("a", encoding="utf-8") as log_handle:
        tee = Tee(original_stdout, log_handle)
        sys.stdout = tee
        sys.stderr = tee
        try:
            os.chdir(gui_root)
            os.environ["OSWORLD_BASE_URL"] = cli_args.base_url
            os.environ["OSWORLD_TOKEN"] = cli_args.token
            started_at = iso_now()
            run_uitars = load_run_uitars(gui_root)
            args = build_run_args(run_uitars, cli_args)
            tasks = load_tasks(meta_path, cli_args.domain, cli_args.task_limit)

            summary = {
                "run_started_at": started_at,
                "run_ended_at": None,
                "base_url": cli_args.base_url,
                "token": cli_args.token,
                "domain": cli_args.domain,
                "task_limit": cli_args.task_limit,
                "total_tasks": len(tasks),
                "completed_tasks": 0,
                "stopped_on_client_error": False,
                "client_error_task_id": None,
                "client_error_task_dir": None,
                "result_dir": str(cli_args.result_dir),
                "session_log": str(session_log_path),
                "task_stats_jsonl": str(task_stats_jsonl),
                "task_stats_csv": str(task_stats_csv),
                "runner_logs": {
                    "normal": str(gui_root / "logs" / f"normal-{run_uitars.datetime_str}.log"),
                    "debug": str(gui_root / "logs" / f"debug-{run_uitars.datetime_str}.log"),
                    "sdebug": str(gui_root / "logs" / f"sdebug-{run_uitars.datetime_str}.log"),
                },
            }
            write_json(summary_path, summary)

            if not run_uitars.ping(args.base_url):
                summary["run_ended_at"] = iso_now()
                write_json(summary_path, summary)
                return 1

            print(f"RESULT_DIR {cli_args.result_dir}")
            print(f"SESSION_LOG {session_log_path}")
            print(f"TASK_COUNT {len(tasks)}")

            for index, (domain, task_id) in enumerate(tasks, 1):
                task_started_at = iso_now()
                task_started_monotonic = time.monotonic()
                print(f"RUN_START {index}/{len(tasks)} {domain}/{task_id}")
                result = run_uitars.run_one_example(args, domain, task_id)
                task_ended_at = iso_now()
                elapsed_sec = round(time.monotonic() - task_started_monotonic, 3)

                task_dir = example_dir_for(args, domain, task_id)
                traj_path = task_dir / "traj.jsonl"
                result_path = task_dir / "result.txt"
                records = load_traj_records(traj_path)
                client_error = has_client_error(records)
                raw_result = result_path.read_text(encoding="utf-8").strip() if result_path.exists() else ""
                stats = TaskStats(
                    index=index,
                    total=len(tasks),
                    domain=domain,
                    task_id=task_id,
                    status=result.get("status", "unknown"),
                    outcome=parse_outcome(raw_result),
                    raw_result=raw_result,
                    steps=len(records),
                    client_error=client_error,
                    started_at=task_started_at,
                    ended_at=task_ended_at,
                    elapsed_sec=elapsed_sec,
                    example_dir=str(task_dir),
                )
                append_task_stats(task_stats_jsonl, task_stats_csv, stats)

                summary["completed_tasks"] = index
                if client_error:
                    summary["stopped_on_client_error"] = True
                    summary["client_error_task_id"] = task_id
                    summary["client_error_task_dir"] = str(task_dir)
                write_json(summary_path, summary)

                print(
                    f"RUN_DONE {index}/{len(tasks)} {domain}/{task_id} "
                    f"status={stats.status} outcome={stats.outcome} result={stats.raw_result or 'MISSING'} "
                    f"steps={stats.steps} elapsed_sec={stats.elapsed_sec} "
                    f"started_at={stats.started_at} ended_at={stats.ended_at} "
                    f"client_error={str(stats.client_error).lower()} dir={stats.example_dir}"
                )
                if client_error:
                    print(f"FIRST_CLIENT_ERROR {domain}/{task_id} {task_dir}")
                    break
            else:
                print("NO_CLIENT_ERROR_FOUND")

            summary["run_ended_at"] = iso_now()
            write_json(summary_path, summary)
            return 0
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
