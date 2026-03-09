#!/usr/bin/env python3
"""Stage the next FineWeb-Edu shard chunk into local shared storage.

This script is intended for HPC workflows where compute nodes cannot access
HuggingFace directly. Run it on a login node with internet access to download
the next chunk of parquet shards into:

  /fs1/proj/educational_web_data/dataset/fineweb-edu/<config>/source

The script keeps a state file outside each source directory so repeated
invocations download the next chunk instead of starting over. Multiple configs
can be staged in one command by repeating ``--config``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


HF_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_STORAGE_ROOT = os.environ.get("FINEWEB_STORAGE_ROOT", "/fs1/proj/educational_web_data")


def default_output_dir(config: str) -> str:
    return os.path.join(DEFAULT_STORAGE_ROOT, "dataset", "fineweb-edu", config, "source")


def default_state_path(config: str) -> str:
    return os.path.join(DEFAULT_STORAGE_ROOT, "dataset", "fineweb-edu", config, ".download_state.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the next staged FineWeb shard chunk")
    p.add_argument(
        "--config",
        action="append",
        choices=[
            "CC-MAIN-2025-05",
            "CC-MAIN-2025-08",
            "CC-MAIN-2025-13",
            "CC-MAIN-2025-18",
            "CC-MAIN-2025-21",
            "CC-MAIN-2025-26",
        ],
        help="FineWeb-Edu config to stage. Repeat to stage multiple configs.",
    )
    p.add_argument("--max-gb", type=float, default=500.0, help="Approximate chunk size to stage")
    p.add_argument("--output-dir", type=str, default=None, help="Where to place the staged parquet files")
    p.add_argument("--state-path", type=str, default=None, help="Download state file path")
    p.add_argument("--reset", action="store_true", default=False, help="Reset chunk progress to the beginning")
    args = p.parse_args()
    if not args.config:
        args.config = ["CC-MAIN-2025-26"]
    if len(args.config) > 1 and (args.output_dir or args.state_path):
        raise SystemExit("--output-dir and --state-path only support a single --config at a time.")
    return args


def list_parquet_shards(config: str) -> list[dict[str, Any]]:
    api = HfApi()
    files = api.list_repo_tree(HF_DATASET, repo_type="dataset", path_in_repo=f"data/{config}")
    shards = []
    for file_info in files:
        if hasattr(file_info, "rfilename") and file_info.rfilename.endswith(".parquet"):
            shards.append(
                {
                    "path": file_info.rfilename,
                    "size": getattr(file_info, "size", None),
                }
            )
    shards.sort(key=lambda item: item["path"])
    if not shards:
        raise RuntimeError(f"No parquet shards found for {config} in {HF_DATASET}")
    return shards


def load_state(state_path: str, total_shards: int, *, reset: bool = False) -> dict[str, Any]:
    if reset or not os.path.exists(state_path):
        return {
            "next_shard_idx": 0,
            "last_chunk_start_idx": 0,
            "last_chunk_end_idx": 0,
            "total_shards": total_shards,
            "completed": False,
        }

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("next_shard_idx", 0)
    state.setdefault("last_chunk_start_idx", 0)
    state.setdefault("last_chunk_end_idx", 0)
    state["total_shards"] = total_shards
    state["completed"] = state["next_shard_idx"] >= total_shards
    return state


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def clear_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        else:
            os.remove(full_path)


def stage_config(config: str, *, max_gb: float, output_dir: str, state_path: str, reset: bool) -> dict[str, Any]:
    max_bytes = int(max_gb * 1e9)
    shards = list_parquet_shards(config)
    state = load_state(state_path, len(shards), reset=reset)
    start_idx = int(state["next_shard_idx"])

    if start_idx >= len(shards):
        print(f"All {len(shards)} shards for {config} have already been staged.")
        print("Use --reset if you want to restart from the beginning.")
        return {
            "config": config,
            "output_dir": output_dir,
            "state_path": state_path,
            "completed": True,
            "next_shard_idx": start_idx,
            "total_shards": len(shards),
            "total_bytes": 0,
        }

    clear_directory(output_dir)
    print(f"Staging shards for {config} into {output_dir}")
    print(f"Starting at shard index {start_idx} of {len(shards)}")

    downloaded = []
    total_bytes = 0

    for shard in shards[start_idx:]:
        shard_path = shard["path"]
        estimated = shard.get("size")
        if downloaded and estimated and total_bytes + estimated > max_bytes:
            break

        local_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=shard_path,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        actual_size = os.path.getsize(local_path)
        total_bytes += actual_size
        downloaded.append({"path": shard_path, "local_path": local_path, "size": actual_size})
        print(f"  downloaded: {shard_path} ({actual_size / 1e9:.2f} GB, total {total_bytes / 1e9:.2f} GB)")

        if total_bytes >= max_bytes:
            break

    if not downloaded:
        raise RuntimeError(f"No shards were downloaded for {config}. Check network access and output directory permissions.")

    next_idx = start_idx + len(downloaded)
    manifest = {
        "config": config,
        "output_dir": output_dir,
        "max_gb": max_gb,
        "current_shard_idx": start_idx,
        "next_shard_idx": next_idx,
        "total_shards": len(shards),
        "completed": next_idx >= len(shards),
        "total_bytes": total_bytes,
        "downloaded": downloaded,
    }
    save_json(os.path.join(output_dir, "_chunk_manifest.json"), manifest)

    state.update(
        {
            "next_shard_idx": next_idx,
            "last_chunk_start_idx": start_idx,
            "last_chunk_end_idx": next_idx,
            "total_shards": len(shards),
            "completed": next_idx >= len(shards),
        }
    )
    save_json(state_path, state)

    print()
    print(f"{config}: chunk ready: shard range [{start_idx}, {next_idx})")
    print(f"{config}: state file: {state_path}")
    if next_idx < len(shards):
        print(f"{config}: run the downloader again later to stage the next chunk.")
    else:
        print(f"{config}: this was the final chunk for the selected config.")
    print()

    return {
        "config": config,
        "output_dir": output_dir,
        "state_path": state_path,
        "completed": next_idx >= len(shards),
        "next_shard_idx": next_idx,
        "total_shards": len(shards),
        "total_bytes": total_bytes,
    }


def main() -> None:
    args = parse_args()
    results = []

    for config in args.config:
        output_dir = os.path.abspath(args.output_dir or default_output_dir(config))
        state_path = os.path.abspath(args.state_path or default_state_path(config))
        result = stage_config(
            config,
            max_gb=args.max_gb,
            output_dir=output_dir,
            state_path=state_path,
            reset=args.reset,
        )
        results.append(result)

    print("Next step:")
    print("  submit the Star sbatch job to train the staged config set")
    staged_dirs = ":".join(result["output_dir"] for result in results)
    print(f"  LOCAL_DATA_DIRS={staged_dirs}")
    print(f"  CONFIGS={':'.join(result['config'] for result in results)}")


if __name__ == "__main__":
    main()
