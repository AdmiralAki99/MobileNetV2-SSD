from pathlib import Path
from typing import Any
import json
import re

_CKPT_INDEX_RE = re.compile(r"^(?P<prefix>.+-)?(?P<stem>ckpt-(?P<step>\d+))\.index$")

# Keys that define model architecture — if these differ between the saved
# checkpoint config and the current config, the variable shapes will not
# match and restore will either crash or silently load garbage weights.
_ARCHITECTURE_KEYS = ("num_classes", "backbone", "heads", "priors", "input_size")

# Keys that affect training behaviour but NOT variable shapes.
# Safe to change between runs (e.g. lower LR for fine-tuning).
_TRAINING_KEYS = ("optimizer", "scheduler", "train", "augmentation", "eval", "sampler", "loss", "matcher")


def discover_checkpoint(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Convenience: if the caller passed ".../checkpoints", descend into "last/"
    if checkpoint_dir.name == "checkpoints":
        checkpoint_dir = checkpoint_dir / "last"

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None

    # Scan for .index files and keep the one with the highest step number
    best = None  # (step, ckpt_prefix_path)
    for idx in checkpoint_dir.glob("*.index"):
        m = _CKPT_INDEX_RE.match(idx.name)
        if not m:
            continue
        step = int(m.group("step"))
        ckpt_prefix = idx.with_suffix("")  # ".../ckpt-200.index" -> ".../ckpt-200"
        if best is None or step > best[0]:
            best = (step, ckpt_prefix)

    if best is None:
        return None

    step, ckpt_prefix = best

    return {
        "ckpt_path": ckpt_prefix,
        "step": step,
        "checkpoint_dir": checkpoint_dir,
        "has_best_dir": (checkpoint_dir.parent / "best").exists(),
    }



def find_latest_run_by_fingerprint(runs_root: Path, fingerprint_short: str):
    runs_root = Path(runs_root)

    if not runs_root.exists() or not runs_root.is_dir():
        return None

    # --- Step 1: find experiment directories whose fingerprint.json matches ---
    matching_experiment_dirs = []

    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue

        fp_file = candidate / "fingerprint.json"
        if not fp_file.exists():
            continue

        try:
            with open(fp_file, "r") as f:
                fp_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Compare the authoritative "short" field, not the directory name
        if fp_data.get("short") == fingerprint_short:
            matching_experiment_dirs.append(candidate)

    if not matching_experiment_dirs:
        return None

    # --- Step 2: inside each matching experiment, scan timestamp dirs
    #     newest-first, and return the first one with a real checkpoint ---

    for experiment_dir in matching_experiment_dirs:
        logs_dir = experiment_dir / "logs"
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue

        # Timestamp dirs are YYYYMMDD_HHMMSS — lexicographic sort = chronological
        timestamp_dirs = sorted(
            [d for d in logs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,  # newest first
        )

        for ts_dir in timestamp_dirs:
            ckpt_last = ts_dir / "checkpoints" / "last"
            result = discover_checkpoint(ckpt_last)
            if result is not None:
                # Attach the experiment-level directory so the caller can
                # read config.json / fingerprint.json from it
                result["experiment_dir"] = experiment_dir
                return result

    return None


# Returns (is_compatible, warnings)
#   is_compatible = False  →  architecture mismatch, restore will fail
#   is_compatible = True   →  safe to restore; warnings list what changed
#
def _get_nested(d: dict, key: str, default=None):
    """Fetch a possibly-nested key like 'train.amp.enabled'."""
    parts = key.split(".")
    for part in parts:
        if not isinstance(d, dict):
            return default
        d = d.get(part, default)
    return d


def validate_checkpoint_compatibility(
    saved_config: dict[str, Any],
    current_config: dict[str, Any],
) -> tuple[bool, list[str]]:
    warnings = []
    is_compatible = True

    # --- Check architecture keys (must match) ---
    for key in _ARCHITECTURE_KEYS:
        saved_val = saved_config.get(key)
        current_val = current_config.get(key)

        if saved_val != current_val:
            warnings.append(
                f"ARCHITECTURE MISMATCH '{key}': "
                f"saved={saved_val}, current={current_val}"
            )
            is_compatible = False

    # --- Check AMP config (special case — affects optimizer checkpoint structure) ---
    saved_amp_enabled = _get_nested(saved_config, "train.amp.enabled", False)
    current_amp_enabled = _get_nested(current_config, "train.amp.enabled", False)
    saved_amp_policy = _get_nested(saved_config, "train.amp.policy", "float32")
    current_amp_policy = _get_nested(current_config, "train.amp.policy", "float32")

    if saved_amp_enabled != current_amp_enabled:
        warnings.append(
            f"AMP MISMATCH: saved amp.enabled={saved_amp_enabled}, "
            f"current amp.enabled={current_amp_enabled}  "
            f"— optimizer checkpoint keys will not match"
        )
        is_compatible = False

    if saved_amp_enabled and current_amp_enabled and saved_amp_policy != current_amp_policy:
        warnings.append(
            f"AMP POLICY CHANGED: saved={saved_amp_policy}, "
            f"current={current_amp_policy}"
        )

    # --- Check training keys (safe to differ — just warn) ---
    for key in _TRAINING_KEYS:
        saved_val = saved_config.get(key)
        current_val = current_config.get(key)

        if saved_val != current_val:
            warnings.append(
                f"Training config changed '{key}': "
                f"saved and current differ (this is safe)"
            )

    return is_compatible, warnings



def load_saved_config(experiment_dir: Path) -> dict[str, Any] | None:
    config_path = Path(experiment_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None



def collect_resumable_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs_root = Path(runs_root)

    if not runs_root.exists() or not runs_root.is_dir():
        return []

    candidates = []

    for experiment_dir in runs_root.iterdir():
        if not experiment_dir.is_dir():
            continue

        # --- Read experiment-level metadata ---
        fp_data = _read_json(experiment_dir / "fingerprint.json")
        status_data = _read_json(experiment_dir / "status.json")
        config_data = _read_json(experiment_dir / "config.json")

        # Skip if we can't read the fingerprint (not a valid run directory)
        if fp_data is None:
            continue

        # Skip completed runs — nothing to resume
        status = status_data.get("status", "unknown") if status_data else "unknown"
        if status == "completed":
            continue

        fingerprint_short = fp_data.get("short", "?")
        experiment_id = "?"
        if config_data:
            experiment_id = config_data.get("experiment", {}).get("id", "?")

        # --- Scan timestamp dirs for checkpoints ---
        logs_dir = experiment_dir / "logs"
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue

        for ts_dir in sorted(
            [d for d in logs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,  # newest first
        ):
            ckpt_result = discover_checkpoint(ts_dir / "checkpoints" / "last")
            if ckpt_result is None:
                continue

            # Format the timestamp for display: "20260210_163102" -> "2026-02-10 16:31"
            ts_raw = ts_dir.name
            ts_display = ts_raw
            if len(ts_raw) == 15 and ts_raw[8] == "_":
                ts_display = (
                    f"{ts_raw[0:4]}-{ts_raw[4:6]}-{ts_raw[6:8]} "
                    f"{ts_raw[9:11]}:{ts_raw[11:13]}"
                )

            candidates.append({
                "experiment_id": experiment_id,
                "fingerprint_short": fingerprint_short,
                "status": status,
                "timestamp": ts_display,
                "step": ckpt_result["step"],
                "ckpt_path": ckpt_result["ckpt_path"],
                "checkpoint_dir": ckpt_result["checkpoint_dir"],
                "has_best_dir": ckpt_result["has_best_dir"],
                "experiment_dir": experiment_dir,
            })

    # Sort by timestamp descending (newest first) then by step descending
    candidates.sort(key=lambda c: (c["timestamp"], c["step"]), reverse=True)

    return candidates


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# 6. select_run_interactive  –  print a numbered list, return user's choice
# ---------------------------------------------------------------------------
def select_run_interactive(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        print("\n  No resumable runs found.\n")
        return None

    print("\n  Available runs to resume:\n")

    for i, c in enumerate(candidates, start=1):
        print(
            f"  [{i}]  {c['experiment_id']}"
            f"  |  {c['fingerprint_short']}"
            f"  |  step {c['step']}"
            f"  |  {c['timestamp']}"
            f"  |  {c['status']}"
        )

    print(f"\n  [0]  Start fresh (no resume)\n")

    while True:
        try:
            choice = input("  Select [0-{}]: ".format(len(candidates))).strip()
            if not choice:
                continue
            idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
            print(f"  Please enter a number between 0 and {len(candidates)}")
        except ValueError:
            print("  Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment or user pressed Ctrl+C
            print()
            return None


if __name__ == "__main__":
    # Quick smoke tests
    print("=== discover_checkpoint ===")
    print(discover_checkpoint(Path("./runs/exp001_2d8a0d709f40/logs/20260210_163102/checkpoints")))

    print("\n=== find_latest_run_by_fingerprint ===")
    print(find_latest_run_by_fingerprint(Path("./runs"), "2d8a0d709f40"))

    print("\n=== collect_resumable_runs + interactive select ===")
    runs = collect_resumable_runs(Path("./runs"))
    if runs:
        selected = select_run_interactive(runs)
        print(f"\nSelected: {selected}")
    else:
        print("No resumable runs found")

