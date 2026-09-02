"""
Batch runner for run_differencing_experiment.py.

Executes a grid of configurations: y_columns × lag_days × models × fs_k_features.
Edit the grid below to control what gets run.

Usage:
    python run_all_differencing.py
"""

import subprocess
import sys
import itertools
from datetime import datetime

# =============================================================================
# Configuration grid
# =============================================================================

DATA_PATH = "../data/costimier_turnup.parquet"

Y_COLUMNS = [
    #"Steam_power",
    "MBS_SCT_CD",
    "Steam_power_corrected",
    "Steam__kWh/T_",
    "Electrical_power_MW",
    "Electricity__kWh/T_",
    "Starch_uptake_by_paper_Top_Roll__g/m2_",
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_",
    #"MBS_SCT_CD",
    "MBS_SCT_MD",
    "MBS_Burst",
    "MBS_CMT30"
]

LAG_DAYS = [0.25]
EWM_HOURS = [0, 4, 6, 8]

MODELS = [
    "ridge",
    "fs_ridge",
    #"splines_ridge",
    #"hgbr",
    #"realmlp",
    #"gam",
    #"gp",
]

# Only used when model == "fs_ridge"
FS_K_FEATURES = [5, 10, 15, 20, 30, 35]

# =============================================================================
# Runner
# =============================================================================

def build_commands():
    """Generate all (command, description) pairs."""
    commands = []
    for y_col in Y_COLUMNS:
        for lag in LAG_DAYS:
            for ewmh in EWM_HOURS:
                for model in MODELS:
                    if model == "fs_ridge":
                        for k in FS_K_FEATURES:
                            cmd = [
                                sys.executable, "run_differencing_experiment.py",
                                "--y_column", y_col,
                                "--data_path", DATA_PATH,
                                "--model", model,
                                "--lag_days", str(lag),
                                "--ewm_halflife", str(ewmh),
                                "--fs_k_features", str(k),
                            ]
                            desc = f"{y_col} | lag={lag}d | ewm_halflife={ewmh} | {model} | k={k}"
                            commands.append((cmd, desc))
                    else:
                        cmd = [
                            sys.executable, "run_differencing_experiment.py",
                            "--y_column", y_col,
                            "--data_path", DATA_PATH,
                            "--model", model,
                            "--lag_days", str(lag),
                            "--ewm_halflife", str(ewmh),
                        ]
                        desc = f"{y_col} | lag={lag}d | ewm_halflife={ewmh} | {model}"
                        commands.append((cmd, desc))
    return commands


def main():
    commands = build_commands()
    n_total = len(commands)
    print(f"{'='*70}")
    print(f"Batch runner: {n_total} experiments to run")
    print(f"{'='*70}")
    for i, (_, desc) in enumerate(commands):
        print(f"  [{i+1:3d}/{n_total}] {desc}")
    print()

    results = []
    start_all = datetime.now()

    for i, (cmd, desc) in enumerate(commands):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{n_total}] {desc}")
        print(f"{'='*70}")
        t0 = datetime.now()
        proc = subprocess.run(cmd, capture_output=False)
        elapsed = (datetime.now() - t0).total_seconds()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append({"desc": desc, "status": status, "seconds": elapsed})
        print(f"  -> {status} ({elapsed:.1f}s)")

    # Summary
    total_time = (datetime.now() - start_all).total_seconds()
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {total_time:.0f}s total")
    print(f"{'='*70}")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = n_total - n_ok
    print(f"  Succeeded: {n_ok}/{n_total}")
    if n_fail:
        print(f"  FAILED:    {n_fail}")
        for r in results:
            if r["status"] != "OK":
                print(f"    - {r['desc']}: {r['status']}")


if __name__ == "__main__":
    main()
