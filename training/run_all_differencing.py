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

DATA_PATH = "../data/costimiser_turnup.parquet"

Y_COLUMNS = [
    #"Steam_power",
    #"Steam_power_corrected",
    #"Steam__kWh/T_",
    #"Electrical_power_MW",
    #"Electricity__kWh/T_",
    #"Starch_uptake_by_paper_Top_Roll__g/m2_",
    #"Starch_uptake_by_paper_Bottom_Roll__g/m2_",
    "MBS_SCT_CD",
    #"MBS_SCT_MD",
    #"MBS_Burst",
    #"MBS_CMT30"
]

LAG_DAYS = [0.1, 0.25]
EWM_HOURS = [0, 4, 6, 8]
EWM_HOURS = [6]

MODELS = [
    "ridge",
    #"fs_ridge",
    #"splines_ridge",
    #"hgbr",
    #"realmlp",
    #"gam",
    #"gp",
]

# Only used when model == "fs_ridge"
FS_K_FEATURES = [5, 10, 15, 20, 25, 30, 35]

# --- Conditional-independence feature selection ---
# Swept as a grid dimension: each listed value produces a separate run per experiment.
#   [False]        -> only the normal CONTROL_VARS runs
#   [True]         -> only CI-selected runs
#   [False, True]  -> BOTH, so you can compare CI vs non-CI side by side
# When True, predictors are chosen by the partial-correlation search over
# config.CONSIDERED_VARS (config.FIXED_VARIABLES always kept), replacing CONTROL_VARS.
CONDITIONAL_INDEPENDENCE = [True]
CI_MAX_STATES = 5000       # max subsets the CI search visits
CI_MAX_TESTS = 500_000     # max total partial-correlation tests
CI_MAX_K = 3               # max conditioning-set size per test
# Swept grid dimension (CI runs only): one CI run per alpha per experiment.
CI_ALPHA = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# =============================================================================
# Runner
# =============================================================================

def build_commands():
    """Generate all (command, description) pairs.

    CONDITIONAL_INDEPENDENCE is swept as a grid dimension: for each value in it, every
    experiment is emitted once (so [False, True] produces both the normal CONTROL_VARS
    run and the CI-selected run per configuration). --allow_cross_grammage is a
    store_true flag left OFF (same-grammage differencing); append it to opt in.
    """
    def ci_args_tag(use_ci, alpha=None):
        if use_ci:
            return (["--conditional_independence",
                     "--alpha", str(alpha),
                     "--max_states", str(CI_MAX_STATES),
                     "--max_tests", str(CI_MAX_TESTS),
                     "--max_k", str(CI_MAX_K)],
                    f" | CI(alpha={alpha},states={CI_MAX_STATES},k={CI_MAX_K})")
        return [], " | no-CI"

    ci_values = CONDITIONAL_INDEPENDENCE
    if not isinstance(ci_values, (list, tuple)):
        ci_values = [ci_values]
    alphas = CI_ALPHA if isinstance(CI_ALPHA, (list, tuple)) else [CI_ALPHA]
    # (use_ci, alpha) variants: non-CI has no alpha dimension; CI sweeps alphas.
    variants = []
    for u in ci_values:
        if u:
            variants.extend((True, a) for a in alphas)
        else:
            variants.append((False, None))

    commands = []
    for y_col in Y_COLUMNS:
        for lag in LAG_DAYS:
            for ewmh in EWM_HOURS:
                for use_ci, alpha in variants:
                    ci_args, ci_tag = ci_args_tag(use_ci, alpha)

                    if use_ci:
                        # CI IS the feature selection -> no MI top-k sweep, no model
                        # sweep. Emit a single ridge fit per (target, lag, halflife).
                        cmd = [
                            sys.executable, "run_differencing_experiment.py",
                            "--y_column", y_col,
                            "--data_path", DATA_PATH,
                            "--model", "ridge",
                            "--lag_days", str(lag),
                            "--ewm_halflife", str(ewmh),
                        ] + ci_args
                        desc = f"{y_col} | lag={lag}d | ewm_halflife={ewmh} | ridge{ci_tag}"
                        commands.append((cmd, desc))
                        continue

                    # Non-CI: sweep MODELS (and FS_K_FEATURES for fs_ridge) as before.
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
                                desc = f"{y_col} | lag={lag}d | ewm_halflife={ewmh} | {model} | k={k}{ci_tag}"
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
                            desc = f"{y_col} | lag={lag}d | ewm_halflife={ewmh} | {model}{ci_tag}"
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
