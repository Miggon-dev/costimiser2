"""
Target-centred conditional-dependence discovery for the SCT-CD output.

This module is a faithful, importable port of the exploratory notebook
`causal_discovery.ipynb`. Running it as a script reproduces the notebook's
analysis (identical robust neighbours and skeleton) and writes the same
interactive hierarchical dependence graph.
"""

import sys
from pathlib import Path

# Make the parent directory (where utility.py lives) importable, and keep
# this directory importable for the local config / data_cleaning modules.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from itertools import combinations
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr

from utility import _feature_engineering, setpoint_df, FeatureCreator
from data_cleaning import unique_in_order


# ============================================================
# Configuration
# ============================================================

Y_COLUMN = "MBS_SCT_CD"
OUTPUT_VAR = "MBS_SCT_CD"
MAX_COND_DIM = 3
MIN_ABS_VAL = 0.05
ALPHA = 0.05
OUTPUT_DIR = Path("conditional_dependence_results")
DATA_PATH = "../data/costimiser_turnup.parquet"

# ============================================================
# Conditional-dependence skeleton (notebook cell)
# ============================================================

class ConditionalDependenceSkeleton:
    def __init__(
        self,
        dataframe,
        *,
        output_var: str = OUTPUT_VAR,
        max_cond_dim: int = MAX_COND_DIM,
        min_abs_val: float = MIN_ABS_VAL,
        alpha: float = ALPHA,
        verbosity: int = 1,
    ):
        self.dataframe = dataframe
        self.output_var = output_var
        self.max_cond_dim = int(max_cond_dim)
        self.min_abs_val = float(min_abs_val)
        self.alpha = float(alpha)
        self.verbosity = int(verbosity)

        self.var_names = list(dataframe.var_names)
        self.name_to_idx = {
            name: i for i, name in enumerate(self.var_names)
        }

        if self.output_var not in self.name_to_idx:
            raise ValueError(
                f"Output variable {self.output_var!r} "
                "is not in dataframe.var_names"
            )

        self.triple_parcorr = ParCorr(significance="analytic")
        self.triple_parcorr.set_dataframe(self.dataframe)

        self.graph_: Optional[nx.Graph] = None
        self.separation_df_: Optional[pd.DataFrame] = None
        self.edge_diagnostics_df_: Optional[pd.DataFrame] = None

        self.target_screening_df_: Optional[pd.DataFrame] = None
        self.separator_motifs_df_: Optional[pd.DataFrame] = None
        self.selected_variables_: Optional[list[str]] = None
        self.separating_variables_: Optional[list[str]] = None

    # ============================================================
    # Basic CI utilities
    # ============================================================

    def parcorr_test(
        self,
        x_name: str,
        y_name: str,
        z_names: Optional[Iterable[str]] = None,
    ) -> dict:
        if z_names is None:
            z_names = []

        z_names = list(z_names)

        x = self.name_to_idx[x_name]
        y = self.name_to_idx[y_name]
        z = [self.name_to_idx[name] for name in z_names]

        val, pval = self.triple_parcorr.run_test(
            X=[(x, 0)],
            Y=[(y, 0)],
            Z=[(zz, 0) for zz in z],
        )

        return {
            "val": float(val),
            "p": float(pval),
        }

    def classify_test(self, result: dict) -> str:
        val = result["val"]
        p = result["p"]

        if p < self.alpha:
            if abs(val) >= self.min_abs_val:
                return "DEP"
            return "WEAK"

        return "INDEP"

    def practically_independent(self, result: dict) -> bool:
        return abs(result["val"]) < self.min_abs_val

    @staticmethod
    def _complete_graph(variable_names: list[str]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(variable_names)
        G.add_edges_from(combinations(variable_names, 2))
        return G

    # ============================================================
    # PHASE 1
    # Target screening
    # ============================================================

    def _target_screening(
        self,
        variable_names: list[str],
    ):
        target = self.output_var

        all_candidates = [
            v
            for v in variable_names
            if v != target
        ]

        diagnostics = []
        separation_records = []
        motif_records = []

        if self.verbosity:
            print("\n" + "=" * 70)
            print("PHASE 1: TARGET SCREENING")
            print("=" * 70)
            print(f"Target: {target}")
            print(f"Candidate variables: {len(all_candidates):,}")

        # --------------------------------------------------------
        # k = 0
        # --------------------------------------------------------

        if self.verbosity:
            print("\nTarget screening: conditioning dimension 0...")

        marginal_survivors = []

        for x_name in all_candidates:
            result = self.parcorr_test(
                target,
                x_name,
                [],
            )

            abs_val = abs(result["val"])

            if self.practically_independent(result):
                separation_records.append({
                    "X": x_name,
                    "Y": target,
                    "conditioning_set": tuple(),
                    "cond_dim": 0,
                    "val": result["val"],
                    "abs_val": abs_val,
                    "p": result["p"],
                    "status": self.classify_test(result),
                })

                diagnostics.append({
                    "X": x_name,
                    "Y": target,
                    "marginal_val": result["val"],
                    "marginal_abs_val": abs_val,
                    "marginal_p": result["p"],
                    "best_separator": None,
                    "conditional_val": None,
                    "conditional_abs_val": None,
                    "conditional_p": None,
                    "retained": False,
                })

            else:
                marginal_survivors.append(x_name)

                diagnostics.append({
                    "X": x_name,
                    "Y": target,
                    "marginal_val": result["val"],
                    "marginal_abs_val": abs_val,
                    "marginal_p": result["p"],
                    "best_separator": None,
                    "conditional_val": None,
                    "conditional_abs_val": None,
                    "conditional_p": None,
                    "retained": True,
                })

        if self.verbosity:
            print(
                f"Marginally associated with {target}: "
                f"{len(marginal_survivors):,}"
            )

        # --------------------------------------------------------
        # k = 1
        # Search ALL Z and keep the best separator for each X
        # --------------------------------------------------------

        robust_neighbours = []
        separating_variables = set()

        if self.max_cond_dim >= 1:

            if self.verbosity:
                print(
                    "\nTarget screening: conditioning dimension 1..."
                )
                print(
                    "Testing all single-variable conditioners "
                    "from the full variable pool."
                )

            diag_lookup = {
                row["X"]: row
                for row in diagnostics
            }

            total_tested = 0

            for i, x_name in enumerate(
                marginal_survivors,
                start=1,
            ):

                candidate_conditioners = [
                    z
                    for z in all_candidates
                    if z != x_name
                ]

                best_separator = None
                best_result = None
                best_abs_val = float("inf")

                for z_name in candidate_conditioners:

                    result = self.parcorr_test(
                        target,
                        x_name,
                        [z_name],
                    )

                    total_tested += 1
                    abs_val = abs(result["val"])

                    if abs_val < best_abs_val:
                        best_abs_val = abs_val
                        best_separator = z_name
                        best_result = result

                diag = diag_lookup[x_name]

                if best_result is not None:
                    diag["best_separator"] = best_separator
                    diag["conditional_val"] = best_result["val"]
                    diag["conditional_abs_val"] = best_abs_val
                    diag["conditional_p"] = best_result["p"]

                if (
                    best_result is not None
                    and best_abs_val < self.min_abs_val
                ):
                    separating_variables.add(best_separator)

                    separation_records.append({
                        "X": x_name,
                        "Y": target,
                        "conditioning_set": (best_separator,),
                        "cond_dim": 1,
                        "val": best_result["val"],
                        "abs_val": best_abs_val,
                        "p": best_result["p"],
                        "status": self.classify_test(best_result),
                    })

                    motif_records.append({
                        "associated_variable": x_name,
                        "separator": best_separator,
                        "target": target,
                        "marginal_val": diag["marginal_val"],
                        "marginal_abs_val": diag["marginal_abs_val"],
                        "marginal_p": diag["marginal_p"],
                        "conditional_val": best_result["val"],
                        "conditional_abs_val": best_abs_val,
                        "conditional_p": best_result["p"],
                        "attenuation": (
                            1.0
                            - best_abs_val
                            / diag["marginal_abs_val"]
                            if diag["marginal_abs_val"] > 0
                            else None
                        ),
                    })

                    diag["retained"] = False

                else:
                    robust_neighbours.append(x_name)
                    diag["retained"] = True

                if self.verbosity >= 2 and i % 25 == 0:
                    print(
                        f"Processed {i:,}/"
                        f"{len(marginal_survivors):,} candidates..."
                    )

            if self.verbosity:
                print(f"Tested {total_tested:,} CI relations")
                print(
                    f"Robust target neighbours: "
                    f"{len(robust_neighbours):,}"
                )
                print(
                    f"Distinct best separators: "
                    f"{len(separating_variables):,}"
                )

        else:
            robust_neighbours = list(marginal_survivors)

        self.target_screening_df_ = pd.DataFrame(diagnostics)
        self.separator_motifs_df_ = pd.DataFrame(motif_records)
        self.separating_variables_ = sorted(separating_variables)

        return (
            robust_neighbours,
            separation_records,
        )

    # ============================================================
    # PHASE 2
    # Deeper skeleton only among robust target neighbours
    # ============================================================

    def _fit_robust_skeleton(
        self,
        variable_names: list[str],
    ):
        G = self._complete_graph(variable_names)

        separation_records = []
        best_by_edge = {}

        for k in range(self.max_cond_dim + 1):

            if self.verbosity:
                print(
                    f"\nRobust skeleton: conditioning sets "
                    f"of size {k}..."
                )

            adjacency_snapshot = {
                node: set(G.neighbors(node))
                for node in G.nodes
            }

            edges_snapshot = list(G.edges())

            tested_this_round = 0
            removed_this_round = 0

            for x_name, y_name in edges_snapshot:

                if not G.has_edge(x_name, y_name):
                    continue

                if k == 0:
                    cond_sets = [tuple()]

                else:
                    neigh_x = (
                        adjacency_snapshot[x_name]
                        - {y_name}
                    )

                    neigh_y = (
                        adjacency_snapshot[y_name]
                        - {x_name}
                    )

                    candidates = sorted(
                        neigh_x | neigh_y
                    )

                    if len(candidates) < k:
                        continue

                    cond_sets = combinations(
                        candidates,
                        k,
                    )

                for cond_set in cond_sets:

                    tested_this_round += 1

                    result = self.parcorr_test(
                        x_name,
                        y_name,
                        cond_set,
                    )

                    abs_val = abs(result["val"])

                    edge_key = tuple(
                        sorted((x_name, y_name))
                    )

                    current_best = best_by_edge.get(
                        edge_key
                    )

                    if (
                        current_best is None
                        or abs_val < current_best["abs_val"]
                    ):
                        best_by_edge[edge_key] = {
                            "X": edge_key[0],
                            "Y": edge_key[1],
                            "best_conditioning_set":
                                tuple(cond_set),
                            "best_cond_dim": k,
                            "val": result["val"],
                            "abs_val": abs_val,
                            "p": result["p"],
                            "status":
                                self.classify_test(result),
                        }

                    if self.practically_independent(result):

                        G.remove_edge(
                            x_name,
                            y_name,
                        )

                        separation_records.append({
                            "X": edge_key[0],
                            "Y": edge_key[1],
                            "conditioning_set":
                                tuple(cond_set),
                            "cond_dim": k,
                            "val": result["val"],
                            "abs_val": abs_val,
                            "p": result["p"],
                            "status":
                                self.classify_test(result),
                        })

                        removed_this_round += 1
                        break

            if self.verbosity:
                print(
                    f"Tested {tested_this_round:,} CI relations"
                )
                print(
                    f"Removed {removed_this_round:,} edges; "
                    f"{G.number_of_edges():,} remain"
                )

            next_k = k + 1

            if next_k <= self.max_cond_dim:

                can_continue = False

                for x_name, y_name in G.edges():

                    candidates = (
                        (
                            set(G.neighbors(x_name))
                            - {y_name}
                        )
                        |
                        (
                            set(G.neighbors(y_name))
                            - {x_name}
                        )
                    )

                    if len(candidates) >= next_k:
                        can_continue = True
                        break

                if not can_continue:

                    if self.verbosity:
                        print(
                            "Stopping early: no remaining edge "
                            "has enough candidate conditioners "
                            f"for dimension {next_k}."
                        )

                    break

        return (
            G,
            separation_records,
            best_by_edge,
        )

    # ============================================================
    # Main fit
    # ============================================================

    def fit(
        self,
        variable_names: Optional[Iterable[str]] = None,
    ) -> nx.Graph:

        if variable_names is None:
            variable_names = list(self.var_names)
        else:
            variable_names = list(variable_names)

        missing = [
            v
            for v in variable_names
            if v not in self.name_to_idx
        ]

        if missing:
            raise ValueError(
                f"Variables not found in dataframe: {missing}"
            )

        if self.output_var not in variable_names:
            variable_names = [
                self.output_var,
                *variable_names,
            ]

        if self.verbosity:
            print(
                f"Variables entering screening: "
                f"{len(variable_names):,}"
            )

        # ========================================================
        # Phase 1
        # ========================================================

        (
            robust_neighbours,
            target_separation_records,
        ) = self._target_screening(
            variable_names
        )

        # ========================================================
        # Phase 2 variable set
        # ========================================================

        selected_variables = [
            self.output_var,
            *robust_neighbours,
        ]

        self.selected_variables_ = selected_variables

        if self.verbosity:
            print("\n" + "=" * 70)
            print("PHASE 2: ROBUST TARGET SKELETON")
            print("=" * 70)
            print(
                f"Variables entering deeper skeleton: "
                f"{len(selected_variables):,}"
            )

        # ========================================================
        # Phase 2
        # ========================================================

        (
            G,
            reduced_separation_records,
            best_by_edge,
        ) = self._fit_robust_skeleton(
            selected_variables
        )

        self.graph_ = G

        self.separation_df_ = pd.DataFrame(
            target_separation_records
            + reduced_separation_records,
            columns=[
                "X",
                "Y",
                "conditioning_set",
                "cond_dim",
                "val",
                "abs_val",
                "p",
                "status",
            ],
        )

        self.edge_diagnostics_df_ = pd.DataFrame(
            list(best_by_edge.values()),
            columns=[
                "X",
                "Y",
                "best_conditioning_set",
                "best_cond_dim",
                "val",
                "abs_val",
                "p",
                "status",
            ],
        )

        if not self.edge_diagnostics_df_.empty:

            retained = {
                tuple(sorted(edge))
                for edge in G.edges()
            }

            self.edge_diagnostics_df_["retained"] = [
                (row.X, row.Y) in retained
                for row
                in self.edge_diagnostics_df_.itertuples()
            ]

        return G

    # ============================================================
    # Helpers
    # ============================================================

    def neighbours(self, variable: str) -> list[str]:

        if self.graph_ is None:
            raise RuntimeError("Call fit() first.")

        return sorted(
            self.graph_.neighbors(variable)
        )

    def separating_record(
        self,
        x_name: str,
        y_name: str,
    ) -> pd.DataFrame:

        if self.separation_df_ is None:
            raise RuntimeError("Call fit() first.")

        a, b = sorted((x_name, y_name))

        temp = self.separation_df_.copy()

        mask = temp.apply(
            lambda row:
                tuple(sorted((row["X"], row["Y"])))
                == (a, b),
            axis=1,
        )

        return temp.loc[mask].copy()

    def local_subgraph(
        self,
        center: Optional[str] = None,
        depth: int = 2,
    ) -> nx.Graph:

        if self.graph_ is None:
            raise RuntimeError("Call fit() first.")

        if center is None:
            center = self.output_var

        nodes = {center}
        frontier = {center}

        for _ in range(depth):

            new_frontier = set()

            for node in frontier:
                new_frontier.update(
                    self.graph_.neighbors(node)
                )

            new_frontier -= nodes

            if not new_frontier:
                break

            nodes.update(new_frontier)
            frontier = new_frontier

        return self.graph_.subgraph(nodes).copy()

    def plot_edge_table(
        self,
        graph: Optional[nx.Graph] = None,
    ) -> pd.DataFrame:

        if graph is None:

            if self.graph_ is None:
                raise RuntimeError("Call fit() first.")

            graph = self.graph_

        rows = []

        for a, b in graph.edges():

            if a == self.output_var:

                rows.append({
                    "source": b,
                    "target": a,
                    "type": "to_output",
                })

            elif b == self.output_var:

                rows.append({
                    "source": a,
                    "target": b,
                    "type": "to_output",
                })

            else:

                aa, bb = sorted((a, b))

                rows.append({
                    "source": aa,
                    "target": bb,
                    "type": "undirected",
                })

        return pd.DataFrame(rows)

    def save_results(
        self,
        output_dir: Path | str = OUTPUT_DIR,
        local_depth: int = 2,
    ) -> None:

        if self.graph_ is None:
            raise RuntimeError("Call fit() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        sep = self.separation_df_.copy()

        if not sep.empty:

            sep["conditioning_set"] = (
                sep["conditioning_set"].apply(
                    lambda x: " | ".join(x)
                )
            )

            sep.to_csv(
                output_dir / "separation_sets.csv",
                index=False,
            )

        diag = self.edge_diagnostics_df_.copy()

        if not diag.empty:

            diag["best_conditioning_set"] = (
                diag["best_conditioning_set"].apply(
                    lambda x: " | ".join(x)
                )
            )

            diag.to_csv(
                output_dir / "edge_diagnostics.csv",
                index=False,
            )

        if (
            self.target_screening_df_ is not None
            and not self.target_screening_df_.empty
        ):
            self.target_screening_df_.to_csv(
                output_dir / "target_screening.csv",
                index=False,
            )

        if (
            self.separator_motifs_df_ is not None
            and not self.separator_motifs_df_.empty
        ):
            self.separator_motifs_df_.to_csv(
                output_dir / "separator_motifs.csv",
                index=False,
            )

        if self.selected_variables_ is not None:

            pd.DataFrame({
                "variable": self.selected_variables_
            }).to_csv(
                output_dir / "selected_variables.csv",
                index=False,
            )

        if self.separating_variables_ is not None:

            pd.DataFrame({
                "variable": self.separating_variables_
            }).to_csv(
                output_dir / "separating_variables.csv",
                index=False,
            )

        nx.write_graphml(
            self.graph_,
            output_dir / "dependence_skeleton.graphml",
        )

        local_graph = self.local_subgraph(
            self.output_var,
            depth=local_depth,
        )

        nx.write_graphml(
            local_graph,
            output_dir
            / f"{self.output_var}_local.graphml",
        )

        self.plot_edge_table(
            local_graph
        ).to_csv(
            output_dir
            / f"{self.output_var}_local_edges.csv",
            index=False,
        )

# ============================================================
# Analysis wrapper (notebook cell)
# ============================================================

def run_dependence_analysis(
    dataframe,
    *,
    output_var: str = OUTPUT_VAR,
    variable_names: Optional[Iterable[str]] = None,
    max_cond_dim: int = MAX_COND_DIM,
    min_abs_val: float = MIN_ABS_VAL,
    alpha: float = ALPHA,
    output_dir: Optional[Path | str] = OUTPUT_DIR,
    verbosity: int = 1,
    max_target_neighbours: int = 40,
):
    analyser = ConditionalDependenceSkeleton(
        dataframe=dataframe,
        output_var=output_var,
        max_cond_dim=max_cond_dim,
        min_abs_val=min_abs_val,
        alpha=alpha,
        verbosity=verbosity,
    )

    graph = analyser.fit(variable_names=variable_names)
    output_neighbours = analyser.neighbours(output_var)

    if verbosity:
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        print(f"\nRemaining direct neighbours of {output_var}:")
        for name in output_neighbours:
            print(f"  - {name}")

    if output_dir is not None:
        analyser.save_results(output_dir)
        if verbosity:
            print(f"\nResults written to: {Path(output_dir).resolve()}")

    local_graph = analyser.local_subgraph(output_var, depth=2)

    return {
        "analyser": analyser,
        "graph": graph,
        "separation_df": analyser.separation_df_,
        "edge_diagnostics_df": analyser.edge_diagnostics_df_,
        "output_neighbours": output_neighbours,
        "local_graph": local_graph,
        "local_edges": analyser.plot_edge_table(local_graph),
    }

# ============================================================
# Segment-wise EWM filter (notebook cell)
# ============================================================

def ewm_filter_by_segment(
    df,
    columns,
    halflife="120min",
    max_gap="12h",
):
    df = df.copy().sort_index()

    # New segment at grade change or >12 h gap
    grade_change = (
        df["AB_Grade_ID"]
        .ne(df["AB_Grade_ID"].shift().fillna(True))
    )

    time_gap = (
        df.index.to_series()
        .diff()
        .gt(pd.Timedelta(max_gap)
        ).fillna(False)
    )

    segment_start = (grade_change | time_gap).fillna(True)

    segment = pd.Series(
        segment_start.to_numpy(dtype=bool),
        index=df.index,
    ).cumsum()

    result = df.copy()

    for _, idx in df.groupby(segment).groups.items():

        block = df.loc[idx, columns]

        result.loc[idx, columns] = (
            block
            .ewm(
                halflife=halflife,
                times=block.index,
                adjust=True,
            )
            .mean()
        )

    return result

# ============================================================
# Interactive graph (notebook cell)
# ============================================================

def plot_interactive_dependency_graph(
    results,
    output_var="MBS_SCT_CD",
    output_label="SCT CD",
    output_file="interactive_dependence_graph.html",
    height="850px",
    show_blocked_relations=True,
):
    """
    Create an interactive dependence graph.

    Rules
    -----
    1. Output is always visible.
    2. Robust neighbours are always visible.
    3. A separator is initially visible only if it is also a robust neighbour.
    4. Pure associated/blocked variables start hidden.
    5. Clicking a visible separator expands/collapses its associated variables.
    6. If show_blocked_relations=True, revealed blocked variables also get
       dashed X -> output relations.
    7. No pure separator is shown as an orphan in the collapsed graph.

    Parameters
    ----------
    results : dict
        Result dictionary from run_dependence_analysis().

    output_var : str
        Name of output variable.

    output_label : str
        Display label for output node.

    output_file : str or Path
        HTML output path.

    height : str
        HTML graph height.

    show_blocked_relations : bool
        If True, revealed blocked variables also show dashed X -> output links.

    Returns
    -------
    Path
        Path to generated HTML.
    """

    # ============================================================
    # Retrieve motif dataframe
    # ============================================================

    if "separator_motifs_df" in results:
        motifs = results["separator_motifs_df"].copy()

    else:
        analyser = results.get("analyser")

        if analyser is None:
            raise KeyError(
                "Could not find 'separator_motifs_df' or 'analyser' in results."
            )

        if not hasattr(analyser, "separator_motifs_df_"):
            raise AttributeError(
                "The analyser does not contain 'separator_motifs_df_'."
            )

        motifs = analyser.separator_motifs_df_.copy()

    # ============================================================
    # Robust neighbours
    # ============================================================

    if "output_neighbours" not in results:
        raise KeyError(
            "results does not contain 'output_neighbours'."
        )

    robust = [
        str(x)
        for x in results["output_neighbours"]
    ]

    output_var = str(output_var)

    # ============================================================
    # Validate motifs
    # ============================================================

    required_columns = {
        "associated_variable",
        "separator",
        "marginal_abs_val",
        "conditional_abs_val",
        "attenuation",
    }

    missing = (
        required_columns
        - set(motifs.columns)
    )

    if missing:
        raise ValueError(
            "separator_motifs_df is missing columns: "
            f"{sorted(missing)}"
        )

    motifs = (
        motifs
        .dropna(
            subset=[
                "associated_variable",
                "separator",
            ]
        )
        .copy()
    )

    motifs["associated_variable"] = (
        motifs["associated_variable"]
        .astype(str)
    )

    motifs["separator"] = (
        motifs["separator"]
        .astype(str)
    )

    # ============================================================
    # Roles
    # ============================================================

    separators = sorted(
        motifs["separator"]
        .drop_duplicates()
        .tolist()
    )

    associated = sorted(
        motifs["associated_variable"]
        .drop_duplicates()
        .tolist()
    )

    robust_set = set(robust)
    separator_set = set(separators)
    associated_set = set(associated)

    # ============================================================
    # Only initially meaningful nodes are included visibly
    #
    # Pure separators and pure associated vars still exist in the
    # dataset, but start hidden.
    # ============================================================

    all_names = (
        [output_var]
        + robust
        + separators
        + associated
    )

    all_names = list(
        dict.fromkeys(all_names)
    )

    node_id = {
        name: f"n{i}"
        for i, name in enumerate(all_names)
    }

    # ============================================================
    # Nodes
    # ============================================================

    nodes = []

    for name in all_names:

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------

        if name == output_var:

            nodes.append({
                "id": node_id[name],
                "label": output_label,
                "title": (
                    "<b>Output</b><br>"
                    + output_var
                ),
                "group": "output",
                "level": 3,
                "shape": "ellipse",
                "size": 32,
                "hidden": False,
            })

            continue

        # --------------------------------------------------------
        # Robust neighbour
        # --------------------------------------------------------

        if name in robust_set:

            title = (
                "<b>Robust SCT neighbour</b><br>"
                + name
            )

            if name in separator_set:

                n_children = int(
                    motifs["separator"]
                    .eq(name)
                    .sum()
                )

                title += (
                    "<br><br>"
                    "<b>Also acts as separator</b><br>"
                    f"{n_children} blocked variable"
                    f"{'' if n_children == 1 else 's'}"
                    "<br><br>"
                    "<b>Click to expand/collapse</b>"
                )

            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": title,
                "group": "robust",
                "level": 2,
                "shape": "dot",
                "size": 24,
                "hidden": False,
            })

            continue

        # --------------------------------------------------------
        # Pure separator
        #
        # Important:
        # hidden initially, otherwise it becomes an orphan.
        # --------------------------------------------------------

        if name in separator_set:

            n_children = int(
                motifs["separator"]
                .eq(name)
                .sum()
            )

            title = (
                "<b>Separating variable</b><br>"
                + name
                + "<br><br>"
                + f"{n_children} blocked variable"
                + (
                    ""
                    if n_children == 1
                    else "s"
                )
            )

            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": title,
                "group": "separator",
                "level": 1,
                "shape": "dot",
                "size": 18,
                "hidden": True,
            })

            continue

        # --------------------------------------------------------
        # Pure associated / blocked variable
        # --------------------------------------------------------

        if name in associated_set:

            rows = motifs[
                motifs["associated_variable"]
                == name
            ]

            row = rows.iloc[0]

            marginal_abs_val = row["marginal_abs_val"]
            conditional_abs_val = row["conditional_abs_val"]
            attenuation = row["attenuation"]
            separator_name = str(row["separator"])

            marginal_txt = (
                f"{marginal_abs_val:.3f}"
                if pd.notna(marginal_abs_val)
                else "NA"
            )

            conditional_txt = (
                f"{conditional_abs_val:.3f}"
                if pd.notna(conditional_abs_val)
                else "NA"
            )

            attenuation_txt = (
                f"{100 * attenuation:.1f}%"
                if pd.notna(attenuation)
                else "NA"
            )

            title = (
                "Blocked process variable\n"
                + name
                + "\n\n"
                + "Blocking conditioning set:\n"
                + separator_name
                + "\n\n"
                + f"|r marginal| = {marginal_txt}\n"
                + f"|r conditioned| = {conditional_txt}\n"
                + f"Attenuation = {attenuation_txt}"
            )

            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": title,
                "group": "associated",
                "level": 0,
                "shape": "dot",
                "size": 11,
                "hidden": True,
            })

    # ============================================================
    # Duplicate-ID guard
    # ============================================================

    node_ids = [
        n["id"]
        for n in nodes
    ]

    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError(
            "Duplicate node IDs detected."
        )

    # ============================================================
    # Edges
    # ============================================================

    edges = []
    edge_counter = 0

    # ------------------------------------------------------------
    # Robust -> output
    # ------------------------------------------------------------

    for name in robust:

        if name == output_var:
            continue

        if name not in node_id:
            continue

        edge_counter += 1

        edges.append({
            "id": f"e{edge_counter}",
            "from": node_id[name],
            "to": node_id[output_var],
            "arrows": "to",
            "width": 2.5,
            "dashes": False,
            "type": "robust",
            "hidden": False,
        })

    # ------------------------------------------------------------
    # Associated -> separator
    #
    # Always available, initially hidden.
    # ------------------------------------------------------------

    for _, row in motifs.iterrows():

        x = str(row["associated_variable"])
        z = str(row["separator"])

        if (
            x not in node_id
            or z not in node_id
        ):
            continue

        edge_counter += 1

        edges.append({
            "id": f"e{edge_counter}",
            "from": node_id[x],
            "to": node_id[z],
            "width": 1.5,
            "dashes": False,
            "type": "motif",
            "hidden": True,
        })

    # ------------------------------------------------------------
    # Optional dashed X -> output
    #
    # Initially hidden. They appear only when X is revealed.
    # ------------------------------------------------------------

    if show_blocked_relations:

        added = set()

        for _, row in motifs.iterrows():

            x = str(row["associated_variable"])

            if x == output_var:
                continue

            if x in robust_set:
                continue

            if x in added:
                continue

            if x not in node_id:
                continue

            added.add(x)

            edge_counter += 1

            edges.append({
                "id": f"e{edge_counter}",
                "from": node_id[x],
                "to": node_id[output_var],
                "arrows": "to",
                "width": 1.1,
                "dashes": True,
                "type": "blocked",
                "hidden": True,
            })

    # ============================================================
    # Separator -> children mapping
    # ============================================================

    children_by_separator = {}

    for separator, group in motifs.groupby(
        "separator",
        sort=False,
    ):

        separator = str(separator)

        if separator not in node_id:
            continue

        children = [
            node_id[str(x)]
            for x in (
                group["associated_variable"]
                .drop_duplicates()
                .tolist()
            )
            if str(x) in node_id
        ]

        children_by_separator[
            node_id[separator]
        ] = children

    # ============================================================
    # Reverse mapping:
    # child -> separators
    # ============================================================

    separators_by_child = {}

    for separator_id, children in children_by_separator.items():

        for child_id in children:

            separators_by_child.setdefault(
                child_id,
                []
            ).append(
                separator_id
            )

    # ============================================================
    # Pure associated nodes
    # ============================================================

    collapsible_nodes = {
        node_id[name]
        for name in associated
        if (
            name not in robust_set
            and name not in separator_set
            and name != output_var
        )
    }

    # ============================================================
    # Pure separators
    # ============================================================

    pure_separator_nodes = {
        node_id[name]
        for name in separators
        if (
            name not in robust_set
            and name != output_var
        )
    }

    # ============================================================
    # JSON
    # ============================================================

    nodes_json = json.dumps(
        nodes,
        ensure_ascii=False,
    )

    edges_json = json.dumps(
        edges,
        ensure_ascii=False,
    )

    children_json = json.dumps(
        children_by_separator,
        ensure_ascii=False,
    )

    separators_by_child_json = json.dumps(
        separators_by_child,
        ensure_ascii=False,
    )

    collapsible_json = json.dumps(
        list(collapsible_nodes),
        ensure_ascii=False,
    )

    pure_separator_json = json.dumps(
        list(pure_separator_nodes),
        ensure_ascii=False,
    )

    show_blocked_json = json.dumps(
        bool(show_blocked_relations)
    )

    # ============================================================
    # HTML
    # ============================================================

    html = """
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<title>Interactive dependence graph</title>

<script
    type="text/javascript"
    src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js">
</script>

<style>

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: white;
}

#controls {
    padding: 10px 14px;
    border-bottom: 1px solid #ddd;
    background: #fafafa;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

#network {
    width: 100%;
    height: __HEIGHT__;
}

button {
    padding: 7px 12px;
    border: 1px solid #aaa;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 13px;
}

button:hover {
    background: #f0f0f0;
}

.help {
    margin-left: 10px;
    font-size: 13px;
    color: #555;
}

#status {
    margin-left: auto;
    font-size: 12px;
    color: #666;
}

</style>

</head>


<body>


<div id="controls">

    <button id="collapseAll">
        Collapse all
    </button>

    <button id="expandAll">
        Expand all
    </button>

    <button id="fitGraph">
        Fit
    </button>

    <span
        id="helpText"
        class="help">
    </span>

    <span id="status"></span>

</div>


<div id="network"></div>


<script>


const nodes = new vis.DataSet(
    __NODES_JSON__
);


const edges = new vis.DataSet(
    __EDGES_JSON__
);


const childrenBySeparator =
    __CHILDREN_JSON__;


const separatorsByChild =
    __SEPARATORS_BY_CHILD_JSON__;


const collapsibleNodes =
    new Set(
        __COLLAPSIBLE_NODES_JSON__
    );


const pureSeparatorNodes =
    new Set(
        __PURE_SEPARATOR_NODES_JSON__
    );


const showBlockedRelations =
    __SHOW_BLOCKED_RELATIONS__;


const expandedSeparators =
    new Set();


const container =
    document.getElementById(
        "network"
    );


const statusElement =
    document.getElementById(
        "status"
    );


const helpElement =
    document.getElementById(
        "helpText"
    );


const data = {
    nodes: nodes,
    edges: edges
};


const options = {

    layout: {

        hierarchical: {

            enabled: true,

            direction: "LR",

            sortMethod: "directed",

            levelSeparation: 330,

            nodeSpacing: 42,

            treeSpacing: 180,

            blockShifting: true,

            edgeMinimization: true,

            parentCentralization: true
        }
    },

    physics: {
        enabled: false
    },

    interaction: {

        hover: true,

        navigationButtons: true,

        keyboard: true,

        tooltipDelay: 150,

        zoomView: true,

        dragView: true
    },

    nodes: {

        borderWidth: 1,

        font: {
            size: 13,
            face: "Arial"
        }
    },

    groups: {

        output: {
            shape: "ellipse",
            size: 32
        },

        robust: {
            shape: "dot",
            size: 24
        },

        separator: {
            shape: "dot",
            size: 18
        },

        associated: {
            shape: "dot",
            size: 11
        }
    },

    edges: {

        smooth: {

            enabled: true,

            type: "cubicBezier",

            forceDirection: "horizontal",

            roundness: 0.25
        }
    }
};


const network =
    new vis.Network(
        container,
        data,
        options
    );


// ============================================================
// Help
// ============================================================

if (showBlockedRelations) {

    helpElement.textContent =
        "Click visible separator nodes to expand/collapse their parents. "
        + "Solid = robust relation. "
        + "Dashed = relation removed by conditioning.";

} else {

    helpElement.textContent =
        "Robust network with expandable blocking structures. "
        + "Dashed blocked relations are hidden.";
}


// ============================================================
// Child still needed by another expanded separator?
// ============================================================

function childStillNeeded(childId) {

    const parentSeparators =
        separatorsByChild[
            childId
        ] || [];

    for (
        const separatorId
        of parentSeparators
    ) {

        if (
            expandedSeparators.has(
                separatorId
            )
        ) {
            return true;
        }
    }

    return false;
}


// ============================================================
// Show/hide dashed X -> output edge
// ============================================================

function updateBlockedEdge(childId) {

    if (!showBlockedRelations) {
        return;
    }

    const node =
        nodes.get(
            childId
        );

    if (!node) {
        return;
    }

    const blockedEdges =
        edges.get({

            filter: function(edge) {

                return (
                    edge.type === "blocked"
                    &&
                    edge.from === childId
                );
            }
        });

    blockedEdges.forEach(
        edge => {

            edges.update({
                id: edge.id,
                hidden: Boolean(node.hidden)
            });
        }
    );
}


// ============================================================
// Reveal pure separator when required
// ============================================================

function revealSeparator(
    separatorId
) {

    if (
        pureSeparatorNodes.has(
            separatorId
        )
    ) {

        nodes.update({
            id: separatorId,
            hidden: false
        });
    }
}


// ============================================================
// Hide pure separator if it would become orphaned
// ============================================================

function hideSeparatorIfUnused(
    separatorId
) {

    if (
        !pureSeparatorNodes.has(
            separatorId
        )
    ) {
        return;
    }

    if (
        expandedSeparators.has(
            separatorId
        )
    ) {
        return;
    }

    nodes.update({
        id: separatorId,
        hidden: true
    });
}


// ============================================================
// Expand separator
// ============================================================

function expand(
    separatorId
) {

    const children =
        childrenBySeparator[
            separatorId
        ];

    if (!children) {
        return;
    }

    revealSeparator(
        separatorId
    );

    expandedSeparators.add(
        separatorId
    );

    children.forEach(
        childId => {

            if (
                collapsibleNodes.has(
                    childId
                )
            ) {

                nodes.update({
                    id: childId,
                    hidden: false
                });
            }

            const motifEdges =
                edges.get({

                    filter: function(edge) {

                        return (
                            edge.type === "motif"
                            &&
                            edge.from === childId
                            &&
                            edge.to === separatorId
                        );
                    }
                });

            motifEdges.forEach(
                edge => {

                    edges.update({
                        id: edge.id,
                        hidden: false
                    });
                }
            );

            updateBlockedEdge(
                childId
            );
        }
    );

    updateStatus();
}


// ============================================================
// Collapse separator
// ============================================================

function collapse(
    separatorId
) {

    const children =
        childrenBySeparator[
            separatorId
        ];

    if (!children) {
        return;
    }

    expandedSeparators.delete(
        separatorId
    );

    children.forEach(
        childId => {

            const motifEdges =
                edges.get({

                    filter: function(edge) {

                        return (
                            edge.type === "motif"
                            &&
                            edge.from === childId
                            &&
                            edge.to === separatorId
                        );
                    }
                });

            motifEdges.forEach(
                edge => {

                    edges.update({
                        id: edge.id,
                        hidden: true
                    });
                }
            );

            if (
                collapsibleNodes.has(
                    childId
                )
                &&
                !childStillNeeded(
                    childId
                )
            ) {

                nodes.update({
                    id: childId,
                    hidden: true
                });
            }

            updateBlockedEdge(
                childId
            );
        }
    );

    hideSeparatorIfUnused(
        separatorId
    );

    updateStatus();
}


// ============================================================
// Toggle
// ============================================================

function toggleSeparator(
    separatorId
) {

    if (
        expandedSeparators.has(
            separatorId
        )
    ) {

        collapse(
            separatorId
        );

    } else {

        expand(
            separatorId
        );
    }
}


// ============================================================
// Status
// ============================================================

function updateStatus() {

    statusElement.textContent =
        "Expanded separators: "
        + expandedSeparators.size;
}


// ============================================================
// Click
// ============================================================

network.on(

    "click",

    function(params) {

        if (
            params.nodes.length !== 1
        ) {
            return;
        }

        const id =
            params.nodes[0];

        if (
            !Object.prototype
                .hasOwnProperty
                .call(
                    childrenBySeparator,
                    id
                )
        ) {
            return;
        }

        toggleSeparator(
            id
        );
    }
);


// ============================================================
// Collapse all
// ============================================================

document
    .getElementById(
        "collapseAll"
    )
    .addEventListener(

        "click",

        function() {

            const expanded =
                Array.from(
                    expandedSeparators
                );

            expanded.forEach(
                separatorId => {

                    collapse(
                        separatorId
                    );
                }
            );

            network.fit({
                animation: true
            });
        }
    );


// ============================================================
// Expand all
// ============================================================

document
    .getElementById(
        "expandAll"
    )
    .addEventListener(

        "click",

        function() {

            Object.keys(
                childrenBySeparator
            ).forEach(
                separatorId => {

                    expand(
                        separatorId
                    );
                }
            );

            network.fit({
                animation: true
            });
        }
    );


// ============================================================
// Fit
// ============================================================

document
    .getElementById(
        "fitGraph"
    )
    .addEventListener(

        "click",

        function() {

            network.fit({
                animation: true
            });
        }
    );


// ============================================================
// Initial state
// ============================================================

updateStatus();


setTimeout(

    function() {

        network.fit({
            animation: false
        });

    },

    150
);


</script>

</body>

</html>
"""

    # ============================================================
    # Substitute placeholders
    # ============================================================

    html = (
        html
        .replace(
            "__HEIGHT__",
            str(height),
        )
        .replace(
            "__NODES_JSON__",
            nodes_json,
        )
        .replace(
            "__EDGES_JSON__",
            edges_json,
        )
        .replace(
            "__CHILDREN_JSON__",
            children_json,
        )
        .replace(
            "__SEPARATORS_BY_CHILD_JSON__",
            separators_by_child_json,
        )
        .replace(
            "__COLLAPSIBLE_NODES_JSON__",
            collapsible_json,
        )
        .replace(
            "__PURE_SEPARATOR_NODES_JSON__",
            pure_separator_json,
        )
        .replace(
            "__SHOW_BLOCKED_RELATIONS__",
            show_blocked_json,
        )
    )

    # ============================================================
    # Write file
    # ============================================================

    output_file = Path(
        output_file
    )

    output_file.write_text(
        html,
        encoding="utf-8",
    )

    # ============================================================
    # Diagnostics
    # ============================================================

    print(
        "Interactive graph written to:"
    )
    print(
        output_file.resolve()
    )
    print()

    print(
        "Nodes:",
        len(nodes),
    )

    print(
        "Unique node IDs:",
        len(
            {
                node["id"]
                for node in nodes
            }
        ),
    )

    print(
        "Edges:",
        len(edges),
    )

    print(
        "Robust neighbours:",
        len(robust),
    )

    print(
        "Separators:",
        len(separators),
    )

    print(
        "Pure separators initially hidden:",
        len(pure_separator_nodes),
    )

    print(
        "Associated variables:",
        len(associated),
    )

    print(
        "Motifs:",
        len(motifs),
    )

    print(
        "Show blocked relations:",
        show_blocked_relations,
    )

    return output_file


# ============================================================
# Interactive graph (pyvis, physics-based)
# ============================================================

def _build_graph_data(
    results,
    output_var,
    output_label,
    show_blocked_relations,
):
    """
    Build nodes, edges and expand/collapse maps from a
    run_dependence_analysis() result.

    This is the same data the notebook-faithful renderer uses, factored
    out so the pyvis renderer can share it. Roles, initial visibility and
    separator/child maps are identical; only the tooltip text is enriched
    for the associated (blocked) nodes.
    """
    if "separator_motifs_df" in results:
        motifs = results["separator_motifs_df"].copy()
    else:
        analyser = results.get("analyser")
        if analyser is None or not hasattr(
            analyser, "separator_motifs_df_"
        ):
            raise KeyError(
                "results lacks 'separator_motifs_df' / 'analyser'."
            )
        motifs = analyser.separator_motifs_df_.copy()

    if "output_neighbours" not in results:
        raise KeyError("results does not contain 'output_neighbours'.")

    robust = [str(x) for x in results["output_neighbours"]]
    output_var = str(output_var)

    required_columns = {
        "associated_variable",
        "separator",
        "marginal_abs_val",
        "conditional_abs_val",
        "attenuation",
    }
    missing = required_columns - set(motifs.columns)
    if missing:
        raise ValueError(
            f"separator_motifs_df is missing columns: {sorted(missing)}"
        )

    motifs = motifs.dropna(
        subset=["associated_variable", "separator"]
    ).copy()
    motifs["associated_variable"] = motifs[
        "associated_variable"
    ].astype(str)
    motifs["separator"] = motifs["separator"].astype(str)

    separators = sorted(motifs["separator"].drop_duplicates().tolist())
    associated = sorted(
        motifs["associated_variable"].drop_duplicates().tolist()
    )

    robust_set = set(robust)
    separator_set = set(separators)
    associated_set = set(associated)

    all_names = list(
        dict.fromkeys([output_var] + robust + separators + associated)
    )
    node_id = {name: f"n{i}" for i, name in enumerate(all_names)}

    # ----- palette -----
    palette = {
        "output": {"bg": "#f0563a", "bd": "#b23b1e",
                    "hbg": "#ff7355", "hbd": "#8f2c14"},
        "robust": {"bg": "#4a9fe0", "bd": "#1f6fb2",
                    "hbg": "#6fb6ef", "hbd": "#155187"},
        "separator": {"bg": "#5cc98d", "bd": "#2e8b57",
                       "hbg": "#7ad9a6", "hbd": "#1f6b41"},
        "associated": {"bg": "#c3c8d0", "bd": "#8a8f98",
                        "hbg": "#d7dbe1", "hbd": "#6c717a"},
    }

    def color(group):
        s = palette[group]
        return {
            "border": s["bd"],
            "background": s["bg"],
            "highlight": {"border": s["hbd"], "background": s["hbg"]},
            "hover": {"border": s["hbd"], "background": s["hbg"]},
        }

    def tip(inner):
        return (
            "<div style='font-family:Arial;font-size:12px;"
            "line-height:1.45;max-width:280px;padding:2px 4px'>"
            + inner
            + "</div>"
        )

    nodes = []
    for name in all_names:
        if name == output_var:
            nodes.append({
                "id": node_id[name],
                "label": output_label,
                "title": tip(f"<b>Output</b><br>{output_var}"),
                "group": "output",
                "shape": "star",
                "size": 34,
                "color": color("output"),
                "font": {"size": 20, "face": "Arial", "color": "#222"},
                "hidden": False,
            })
            continue

        if name in robust_set:
            inner = f"<b>Robust SCT neighbour</b><br>{name}"
            if name in separator_set:
                n_children = int(motifs["separator"].eq(name).sum())
                inner += (
                    "<br><br><i>Also acts as separator</i><br>"
                    f"{n_children} blocked variable"
                    f"{'' if n_children == 1 else 's'}"
                    "<br><br>Click to expand / collapse"
                )
            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": tip(inner),
                "group": "robust",
                "shape": "dot",
                "size": 24,
                "color": color("robust"),
                "font": {"size": 15, "face": "Arial", "color": "#222"},
                "hidden": False,
            })
            continue

        if name in separator_set:
            n_children = int(motifs["separator"].eq(name).sum())
            inner = (
                f"<b>Separating variable</b><br>{name}<br><br>"
                f"{n_children} blocked variable"
                f"{'' if n_children == 1 else 's'}"
                "<br><br>Click to expand / collapse"
            )
            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": tip(inner),
                "group": "separator",
                "shape": "diamond",
                "size": 18,
                "color": color("separator"),
                "font": {"size": 13, "face": "Arial", "color": "#222"},
                "hidden": True,
            })
            continue

        if name in associated_set:
            row = motifs.loc[
                motifs["associated_variable"].eq(name)
            ].iloc[0]
            m = row["marginal_abs_val"]
            c = row["conditional_abs_val"]
            a = row["attenuation"]
            sep = str(row["separator"])
            m_txt = f"{m:.3f}" if pd.notna(m) else "NA"
            c_txt = f"{c:.3f}" if pd.notna(c) else "NA"
            a_txt = f"{100 * a:.1f}%" if pd.notna(a) else "NA"
            inner = (
                f"<b>Blocked process variable</b><br>{name}"
                "<br><br><u>Blocking conditioning set</u><br>"
                f"{sep}<br><br>"
                f"|r marginal| = {m_txt}<br>"
                f"|r conditioned| = {c_txt}<br>"
                f"Attenuation = {a_txt}"
            )
            nodes.append({
                "id": node_id[name],
                "label": name,
                "title": tip(inner),
                "group": "associated",
                "shape": "dot",
                "size": 11,
                "color": color("associated"),
                "font": {"size": 11, "face": "Arial", "color": "#555"},
                "hidden": True,
            })

    if len({n["id"] for n in nodes}) != len(nodes):
        raise RuntimeError("Duplicate node IDs detected.")

    edges = []
    counter = 0
    for name in robust:
        if name == output_var or name not in node_id:
            continue
        counter += 1
        edges.append({
            "id": f"e{counter}",
            "from": node_id[name],
            "to": node_id[output_var],
            "arrows": "to",
            "width": 2.5,
            "dashes": False,
            "type": "robust",
            "color": {"color": "#4a9fe0", "highlight": "#1f6fb2"},
            "hidden": False,
        })

    for _, row in motifs.iterrows():
        x = str(row["associated_variable"])
        z = str(row["separator"])
        if x not in node_id or z not in node_id:
            continue
        counter += 1
        edges.append({
            "id": f"e{counter}",
            "from": node_id[x],
            "to": node_id[z],
            "width": 1.5,
            "dashes": False,
            "type": "motif",
            "color": {"color": "#5cc98d", "highlight": "#2e8b57"},
            "hidden": True,
        })

    if show_blocked_relations:
        added = set()
        for _, row in motifs.iterrows():
            x = str(row["associated_variable"])
            if (
                x == output_var
                or x in robust_set
                or x in added
                or x not in node_id
            ):
                continue
            added.add(x)
            counter += 1
            edges.append({
                "id": f"e{counter}",
                "from": node_id[x],
                "to": node_id[output_var],
                "arrows": "to",
                "width": 1.1,
                "dashes": True,
                "type": "blocked",
                "color": {"color": "#b9bec6", "highlight": "#8a8f98"},
                "hidden": True,
            })

    children_by_separator = {}
    for separator, group in motifs.groupby("separator", sort=False):
        separator = str(separator)
        if separator not in node_id:
            continue
        children_by_separator[node_id[separator]] = [
            node_id[str(x)]
            for x in group["associated_variable"].drop_duplicates()
            if str(x) in node_id
        ]

    separators_by_child = {}
    for sep_id, children in children_by_separator.items():
        for child_id in children:
            separators_by_child.setdefault(child_id, []).append(sep_id)

    collapsible_nodes = sorted(
        node_id[name]
        for name in associated
        if name not in robust_set
        and name not in separator_set
        and name != output_var
    )
    pure_separator_nodes = sorted(
        node_id[name]
        for name in separators
        if name not in robust_set and name != output_var
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "children_by_separator": children_by_separator,
        "separators_by_child": separators_by_child,
        "collapsible_nodes": collapsible_nodes,
        "pure_separator_nodes": pure_separator_nodes,
        "robust": robust,
        "separators": separators,
        "associated": associated,
        "motifs": motifs,
    }


def plot_interactive_dependency_graph_pyvis(
    results,
    output_var="MBS_SCT_CD",
    output_label="SCT CD",
    output_file="interactive_dependence_graph_pyvis.html",
    height="850px",
    show_blocked_relations=True,
):
    """
    Render the dependence graph with pyvis (the Python wrapper around the
    same vis-network engine as R's visNetwork).

    Same data and expand/collapse semantics as
    plot_interactive_dependency_graph(), but:
      * a balanced, draggable Barnes-Hut force layout instead of the rigid
        hierarchical layout,
      * coloured node groups (output / robust / separator / blocked),
      * HTML tooltips rendered as elements (no raw tags),
      * a styled control bar + legend and a physics freeze toggle.
    """
    from pyvis.network import Network

    data = _build_graph_data(
        results,
        output_var=output_var,
        output_label=output_label,
        show_blocked_relations=show_blocked_relations,
    )
    nodes = data["nodes"]
    edges = data["edges"]

    net = Network(
        height=str(height),
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="remote",
    )
    net.barnes_hut(
        gravity=-12000,
        central_gravity=0.25,
        spring_length=140,
        spring_strength=0.02,
        damping=0.5,
        overlap=0.1,
    )
    net.set_options(json.dumps({
        "physics": {
            "stabilization": {
                "enabled": True,
                "iterations": 250,
                "fit": True,
            },
            "minVelocity": 0.5,
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
        },
        "nodes": {
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "shadow": {
                "enabled": True,
                "size": 6,
                "x": 1,
                "y": 2,
                "color": "rgba(0,0,0,0.15)",
            },
        },
        "edges": {
            "smooth": {"enabled": True, "type": "dynamic"},
            "selectionWidth": 1.5,
        },
    }))

    # pyvis' add_node/add_edge drop dict-valued kwargs (color/font), so we
    # register ids and then write the styling dicts onto the records pyvis
    # serialises verbatim to JSON.
    for node in nodes:
        net.add_node(node["id"], label=node["label"])
    record = {rec["id"]: rec for rec in net.nodes}
    for node in nodes:
        rec = record[node["id"]]
        rec["label"] = node["label"]
        rec["title"] = node["title"]
        rec["group"] = node["group"]
        rec["shape"] = node["shape"]
        rec["size"] = node["size"]
        rec["hidden"] = bool(node["hidden"])
        rec["color"] = node["color"]
        rec["font"] = node["font"]

    for edge in edges:
        net.add_edge(edge["from"], edge["to"], width=edge["width"])
    for rec, edge in zip(net.edges, edges):
        rec["id"] = edge["id"]
        rec["type"] = edge["type"]
        rec["width"] = edge["width"]
        rec["dashes"] = bool(edge["dashes"])
        rec["hidden"] = bool(edge["hidden"])
        if edge.get("arrows"):
            rec["arrows"] = edge["arrows"]
        rec["color"] = edge["color"]

    base_html = net.generate_html(notebook=False)

    help_text = (
        "Drag nodes to rearrange. Click a separator (diamond) to expand / "
        "collapse the variables it explains away. Solid = robust relation, "
        "dashed = relation removed by conditioning."
        if show_blocked_relations
        else "Drag nodes to rearrange. Click a separator (diamond) to "
        "expand / collapse its blocked variables."
    )

    controls_html = (
        "<div id=\"cd-controls\">"
        "<span class=\"cd-title\">Dependence graph</span>"
        "<button id=\"cd-expandAll\">Expand all</button>"
        "<button id=\"cd-collapseAll\">Collapse all</button>"
        "<button id=\"cd-fit\">Fit</button>"
        "<button id=\"cd-physics\">Freeze layout</button>"
        "<span id=\"cd-legend\">"
        "<span class=\"cd-chip\"><span class=\"cd-dot\" "
        "style=\"background:#f0563a\"></span>Output</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot\" "
        "style=\"background:#4a9fe0\"></span>Robust neighbour</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot cd-diamond\" "
        "style=\"background:#5cc98d\"></span>Separator</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot cd-small\" "
        "style=\"background:#c3c8d0\"></span>Blocked</span>"
        "</span><span id=\"cd-status\"></span></div>"
        "<div id=\"cd-help\">" + help_text + "</div>"
    )

    controls_css = """
<style>
  body { margin: 0; font-family: Arial, Helvetica, sans-serif;
         background: #f4f6f9; }
  #cd-controls { display: flex; align-items: center; flex-wrap: wrap;
    gap: 8px; padding: 10px 16px; background: #ffffff;
    border-bottom: 1px solid #e2e6ec;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  #cd-controls .cd-title { font-weight: 600; font-size: 15px;
    color: #2a2f36; margin-right: 8px; }
  #cd-controls button { padding: 6px 12px; font-size: 13px;
    border: 1px solid #cfd5dd; border-radius: 6px; background: #fff;
    color: #333; cursor: pointer; transition: background .15s; }
  #cd-controls button:hover { background: #eef2f7; }
  #cd-controls button.cd-active { background: #e7f0fb;
    border-color: #4a9fe0; color: #1f6fb2; }
  #cd-legend { display: flex; align-items: center; gap: 14px;
    margin-left: 12px; font-size: 12px; color: #555; }
  .cd-chip { display: inline-flex; align-items: center; gap: 6px; }
  .cd-dot { width: 13px; height: 13px; border-radius: 50%;
    display: inline-block; border: 1px solid rgba(0,0,0,0.2); }
  .cd-dot.cd-small { width: 9px; height: 9px; }
  .cd-dot.cd-diamond { border-radius: 2px; transform: rotate(45deg); }
  #cd-status { margin-left: auto; font-size: 12px; color: #888; }
  #cd-help { padding: 7px 16px; font-size: 12px; color: #6a7079;
    background: #fbfcfd; border-bottom: 1px solid #eef1f5; }
  #mynetwork { border: none !important; }
  .card { border: none !important; box-shadow: none !important; }
</style>
"""

    expand_collapse_js = """
<script type="text/javascript">
(function () {
  function init() {
    if (typeof network === "undefined" || !network) {
      setTimeout(init, 60); return;
    }
    // Render string titles as HTML elements (vis-network shows raw tags
    // otherwise).
    function htmlTitle(html) {
      var el = document.createElement("div");
      el.innerHTML = html; return el;
    }
    (function () {
      var upd = [];
      nodes.get().forEach(function (n) {
        if (n && typeof n.title === "string") {
          upd.push({ id: n.id, title: htmlTitle(n.title) });
        }
      });
      if (upd.length) { nodes.update(upd); }
    })();

    var childrenBySeparator = __CHILDREN_JSON__;
    var separatorsByChild = __SEPARATORS_BY_CHILD_JSON__;
    var collapsibleNodes = new Set(__COLLAPSIBLE_NODES_JSON__);
    var pureSeparatorNodes = new Set(__PURE_SEPARATOR_NODES_JSON__);
    var showBlockedRelations = __SHOW_BLOCKED_RELATIONS__;
    var expandedSeparators = new Set();
    var statusEl = document.getElementById("cd-status");

    function updateStatus() {
      if (statusEl) {
        statusEl.textContent =
          "Expanded separators: " + expandedSeparators.size;
      }
    }
    function childStillNeeded(childId) {
      var p = separatorsByChild[childId] || [];
      for (var i = 0; i < p.length; i++) {
        if (expandedSeparators.has(p[i])) { return true; }
      }
      return false;
    }
    function updateBlockedEdge(childId) {
      if (!showBlockedRelations) { return; }
      var node = nodes.get(childId);
      if (!node) { return; }
      edges.get({ filter: function (e) {
        return e.type === "blocked" && e.from === childId;
      }}).forEach(function (e) {
        edges.update({ id: e.id, hidden: Boolean(node.hidden) });
      });
    }
    function revealSeparator(sepId) {
      if (pureSeparatorNodes.has(sepId)) {
        nodes.update({ id: sepId, hidden: false });
      }
    }
    function hideSeparatorIfUnused(sepId) {
      if (!pureSeparatorNodes.has(sepId)) { return; }
      if (expandedSeparators.has(sepId)) { return; }
      nodes.update({ id: sepId, hidden: true });
    }
    function expand(sepId) {
      var ch = childrenBySeparator[sepId];
      if (!ch) { return; }
      revealSeparator(sepId);
      expandedSeparators.add(sepId);
      ch.forEach(function (childId) {
        if (collapsibleNodes.has(childId)) {
          nodes.update({ id: childId, hidden: false });
        }
        edges.get({ filter: function (e) {
          return e.type === "motif" && e.from === childId && e.to === sepId;
        }}).forEach(function (e) {
          edges.update({ id: e.id, hidden: false });
        });
        updateBlockedEdge(childId);
      });
      updateStatus();
    }
    function collapse(sepId) {
      var ch = childrenBySeparator[sepId];
      if (!ch) { return; }
      expandedSeparators.delete(sepId);
      ch.forEach(function (childId) {
        edges.get({ filter: function (e) {
          return e.type === "motif" && e.from === childId && e.to === sepId;
        }}).forEach(function (e) {
          edges.update({ id: e.id, hidden: true });
        });
        if (collapsibleNodes.has(childId) && !childStillNeeded(childId)) {
          nodes.update({ id: childId, hidden: true });
        }
        updateBlockedEdge(childId);
      });
      hideSeparatorIfUnused(sepId);
      updateStatus();
    }
    function toggle(sepId) {
      if (expandedSeparators.has(sepId)) { collapse(sepId); }
      else { expand(sepId); }
    }
    network.on("click", function (params) {
      if (params.nodes.length !== 1) { return; }
      var id = params.nodes[0];
      if (!Object.prototype.hasOwnProperty.call(childrenBySeparator, id)) {
        return;
      }
      toggle(id);
    });
    var eA = document.getElementById("cd-expandAll");
    var cA = document.getElementById("cd-collapseAll");
    var fB = document.getElementById("cd-fit");
    var pB = document.getElementById("cd-physics");
    if (eA) { eA.addEventListener("click", function () {
      Object.keys(childrenBySeparator).forEach(function (s) { expand(s); });
      network.fit({ animation: true });
    }); }
    if (cA) { cA.addEventListener("click", function () {
      Array.from(expandedSeparators).forEach(function (s) { collapse(s); });
      network.fit({ animation: true });
    }); }
    if (fB) { fB.addEventListener("click", function () {
      network.fit({ animation: true });
    }); }
    if (pB) {
      var on = true;
      pB.addEventListener("click", function () {
        on = !on;
        network.setOptions({ physics: { enabled: on } });
        pB.textContent = on ? "Freeze layout" : "Resume layout";
        pB.classList.toggle("cd-active", !on);
      });
    }
    updateStatus();
    network.once("stabilizationIterationsDone", function () {
      network.fit({ animation: false });
    });
  }
  init();
})();
</script>
"""

    expand_collapse_js = (
        expand_collapse_js
        .replace(
            "__CHILDREN_JSON__",
            json.dumps(data["children_by_separator"], ensure_ascii=False),
        )
        .replace(
            "__SEPARATORS_BY_CHILD_JSON__",
            json.dumps(data["separators_by_child"], ensure_ascii=False),
        )
        .replace(
            "__COLLAPSIBLE_NODES_JSON__",
            json.dumps(data["collapsible_nodes"], ensure_ascii=False),
        )
        .replace(
            "__PURE_SEPARATOR_NODES_JSON__",
            json.dumps(data["pure_separator_nodes"], ensure_ascii=False),
        )
        .replace(
            "__SHOW_BLOCKED_RELATIONS__",
            json.dumps(bool(show_blocked_relations)),
        )
    )

    html = base_html
    if "</head>" in html:
        html = html.replace("</head>", controls_css + "</head>", 1)
    else:
        html = controls_css + html
    if "<body>" in html:
        html = html.replace("<body>", "<body>\n" + controls_html, 1)
    else:
        html = controls_html + html
    if "</body>" in html:
        html = html.replace("</body>", expand_collapse_js + "</body>", 1)
    else:
        html = html + expand_collapse_js

    output_file = Path(output_file)
    output_file.write_text(html, encoding="utf-8")

    print("Interactive pyvis graph written to:")
    print(output_file.resolve())
    print("Nodes:", len(nodes))
    print("Edges:", len(edges))
    print("Robust neighbours:", len(data["robust"]))
    print("Separators:", len(data["separators"]))
    print("Associated variables:", len(data["associated"]))

    return output_file


# ============================================================
# Main
# ============================================================

def main():
    # ============================================================
    # 1. Load raw data
    # ============================================================

    turnup_data = pd.read_parquet(DATA_PATH)


    # ============================================================
    # 2. Build variable list
    # ============================================================


    created_vars = []


    # ============================================================
    # 3. Feature engineering
    # ============================================================

    turnup_data = _feature_engineering(
        turnup_data,
        setpoint_df,
        steam_null=False,
        clip=True,
    )

    ctl_vars = unique_in_order(
        v for v in turnup_data.columns
        if "mbs" not in v.lower() and
        "wedge" not in v.lower() and
        "m2" not in v.lower() and
        "€" not in v and
        "grade" not in v.lower() and
        "overprocessing" not in v.lower() and
        "underprocessing" not in v.lower() and 
        "grammage_check" not in v.lower() and
        "paper_type" not in v.lower()
    )


    turnup_data = (
        turnup_data
        .set_index("Wedge_Time")
        .sort_index()
    )


    if len(created_vars) > 0:

        fc = FeatureCreator(
            features_to_create=created_vars,
            features_to_keep=list(
                set(
                    turnup_data.columns.to_list()
                    + created_vars
                )
            ),
        )

        turnup_data = fc.fit_transform(turnup_data)

        fr = fc.features_required()

    else:
        fr = []



    var_names = [Y_COLUMN]  + ctl_vars


    # ============================================================
    # 4. Create delta basis weight
    # ============================================================
    #
    # This is created from the ORIGINAL, non-EWM-filtered values.
    #
    # Ideally:
    #
    #   actual basis weight - target basis weight
    #
    # Here grammage is assumed to represent the intended/nominal
    # basis weight.
    # ============================================================

    turnup_data["delta_basis_weight"] = (
        turnup_data["Current_basis_weight"]
        - turnup_data["grammage"]
    )



    # ============================================================
    # 5. Define variables for the within-grade analysis
    # ============================================================
    #
    # Remove absolute basis weight and grammage.
    # Replace them with delta_basis_weight.
    # ============================================================

    within_product_vars = [
        v
        for v in var_names
        if v not in {
            "Current_basis_weight",
            "grammage",
            "retention"
        }
    ]


    within_product_vars = unique_in_order(
        within_product_vars
    )


    # ============================================================
    # 6. Within-grade transformation
    # ============================================================
    #
    # IMPORTANT:
    # This is done BEFORE EWM filtering.
    #
    # For each variable:
    #
    #   X_within = X - mean(X | grade)
    #
    # Therefore all variables become deviations from their
    # grade-specific mean.
    #
    # Example:
    #
    #   SCT_within =
    #       SCT - average SCT for that grade
    #
    # ============================================================

    GRADE_VAR = "AB_Grade_ID"

    turnup_within = turnup_data.copy()


    grade_means = (
        turnup_data
        .groupby(GRADE_VAR)[within_product_vars]
        .transform("mean")
    )


    turnup_within[within_product_vars] = (
        turnup_data[within_product_vars]
        - grade_means
    )


    # ============================================================
    # 7. Optional sanity checks
    # ============================================================

    print(
        turnup_within
        .groupby(GRADE_VAR)[within_product_vars]
        .mean()
        .round(6)
    )

    turnup_within[within_product_vars] = (
        turnup_within[within_product_vars]
        .astype("float64")
    )

    filtered_within = ewm_filter_by_segment(
        turnup_within,
        columns=within_product_vars,
        halflife="240min",
    )

    analysis_df = (
        filtered_within[within_product_vars]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    data_array = analysis_df.to_numpy(dtype=float)

    dataframe_within = pp.DataFrame(
        data_array,
        var_names=within_product_vars,
    )

    # ============================================================
    # 10. Conditional-dependence skeleton
    # ============================================================

    results_within = run_dependence_analysis(
        dataframe=dataframe_within,
        output_var=Y_COLUMN,
        max_cond_dim=3,
        min_abs_val=0.05,
        max_target_neighbours=40,
    )

    interactive_file = plot_interactive_dependency_graph_pyvis(
        results_within,
        output_var="MBS_SCT_CD",
        output_label="SCT CD",
        output_file="interactive_dependence_graph.html",
        height="850px",
        show_blocked_relations=True,
    )

    print()
    print("Analysis complete.")
    print("Interactive graph:", interactive_file.resolve())

    return results_within, interactive_file

if __name__ == "__main__":
    results_within, interactive_file = main()
