"""
Within-grade conditional-dependence analysis for SCT CD.

Pipeline
--------
1. Load turnup data.
2. Apply project feature engineering.
3. Select candidate process variables.
4. Create delta basis weight.
5. Convert variables to within-grade deviations.
6. Apply segment-wise EWM filtering.
7. Build a Tigramite dataframe.
8. Run the conditional-dependence analysis.
9. Generate an interactive HTML graph.

Project-specific objects still expected to be available in the runtime:
    DATA_PATH
    setpoint_df
    Y_COLUMN
    _feature_engineering

Everything specific to the conditional-dependence workflow is included here:
    unique_in_order
    ewm_filter_by_segment
    ConditionalDependenceSkeleton
    run_dependence_analysis
    plot_interactive_dependency_graph
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# Configuration
# ============================================================

GRADE_VAR = "AB_Grade_ID"
DEFAULT_OUTPUT_VAR = "MBS_SCT_CD"

EWM_HALFLIFE = "240min"
MAX_COND_DIM = 3
MIN_ABS_VAL = 0.05

INTERACTIVE_OUTPUT = "interactive_dependence_graph.html"
INTERACTIVE_HEIGHT = "850px"
SHOW_BLOCKED_RELATIONS = True


# Variables that should not enter the process-variable pool.
EXCLUDED_EXACT = {
    "Current_basis_weight",
    "grammage",
    "retention",
}

EXCLUDED_NAME_PARTS = (
    "mbs",
    "wedge",
    "m2",
    "grade",
    "overprocessing",
    "underprocessing",
    "grammage_check",
    "paper_type",
)



# ============================================================
# General helpers
# ============================================================

def unique_in_order(values):
    """Return unique values while preserving their original order."""
    return list(dict.fromkeys(values))


# ============================================================
# Segment-wise EWM filter
# ============================================================

def ewm_filter_by_segment(
    data: pd.DataFrame,
    columns: Iterable[str],
    *,
    grade_var: str = GRADE_VAR,
    halflife: str = EWM_HALFLIFE,
    max_gap: str = "12h",
) -> pd.DataFrame:
    """
    Apply a time-aware EWM independently inside continuous production segments.

    A new segment starts when:
        1. AB_Grade_ID changes, or
        2. the time gap between consecutive reels exceeds max_gap.

    The dataframe must already be indexed by Wedge_Time.

    The selected columns are explicitly converted to float64 before assigning
    EWM values. This avoids pandas' incompatible-dtype FutureWarning.
    """
    result = data.copy()
    columns = list(columns)

    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)

    result = result.sort_index()

    if grade_var not in result.columns:
        raise KeyError(
            f"Grade variable '{grade_var}' is required for segmentation."
        )

    missing = [
        column
        for column in columns
        if column not in result.columns
    ]
    if missing:
        raise KeyError(
            f"Columns missing for EWM filtering: {missing}"
        )

    # Important: make the destination columns floating point before
    # assigning the filtered values.
    result[columns] = (
        result[columns]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    grade_change = (
        result[grade_var]
        .ne(result[grade_var].shift())
    )

    gap_change = (
        result.index
        .to_series()
        .diff()
        .gt(pd.Timedelta(max_gap))
        .to_numpy()
    )

    new_segment = (
        grade_change.to_numpy()
        | gap_change
    )

    segment = pd.Series(
        np.cumsum(new_segment),
        index=result.index,
        name="_segment",
    )

    filtered_parts = []

    for _, positions in segment.groupby(segment).groups.items():
        part = result.loc[positions, columns]

        if part.empty:
            continue

        filtered = (
            part
            .ewm(
                halflife=pd.Timedelta(halflife),
                times=part.index,
                adjust=True,
                ignore_na=True,
            )
            .mean()
        )

        filtered_parts.append(filtered)

    if filtered_parts:
        filtered_values = (
            pd.concat(filtered_parts)
            .sort_index()
        )
        result.loc[
            filtered_values.index,
            columns,
        ] = filtered_values[columns]

    return result


# ============================================================
# Conditional-dependence skeleton
# ============================================================

class ConditionalDependenceSkeleton:
    """
    Target-centred conditional-dependence discovery.

    Phase 1
    -------
    Screen the requested output variable against the complete candidate pool.

    k = 0:
        Keep variables whose marginal |r| >= min_abs_val.

    k = 1:
        For every marginally associated X, test Y -- X | Z for every
        possible single Z in the ORIGINAL candidate pool.

        The best separator is the Z producing the smallest absolute partial
        correlation. X is removed from the robust target neighbourhood when:

            |r(Y, X | Z_best)| < min_abs_val

        All such blocking structures are stored in separator_motifs_df_.

    Phase 2
    -------
    Build a complete graph only on:

        output + robust target neighbours

    and perform the deeper Stable-PC-style conditional search up to
    max_cond_dim.

    Practical independence is deliberately defined by effect size:

        abs(partial correlation) < min_abs_val

    rather than by p-value alone.
    """

    def __init__(
        self,
        dataframe,
        *,
        output_var: str,
        max_cond_dim: int = 3,
        min_abs_val: float = 0.05,
        alpha: float = 0.05,
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
            name: i
            for i, name in enumerate(self.var_names)
        }

        if output_var not in self.name_to_idx:
            raise KeyError(
                f"Output variable '{output_var}' is not present in dataframe."
            )

        self.triple_parcorr = ParCorr(
            significance="analytic"
        )
        self.triple_parcorr.set_dataframe(
            self.dataframe
        )

        self.graph_ = None
        self.separation_df_ = None
        self.edge_diagnostics_df_ = None
        self.target_screening_df_ = None
        self.separator_motifs_df_ = None
        self.separator_neighbours_df_ = None
        self.selected_variables_ = None
        self.separating_variables_ = None

    # --------------------------------------------------------
    # CI test
    # --------------------------------------------------------

    def parcorr_test(
        self,
        x_name: str,
        y_name: str,
        conditioning_set=(),
    ) -> dict:
        """Run a contemporaneous Tigramite ParCorr test."""
        x = self.name_to_idx[x_name]
        y = self.name_to_idx[y_name]

        z_names = tuple(conditioning_set)
        z = [
            (self.name_to_idx[name], 0)
            for name in z_names
        ]

        val, pval = self.triple_parcorr.run_test(
            X=[(x, 0)],
            Y=[(y, 0)],
            Z=z,
            tau_max=0,
        )

        return {
            "X": x_name,
            "Y": y_name,
            "conditioning_set": z_names,
            "cond_dim": len(z_names),
            "val": float(val),
            "abs_val": abs(float(val)),
            "pval": float(pval),
        }

    def practically_independent(
        self,
        result: dict,
    ) -> bool:
        """
        Practical independence criterion.

        p-value is retained as a diagnostic, but the graph-removal criterion
        is the absolute partial-correlation threshold.
        """
        return (
            result["abs_val"]
            < self.min_abs_val
        )

    # --------------------------------------------------------
    # Target screening
    # --------------------------------------------------------

    def _screen_target(
        self,
        variable_names: list[str],
    ):
        y = self.output_var

        screening_records = []
        motif_records = []
        robust_neighbours = []

        candidates = [
            name
            for name in variable_names
            if name != y
        ]

        if self.verbosity:
            print()
            print("=" * 80)
            print("TARGET SCREENING")
            print("=" * 80)
            print(f"Output: {y}")
            print(f"Candidate variables: {len(candidates)}")
            print(
                "Practical-independence threshold: "
                f"|r| < {self.min_abs_val}"
            )

        # ----------------------------------------------------
        # k = 0: marginal target screening
        # ----------------------------------------------------

        marginal_results = {}

        for i, x in enumerate(candidates, start=1):
            result = self.parcorr_test(
                x,
                y,
                (),
            )
            marginal_results[x] = result

            if (
                self.verbosity >= 2
                and (
                    i % 50 == 0
                    or i == len(candidates)
                )
            ):
                print(
                    f"k=0 target tests: "
                    f"{i}/{len(candidates)}"
                )

        marginal_candidates = [
            x
            for x in candidates
            if not self.practically_independent(
                marginal_results[x]
            )
        ]

        if self.verbosity:
            print(
                "Marginally associated with target:",
                len(marginal_candidates),
            )

        # ----------------------------------------------------
        # k = 1: exhaustive single-Z screening
        # ----------------------------------------------------

        for i, x in enumerate(candidates, start=1):
            marginal = marginal_results[x]

            # If X is already practically independent marginally,
            # no separator search is needed.
            if self.practically_independent(marginal):
                screening_records.append({
                    "X": x,
                    "Y": y,
                    "marginal_val": marginal["val"],
                    "marginal_abs_val": marginal["abs_val"],
                    "marginal_p": marginal["pval"],
                    "best_separator": None,
                    "conditional_val": np.nan,
                    "conditional_abs_val": np.nan,
                    "conditional_p": np.nan,
                    "retained": False,
                })
                continue

            conditioning_candidates = [
                z
                for z in variable_names
                if z not in {x, y}
            ]

            best = None
            best_separator = None

            for z in conditioning_candidates:
                conditional = self.parcorr_test(
                    x,
                    y,
                    (z,),
                )

                if (
                    best is None
                    or conditional["abs_val"]
                    < best["abs_val"]
                ):
                    best = conditional
                    best_separator = z

            # Degenerate case: only X and Y exist.
            if best is None:
                robust_neighbours.append(x)

                screening_records.append({
                    "X": x,
                    "Y": y,
                    "marginal_val": marginal["val"],
                    "marginal_abs_val": marginal["abs_val"],
                    "marginal_p": marginal["pval"],
                    "best_separator": None,
                    "conditional_val": np.nan,
                    "conditional_abs_val": np.nan,
                    "conditional_p": np.nan,
                    "retained": True,
                })
                continue

            retained = not self.practically_independent(
                best
            )

            screening_records.append({
                "X": x,
                "Y": y,
                "marginal_val": marginal["val"],
                "marginal_abs_val": marginal["abs_val"],
                "marginal_p": marginal["pval"],
                "best_separator": best_separator,
                "conditional_val": best["val"],
                "conditional_abs_val": best["abs_val"],
                "conditional_p": best["pval"],
                "retained": retained,
            })

            if retained:
                robust_neighbours.append(x)
            else:
                attenuation = (
                    1.0
                    - best["abs_val"]
                    / marginal["abs_val"]
                    if marginal["abs_val"] > 0
                    else np.nan
                )

                motif_records.append({
                    "associated_variable": x,
                    "separator": best_separator,
                    "target": y,
                    "marginal_val": marginal["val"],
                    "marginal_abs_val": marginal["abs_val"],
                    "marginal_p": marginal["pval"],
                    "conditional_val": best["val"],
                    "conditional_abs_val": best["abs_val"],
                    "conditional_p": best["pval"],
                    "attenuation": attenuation,
                })

            if (
                self.verbosity >= 1
                and (
                    i % 25 == 0
                    or i == len(candidates)
                )
            ):
                print(
                    "Target k=1 progress: "
                    f"{i}/{len(candidates)}"
                )

        self.target_screening_df_ = pd.DataFrame(
            screening_records
        )

        self.separator_motifs_df_ = pd.DataFrame(
            motif_records,
            columns=[
                "associated_variable",
                "separator",
                "target",
                "marginal_val",
                "marginal_abs_val",
                "marginal_p",
                "conditional_val",
                "conditional_abs_val",
                "conditional_p",
                "attenuation",
            ],
        )

        self.separating_variables_ = unique_in_order(
            self.separator_motifs_df_["separator"]
            .dropna()
            .tolist()
        )

        if self.verbosity:
            print(
                "Robust target neighbours after k=1:",
                len(robust_neighbours),
            )
            print(
                "Validated blocking motifs:",
                len(self.separator_motifs_df_),
            )

        return robust_neighbours


    # --------------------------------------------------------
    # Separator neighbourhood discovery
    # --------------------------------------------------------

    def _discover_separator_neighbours(
        self,
        variable_names: list[str],
    ) -> pd.DataFrame:
        """
        Find variables genuinely related to each separator.

        This is intentionally different from separator_motifs_df_:

        separator_motifs_df_
            X is related to SCT CD, but X ⟂ SCT CD | Z.

        separator_neighbours_df_
            X is related to Z, and the X--Z relationship cannot be
            removed by conditioning on any single W.

        For every separator Z we therefore:
          1. test Z--X marginally;
          2. if it survives the practical threshold, test Z--X | W
             for every possible single W;
          3. retain X only if the best W still leaves
             |r(Z, X | W)| >= min_abs_val.

        This keeps the search local around the separators and avoids
        rebuilding a complete graph over the full variable pool.
        """
        separators = unique_in_order(
            self.separating_variables_ or []
        )

        columns = [
            "separator",
            "neighbour",
            "marginal_val",
            "marginal_abs_val",
            "marginal_p",
            "best_conditioner",
            "conditional_val",
            "conditional_abs_val",
            "conditional_p",
            "retained",
        ]

        if not separators:
            return pd.DataFrame(columns=columns)

        records = []

        if self.verbosity:
            print()
            print("=" * 80)
            print("SEPARATOR NEIGHBOURHOOD DISCOVERY")
            print("=" * 80)
            print(f"Separators: {len(separators)}")

        for separator_index, z in enumerate(separators, start=1):

            candidates = [
                x for x in variable_names
                if x != z
            ]

            retained_count = 0

            for x in candidates:

                marginal = self.parcorr_test(
                    z,
                    x,
                    (),
                )

                if self.practically_independent(marginal):
                    records.append({
                        "separator": z,
                        "neighbour": x,
                        "marginal_val": marginal["val"],
                        "marginal_abs_val": marginal["abs_val"],
                        "marginal_p": marginal["pval"],
                        "best_conditioner": None,
                        "conditional_val": np.nan,
                        "conditional_abs_val": np.nan,
                        "conditional_p": np.nan,
                        "retained": False,
                    })
                    continue

                conditioning_candidates = [
                    w for w in variable_names
                    if w not in {z, x}
                ]

                best = None
                best_conditioner = None

                for w in conditioning_candidates:
                    conditional = self.parcorr_test(
                        z,
                        x,
                        (w,),
                    )

                    if (
                        best is None
                        or conditional["abs_val"] < best["abs_val"]
                    ):
                        best = conditional
                        best_conditioner = w

                if best is None:
                    retained = True
                    conditional_val = np.nan
                    conditional_abs_val = np.nan
                    conditional_p = np.nan
                else:
                    retained = not self.practically_independent(best)
                    conditional_val = best["val"]
                    conditional_abs_val = best["abs_val"]
                    conditional_p = best["pval"]

                if retained:
                    retained_count += 1

                records.append({
                    "separator": z,
                    "neighbour": x,
                    "marginal_val": marginal["val"],
                    "marginal_abs_val": marginal["abs_val"],
                    "marginal_p": marginal["pval"],
                    "best_conditioner": best_conditioner,
                    "conditional_val": conditional_val,
                    "conditional_abs_val": conditional_abs_val,
                    "conditional_p": conditional_p,
                    "retained": retained,
                })

            if self.verbosity:
                print(
                    f"Separator {separator_index}/{len(separators)}: "
                    f"{z} -> {retained_count} robust local neighbours"
                )

        return pd.DataFrame(
            records,
            columns=columns,
        )

    # --------------------------------------------------------
    # Reduced robust skeleton
    # --------------------------------------------------------

    def _fit_reduced_skeleton(
        self,
        selected_variables: list[str],
    ):
        graph = nx.Graph()
        graph.add_nodes_from(selected_variables)

        graph.add_edges_from(
            combinations(
                selected_variables,
                2,
            )
        )

        separation_records = []
        diagnostic_records = []

        if self.verbosity:
            print()
            print("=" * 80)
            print("ROBUST REDUCED SKELETON")
            print("=" * 80)
            print(
                "Variables:",
                len(selected_variables),
            )
            print(
                "Initial edges:",
                graph.number_of_edges(),
            )

        # Stable-PC style: adjacency sets are frozen within each k.
        for k in range(
            self.max_cond_dim + 1
        ):
            if graph.number_of_edges() == 0:
                break

            adjacency_snapshot = {
                node: set(graph.neighbors(node))
                for node in graph.nodes
            }

            edges_at_level = list(
                graph.edges()
            )

            removed_this_level = 0
            tested_this_level = 0

            for x, y in edges_at_level:
                if not graph.has_edge(x, y):
                    continue

                candidates = (
                    adjacency_snapshot[x]
                    | adjacency_snapshot[y]
                ) - {x, y}

                if len(candidates) < k:
                    continue

                best_result = None
                best_conditioning_set = None
                separating_result = None

                for cond_set in combinations(
                    sorted(candidates),
                    k,
                ):
                    result = self.parcorr_test(
                        x,
                        y,
                        cond_set,
                    )

                    tested_this_level += 1

                    if (
                        best_result is None
                        or result["abs_val"]
                        < best_result["abs_val"]
                    ):
                        best_result = result
                        best_conditioning_set = cond_set

                    # For the deeper reduced skeleton, the first
                    # valid separating set is sufficient.
                    if self.practically_independent(
                        result
                    ):
                        separating_result = result
                        break

                if best_result is not None:
                    diagnostic_records.append({
                        "X": x,
                        "Y": y,
                        "conditioning_set": tuple(
                            best_conditioning_set
                        ),
                        "cond_dim": k,
                        "val": best_result["val"],
                        "abs_val": best_result["abs_val"],
                        "pval": best_result["pval"],
                        "status": (
                            "INDEPENDENT"
                            if self.practically_independent(
                                best_result
                            )
                            else "DEPENDENT"
                        ),
                    })

                if separating_result is not None:
                    graph.remove_edge(x, y)
                    removed_this_level += 1

                    separation_records.append({
                        "X": x,
                        "Y": y,
                        "conditioning_set": tuple(
                            separating_result[
                                "conditioning_set"
                            ]
                        ),
                        "cond_dim": k,
                        "val": separating_result["val"],
                        "abs_val": separating_result["abs_val"],
                        "pval": separating_result["pval"],
                    })

            if self.verbosity:
                print(
                    f"k={k}: tested {tested_this_level:,} CI relations; "
                    f"removed {removed_this_level:,}; "
                    f"remaining edges {graph.number_of_edges():,}"
                )

            # No edge can be tested at the next level if every
            # adjacency union is too small.
            next_k = k + 1

            if next_k > self.max_cond_dim:
                break

            can_continue = any(
                len(
                    (
                        set(graph.neighbors(x))
                        | set(graph.neighbors(y))
                    ) - {x, y}
                )
                >= next_k
                for x, y in graph.edges()
            )

            if not can_continue:
                break

        self.separation_df_ = pd.DataFrame(
            separation_records,
            columns=[
                "X",
                "Y",
                "conditioning_set",
                "cond_dim",
                "val",
                "abs_val",
                "pval",
            ],
        )

        self.edge_diagnostics_df_ = pd.DataFrame(
            diagnostic_records,
            columns=[
                "X",
                "Y",
                "conditioning_set",
                "cond_dim",
                "val",
                "abs_val",
                "pval",
                "status",
            ],
        )

        return graph

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def fit(
        self,
        variable_names: Optional[Iterable[str]] = None,
    ):
        if variable_names is None:
            variable_names = self.var_names
        else:
            variable_names = unique_in_order(
                variable_names
            )

        unknown = [
            name
            for name in variable_names
            if name not in self.name_to_idx
        ]

        if unknown:
            raise KeyError(
                "Variables not found in Tigramite dataframe: "
                f"{unknown}"
            )

        if self.output_var not in variable_names:
            variable_names = [
                self.output_var,
                *variable_names,
            ]

        variable_names = unique_in_order(
            variable_names
        )

        robust_neighbours = self._screen_target(
            variable_names
        )

        # Separate question from target blocking:
        # which variables are genuinely related to each separator?
        self.separator_neighbours_df_ = (
            self._discover_separator_neighbours(
                variable_names
            )
        )

        self.selected_variables_ = unique_in_order(
            [
                self.output_var,
                *robust_neighbours,
            ]
        )

        self.graph_ = self._fit_reduced_skeleton(
            self.selected_variables_
        )

        return self.graph_

    def neighbours(
        self,
        variable: str,
    ) -> list[str]:
        if self.graph_ is None:
            raise RuntimeError(
                "Call fit() first."
            )

        if variable not in self.graph_:
            return []

        return list(
            self.graph_.neighbors(variable)
        )

    def local_subgraph(
        self,
        variable: str,
        radius: int = 1,
    ):
        if self.graph_ is None:
            raise RuntimeError(
                "Call fit() first."
            )

        if variable not in self.graph_:
            raise KeyError(
                f"'{variable}' is not in the fitted graph."
            )

        nodes = nx.single_source_shortest_path_length(
            self.graph_,
            variable,
            cutoff=radius,
        ).keys()

        return self.graph_.subgraph(
            nodes
        ).copy()

    def separating_record(
        self,
        x_name: str,
        y_name: str,
    ) -> pd.DataFrame:
        if self.separation_df_ is None:
            raise RuntimeError(
                "Call fit() first."
            )

        a, b = sorted(
            (x_name, y_name)
        )

        temp = self.separation_df_.copy()

        mask = temp.apply(
            lambda row:
                tuple(
                    sorted(
                        (row["X"], row["Y"])
                    )
                )
                == (a, b),
            axis=1,
        )

        return temp.loc[mask].copy()

    @staticmethod
    def plot_edge_table(
        graph,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "X": x,
                    "Y": y,
                }
                for x, y in graph.edges()
            ],
            columns=["X", "Y"],
        )


# ============================================================
# Analysis wrapper
# ============================================================

def run_dependence_analysis(
    dataframe,
    *,
    output_var: str,
    variable_names: Optional[Iterable[str]] = None,
    max_cond_dim: int = MAX_COND_DIM,
    min_abs_val: float = MIN_ABS_VAL,
    alpha: float = 0.05,
    output_dir: Optional[Path | str] = None,
    verbosity: int = 1,
):
    """
    Run the target-centred conditional-dependence workflow and return
    the fitted analyser plus the main result tables.
    """
    analyser = ConditionalDependenceSkeleton(
        dataframe=dataframe,
        output_var=output_var,
        max_cond_dim=max_cond_dim,
        min_abs_val=min_abs_val,
        alpha=alpha,
        verbosity=verbosity,
    )

    graph = analyser.fit(
        variable_names=variable_names
    )

    output_neighbours = analyser.neighbours(
        output_var
    )

    local_graph = analyser.local_subgraph(
        output_var,
        radius=1,
    )

    results = {
        "analyser": analyser,
        "graph": graph,
        "separation_df": analyser.separation_df_,
        "edge_diagnostics_df": analyser.edge_diagnostics_df_,
        "target_screening_df": analyser.target_screening_df_,
        "separator_motifs_df": analyser.separator_motifs_df_,
        "separator_neighbours_df": analyser.separator_neighbours_df_,
        "separating_variables": analyser.separating_variables_,
        "selected_variables": analyser.selected_variables_,
        "output_neighbours": output_neighbours,
        "local_graph": local_graph,
        "local_edges": analyser.plot_edge_table(
            local_graph
        ),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        analyser.target_screening_df_.to_csv(
            output_dir / "target_screening.csv",
            index=False,
        )
        analyser.separator_motifs_df_.to_csv(
            output_dir / "separator_motifs.csv",
            index=False,
        )
        analyser.separator_neighbours_df_.to_csv(
            output_dir / "separator_neighbours.csv",
            index=False,
        )
        analyser.separation_df_.to_csv(
            output_dir / "separation_sets.csv",
            index=False,
        )
        analyser.edge_diagnostics_df_.to_csv(
            output_dir / "edge_diagnostics.csv",
            index=False,
        )
        results["local_edges"].to_csv(
            output_dir / "local_edges.csv",
            index=False,
        )

    return results



# ============================================================
# Variable selection
# ============================================================

def is_candidate_process_variable(column: str) -> bool:
    """Return True when a column is eligible for the analysis."""
    name = str(column)
    lower = name.lower()

    if "€" in name:
        return False

    return not any(
        token in lower
        for token in EXCLUDED_NAME_PARTS
    )


def build_within_product_variables(
    data: pd.DataFrame,
    output_var: str,
) -> list[str]:
    """
    Build the final analysis-variable list.

    Absolute basis weight and nominal grammage are excluded and replaced
    by delta_basis_weight.
    """
    control_vars = unique_in_order(
        column
        for column in data.columns
        if is_candidate_process_variable(column)
    )

    control_vars = [
        variable
        for variable in control_vars
        if variable not in EXCLUDED_EXACT
    ]

    return unique_in_order(
        [output_var]
        + control_vars
        + ["delta_basis_weight"]
    )


# ============================================================
# Within-grade transformation
# ============================================================

def make_within_grade_deviations(
    data: pd.DataFrame,
    variables: list[str],
    grade_var: str,
) -> pd.DataFrame:
    """
    Replace each analysis variable by its deviation from its grade mean:

        X_within = X - E[X | grade]

    Conversion to float64 is done before subtraction so that pandas does
    not attempt to write floating-point deviations into integer/string
    columns.
    """
    result = data.copy()

    numeric = (
        result[variables]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    grade_means = (
        numeric
        .groupby(result[grade_var])
        .transform("mean")
    )

    result[variables] = (
        numeric - grade_means
    )

    return result


# ============================================================
# Interactive graph
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
    Generate an interactive vis-network dependence graph.

    Initial view
    ------------
    - Output and robust neighbours are visible.
    - Pure separators and blocked variables are hidden, preventing
      orphan nodes in the collapsed graph.
    - Robust variables that also act as separators remain clickable.

    Interaction
    -----------
    - Clicking a visible separator expands/collapses its blocked parents.
    - show_blocked_relations=True additionally displays dashed
      blocked-variable -> output relations while those variables are visible.
    """

    # ------------------------------------------------------------
    # Retrieve actual separator neighbourhoods
    # ------------------------------------------------------------

    if "separator_neighbours_df" in results:
        separator_neighbours = (
            results["separator_neighbours_df"].copy()
        )
    else:
        analyser = results.get("analyser")

        if (
            analyser is None
            or not hasattr(analyser, "separator_neighbours_df_")
            or analyser.separator_neighbours_df_ is None
        ):
            raise KeyError(
                "No separator-neighbour results are available. "
                "Re-run run_dependence_analysis() with the updated "
                "ConditionalDependenceSkeleton."
            )

        separator_neighbours = (
            analyser.separator_neighbours_df_.copy()
        )

    # Expansion must use only genuine retained Z--X relations.
    separator_neighbours = separator_neighbours.loc[
        separator_neighbours["retained"].fillna(False)
    ].copy()

    # Adapt the new table to the field names expected by the existing
    # interactive graph code. These are now ACTUAL Z--X relationships,
    # not X--SCT blocking motifs.
    motifs = separator_neighbours.rename(
        columns={
            "neighbour": "associated_variable",
            "best_conditioner": "blocking_conditioner",
        }
    ).copy()

    # `attenuation` is retained only for tooltip compatibility.
    # Here it describes attenuation of the Z--X relation under its
    # strongest single-variable conditioning test.
    motifs["attenuation"] = (
        1.0
        - motifs["conditional_abs_val"]
        / motifs["marginal_abs_val"]
    )

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
            "separator_motifs_df is missing columns: "
            f"{sorted(missing)}"
        )

    motifs = (
        motifs
        .dropna(subset=["associated_variable", "separator"])
        .copy()
    )

    motifs["associated_variable"] = (
        motifs["associated_variable"].astype(str)
    )
    motifs["separator"] = motifs["separator"].astype(str)

    separators = sorted(
        motifs["separator"].drop_duplicates().tolist()
    )
    associated = sorted(
        motifs["associated_variable"].drop_duplicates().tolist()
    )

    robust_set = set(robust)
    separator_set = set(separators)
    associated_set = set(associated)

    all_names = list(
        dict.fromkeys(
            [output_var]
            + robust
            + separators
            + associated
        )
    )

    node_id = {
        name: f"n{i}"
        for i, name in enumerate(all_names)
    }

    # ------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------

    nodes = []

    for name in all_names:

        if name == output_var:
            nodes.append({
                "id": node_id[name],
                "label": output_label,
                "title": f"Output\n{output_var}",
                "group": "output",
                "level": 3,
                "shape": "ellipse",
                "size": 32,
                "hidden": False,
            })
            continue

        if name in robust_set:
            title = (
                "Robust SCT neighbour\n"
                + name
            )

            if name in separator_set:
                n_children = int(
                    motifs["separator"].eq(name).sum()
                )

                title += (
                    "\n\nAlso acts as separator\n"
                    f"{n_children} blocked variable"
                    f"{'' if n_children == 1 else 's'}"
                    "\n\nClick to expand/collapse"
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

        if name in separator_set:
            n_children = int(
                motifs["separator"].eq(name).sum()
            )

            title = (
                "Separating variable\n"
                + name
                + "\n\n"
                + f"{n_children} blocked variable"
                + ("" if n_children == 1 else "s")
            )

            # Pure separators start hidden so the collapsed graph
            # does not contain orphan nodes.
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

        if name in associated_set:
            row = motifs.loc[
                motifs["associated_variable"].eq(name)
            ].iloc[0]

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

            # Plain-text tooltip: vis-network displays this much more
            # reliably than embedded HTML.
            title = (
                "Separator neighbour\n"
                + name
                + "\n\n"
                + "Related separator:\n"
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

    ids = [node["id"] for node in nodes]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate node IDs detected.")

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

    edges = []
    edge_counter = 0

    # Robust neighbour -> output.
    for name in robust:
        if name == output_var or name not in node_id:
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

    # Blocked variable -> separator.
    # These edges are revealed with their corresponding branch.
    for _, row in motifs.iterrows():
        x = str(row["associated_variable"])
        z = str(row["separator"])

        if x not in node_id or z not in node_id:
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

    # Optional dashed blocked-variable -> output relation.
    # It follows the visibility of the blocked variable.
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

    # ------------------------------------------------------------
    # Expansion maps
    # ------------------------------------------------------------

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

    for separator_id, children in children_by_separator.items():
        for child_id in children:
            separators_by_child.setdefault(child_id, []).append(
                separator_id
            )

    collapsible_nodes = {
        node_id[name]
        for name in associated
        if (
            name not in robust_set
            and name not in separator_set
            and name != output_var
        )
    }

    pure_separator_nodes = {
        node_id[name]
        for name in separators
        if (
            name not in robust_set
            and name != output_var
        )
    }

    # ------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
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

    # ------------------------------------------------------------
    # HTML
    #
    # This is deliberately not an f-string because the JavaScript
    # contains many braces.
    # ------------------------------------------------------------

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
    <button id="collapseAll">Collapse all</button>
    <button id="expandAll">Expand all</button>
    <button id="fitGraph">Fit</button>
    <span id="helpText" class="help"></span>
    <span id="status"></span>
</div>

<div id="network"></div>

<script>

const nodes = new vis.DataSet(__NODES_JSON__);
const edges = new vis.DataSet(__EDGES_JSON__);

const childrenBySeparator = __CHILDREN_JSON__;
const separatorsByChild = __SEPARATORS_BY_CHILD_JSON__;

const collapsibleNodes = new Set(
    __COLLAPSIBLE_NODES_JSON__
);

const pureSeparatorNodes = new Set(
    __PURE_SEPARATOR_NODES_JSON__
);

const showBlockedRelations =
    __SHOW_BLOCKED_RELATIONS__;

const expandedSeparators = new Set();

const container =
    document.getElementById("network");

const statusElement =
    document.getElementById("status");

const helpElement =
    document.getElementById("helpText");

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


function childStillNeeded(childId) {
    const parentSeparators =
        separatorsByChild[childId] || [];

    for (const separatorId of parentSeparators) {
        if (expandedSeparators.has(separatorId)) {
            return true;
        }
    }

    return false;
}


function updateBlockedEdge(childId) {
    if (!showBlockedRelations) {
        return;
    }

    const node = nodes.get(childId);

    if (!node) {
        return;
    }

    const blockedEdges = edges.get({
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


function revealSeparator(separatorId) {
    if (pureSeparatorNodes.has(separatorId)) {
        nodes.update({
            id: separatorId,
            hidden: false
        });
    }
}


function hideSeparatorIfUnused(separatorId) {
    if (!pureSeparatorNodes.has(separatorId)) {
        return;
    }

    if (expandedSeparators.has(separatorId)) {
        return;
    }

    nodes.update({
        id: separatorId,
        hidden: true
    });
}


function expand(separatorId) {
    const children =
        childrenBySeparator[separatorId];

    if (!children) {
        return;
    }

    revealSeparator(separatorId);
    expandedSeparators.add(separatorId);

    children.forEach(
        childId => {
            if (collapsibleNodes.has(childId)) {
                nodes.update({
                    id: childId,
                    hidden: false
                });
            }

            const motifEdges = edges.get({
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

            updateBlockedEdge(childId);
        }
    );

    updateStatus();
}


function collapse(separatorId) {
    const children =
        childrenBySeparator[separatorId];

    if (!children) {
        return;
    }

    expandedSeparators.delete(separatorId);

    children.forEach(
        childId => {
            const motifEdges = edges.get({
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
                collapsibleNodes.has(childId)
                &&
                !childStillNeeded(childId)
            ) {
                nodes.update({
                    id: childId,
                    hidden: true
                });
            }

            updateBlockedEdge(childId);
        }
    );

    hideSeparatorIfUnused(separatorId);
    updateStatus();
}


function toggleSeparator(separatorId) {
    if (expandedSeparators.has(separatorId)) {
        collapse(separatorId);
    } else {
        expand(separatorId);
    }
}


function updateStatus() {
    statusElement.textContent =
        "Expanded separators: "
        + expandedSeparators.size;
}


network.on(
    "click",
    function(params) {
        if (params.nodes.length !== 1) {
            return;
        }

        const id = params.nodes[0];

        if (
            !Object.prototype.hasOwnProperty.call(
                childrenBySeparator,
                id
            )
        ) {
            return;
        }

        toggleSeparator(id);
    }
);


document
    .getElementById("collapseAll")
    .addEventListener(
        "click",
        function() {
            const expanded =
                Array.from(expandedSeparators);

            expanded.forEach(
                separatorId => {
                    collapse(separatorId);
                }
            );

            network.fit({
                animation: true
            });
        }
    );


document
    .getElementById("expandAll")
    .addEventListener(
        "click",
        function() {
            Object.keys(
                childrenBySeparator
            ).forEach(
                separatorId => {
                    expand(separatorId);
                }
            );

            network.fit({
                animation: true
            });
        }
    );


document
    .getElementById("fitGraph")
    .addEventListener(
        "click",
        function() {
            network.fit({
                animation: true
            });
        }
    );


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

    html = (
        html
        .replace("__HEIGHT__", str(height))
        .replace("__NODES_JSON__", nodes_json)
        .replace("__EDGES_JSON__", edges_json)
        .replace("__CHILDREN_JSON__", children_json)
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

    output_file = Path(output_file)

    output_file.write_text(
        html,
        encoding="utf-8",
    )

    print(f"Interactive graph written to: {output_file.resolve()}")
    print(f"Nodes: {len(nodes)}")
    print(f"Unique node IDs: {len({node['id'] for node in nodes})}")
    print(f"Edges: {len(edges)}")
    print(f"Robust neighbours: {len(robust)}")
    print(f"Separators: {len(separators)}")
    print(
        "Pure separators initially hidden: "
        f"{len(pure_separator_nodes)}"
    )
    print(f"Associated variables: {len(associated)}")
    print(f"Motifs: {len(motifs)}")
    print(
        "Show dashed target relations: "
        f"{show_blocked_relations}"
    )

    return output_file


# ============================================================
# Main analysis
# ============================================================

def main():
    output_var = Y_COLUMN

    # ------------------------------------------------------------
    # 1. Load raw turnup data
    # ------------------------------------------------------------

    turnup_data = pd.read_parquet(DATA_PATH)

    # ------------------------------------------------------------
    # 2. Project feature engineering
    # ------------------------------------------------------------

    turnup_data = _feature_engineering(
        turnup_data,
        setpoint_df,
        steam_null=False,
        clip=True,
    )

    # ------------------------------------------------------------
    # 3. Time index
    # ------------------------------------------------------------

    if "Wedge_Time" not in turnup_data.columns:
        raise KeyError(
            "'Wedge_Time' is required in the turnup dataframe."
        )

    turnup_data = (
        turnup_data
        .set_index("Wedge_Time")
        .sort_index()
    )

    # ------------------------------------------------------------
    # 4. Delta basis weight
    #
    # Use ORIGINAL values, before within-grade centring and EWM.
    # ------------------------------------------------------------

    required_for_delta = {
        "Current_basis_weight",
        "grammage",
    }

    missing = (
        required_for_delta
        - set(turnup_data.columns)
    )

    if missing:
        raise KeyError(
            "Cannot create delta_basis_weight. "
            f"Missing columns: {sorted(missing)}"
        )

    turnup_data["delta_basis_weight"] = (
        pd.to_numeric(
            turnup_data["Current_basis_weight"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            turnup_data["grammage"],
            errors="coerce",
        )
    )

    # ------------------------------------------------------------
    # 5. Final analysis-variable list
    #
    # Absolute basis weight and grammage are removed and replaced
    # by delta_basis_weight.
    # ------------------------------------------------------------

    within_product_vars = (
        build_within_product_variables(
            turnup_data,
            output_var=output_var,
        )
    )

    missing_analysis_vars = [
        variable
        for variable in within_product_vars
        if variable not in turnup_data.columns
    ]

    if missing_analysis_vars:
        raise KeyError(
            "Analysis variables missing from dataframe: "
            f"{missing_analysis_vars}"
        )

    if GRADE_VAR not in turnup_data.columns:
        raise KeyError(
            f"Grade variable '{GRADE_VAR}' is missing."
        )

    print(
        "Variables entering within-grade analysis:",
        len(within_product_vars),
    )

    # ------------------------------------------------------------
    # 6. Within-grade deviations
    #
    # X_within = X - mean(X | grade)
    #
    # This is done BEFORE EWM filtering.
    # ------------------------------------------------------------

    turnup_within = (
        make_within_grade_deviations(
            turnup_data,
            variables=within_product_vars,
            grade_var=GRADE_VAR,
        )
    )

    # ------------------------------------------------------------
    # 7. Sanity check
    #
    # The grade-wise means should be approximately zero.
    # Printing only the largest absolute residual means keeps the
    # output useful even with hundreds of variables.
    # ------------------------------------------------------------

    residual_grade_means = (
        turnup_within
        .groupby(GRADE_VAR)[within_product_vars]
        .mean()
        .abs()
    )

    max_residuals = (
        residual_grade_means
        .max(axis=0)
        .sort_values(ascending=False)
        .head(10)
    )

    print()
    print(
        "Largest absolute within-grade means "
        "(top 10 variables):"
    )
    print(
        max_residuals.round(8)
    )

    # ------------------------------------------------------------
    # 8. Segment-wise EWM filtering
    # ------------------------------------------------------------

    filtered_within = ewm_filter_by_segment(
        turnup_within,
        columns=within_product_vars,
        halflife=EWM_HALFLIFE,
    )

    # ------------------------------------------------------------
    # 9. Final numeric matrix
    # ------------------------------------------------------------

    analysis_df = (
        filtered_within[within_product_vars]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .astype("float64")
    )

    if analysis_df.empty:
        raise ValueError(
            "No complete rows remain after numeric conversion "
            "and dropna()."
        )

    print()
    print(
        "Rows entering dependence analysis:",
        len(analysis_df),
    )
    print(
        "Variables entering dependence analysis:",
        analysis_df.shape[1],
    )

    data_array = analysis_df.to_numpy(
        dtype=float,
        copy=False,
    )

    dataframe_within = pp.DataFrame(
        data_array,
        var_names=within_product_vars,
    )

    # ------------------------------------------------------------
    # 10. Conditional-dependence analysis
    #
    # No max_target_neighbours cap: target screening considers the
    # complete candidate pool according to the current implementation.
    # ------------------------------------------------------------

    results_within = run_dependence_analysis(
        dataframe=dataframe_within,
        output_var=output_var,
        max_cond_dim=MAX_COND_DIM,
        min_abs_val=MIN_ABS_VAL,
    )

    # ------------------------------------------------------------
    # 11. Interactive graph
    # ------------------------------------------------------------

    interactive_file = (
        plot_interactive_dependency_graph(
            results_within,
            output_var=output_var,
            output_label="SCT CD",
            output_file=INTERACTIVE_OUTPUT,
            height=INTERACTIVE_HEIGHT,
            show_blocked_relations=SHOW_BLOCKED_RELATIONS,
        )
    )

    print()
    print(
        "Analysis complete."
    )
    print(
        "Interactive graph:",
        interactive_file.resolve(),
    )

    return results_within, interactive_file


if __name__ == "__main__":
    results_within, interactive_file = main()
