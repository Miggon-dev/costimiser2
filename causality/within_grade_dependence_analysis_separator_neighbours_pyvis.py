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

INTERACTIVE_OUTPUT = "separator_neighbours_graph_pyvis.html"
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
    "specific"
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


# ============================================================
# Runtime wiring (standalone execution)
#
# The source analysis module expects these project objects to exist.
# They are provided here so the script runs on its own, matching the
# sibling scripts in this folder.
# ============================================================

import sys as _sys

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from utility import _feature_engineering, setpoint_df  # noqa: E402

Y_COLUMN = DEFAULT_OUTPUT_VAR
DATA_PATH = "../data/costimiser_turnup.parquet"



# ============================================================
# Interactive graph (pyvis, physics-based, blue nodes)
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
    Render the separator-neighbour dependence graph with pyvis.

    Same data and interaction semantics as the original vis-network
    renderer (built from separator_neighbours_df, filtered to retained
    Z--X relations), but drawn with a balanced, draggable Barnes-Hut force
    layout. All nodes are shades of BLUE, distinguished by lightness and
    shape:

        output      : dark-blue star
        robust      : medium-blue dot
        separator   : blue diamond
        neighbour   : light-blue small dot

    Clicking a separator (diamond) expands / collapses the variables
    genuinely related to it.
    """
    from pyvis.network import Network

    # --------------------------------------------------------
    # Retrieve actual separator neighbourhoods
    # --------------------------------------------------------
    if "separator_neighbours_df" in results:
        separator_neighbours = results["separator_neighbours_df"].copy()
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
        separator_neighbours = analyser.separator_neighbours_df_.copy()

    # Expansion must use only genuine retained Z--X relations.
    separator_neighbours = separator_neighbours.loc[
        separator_neighbours["retained"].fillna(False)
    ].copy()

    motifs = separator_neighbours.rename(
        columns={
            "neighbour": "associated_variable",
            "best_conditioner": "blocking_conditioner",
        }
    ).copy()

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
            f"separator neighbour table is missing columns: {sorted(missing)}"
        )

    motifs = motifs.dropna(
        subset=["associated_variable", "separator"]
    ).copy()
    motifs["associated_variable"] = motifs["associated_variable"].astype(str)
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

    # --------------------------------------------------------
    # Blue palette (varied by role via lightness)
    # --------------------------------------------------------
    blue = {
        "output": {"bg": "#0d3b66", "bd": "#08284a",
                    "hbg": "#12508c", "hbd": "#08284a"},
        "robust": {"bg": "#2f7fd1", "bd": "#1f5f9e",
                    "hbg": "#4f9ae0", "hbd": "#1f5f9e"},
        "separator": {"bg": "#5aa9e6", "bd": "#3a86c8",
                       "hbg": "#7dbef0", "hbd": "#3a86c8"},
        "associated": {"bg": "#a9d3f5", "bd": "#7fb8e6",
                        "hbg": "#c3e0fa", "hbd": "#7fb8e6"},
    }

    def color(group):
        s = blue[group]
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

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------
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
                "font": {"size": 20, "face": "Arial", "color": "#0d3b66"},
                "hidden": False,
            })
            continue

        if name in robust_set:
            inner = f"<b>Robust SCT neighbour</b><br>{name}"
            if name in separator_set:
                n_children = int(motifs["separator"].eq(name).sum())
                inner += (
                    "<br><br><i>Also acts as separator</i><br>"
                    f"{n_children} related variable"
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
                "font": {"size": 15, "face": "Arial", "color": "#183a5a"},
                "hidden": False,
            })
            continue

        if name in separator_set:
            n_children = int(motifs["separator"].eq(name).sum())
            inner = (
                f"<b>Separating variable</b><br>{name}<br><br>"
                f"{n_children} related variable"
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
                "font": {"size": 13, "face": "Arial", "color": "#183a5a"},
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
                f"<b>Separator neighbour</b><br>{name}"
                "<br><br><u>Related separator</u><br>"
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
                "font": {"size": 11, "face": "Arial", "color": "#3a5871"},
                "hidden": True,
            })

    if len({n["id"] for n in nodes}) != len(nodes):
        raise RuntimeError("Duplicate node IDs detected.")

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------
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
            "color": {"color": "#2f7fd1", "highlight": "#1f5f9e"},
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
            "color": {"color": "#5aa9e6", "highlight": "#3a86c8"},
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
                "color": {"color": "#9fc4e8", "highlight": "#7fb8e6"},
                "hidden": True,
            })

    # --------------------------------------------------------
    # Expansion maps
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # pyvis network
    # --------------------------------------------------------
    net = Network(
        height=str(height),
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#183a5a",
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

    # pyvis' add_node/add_edge drop dict-valued kwargs (color/font); write
    # the styling dicts onto the records pyvis serialises verbatim.
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
        "collapse the variables genuinely related to it. Solid = robust "
        "relation, dashed = relation to output."
        if show_blocked_relations
        else "Drag nodes to rearrange. Click a separator (diamond) to "
        "expand / collapse its related variables."
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
        "style=\"background:#0d3b66\"></span>Output</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot\" "
        "style=\"background:#2f7fd1\"></span>Robust neighbour</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot cd-diamond\" "
        "style=\"background:#5aa9e6\"></span>Separator</span>"
        "<span class=\"cd-chip\"><span class=\"cd-dot cd-small\" "
        "style=\"background:#a9d3f5\"></span>Neighbour</span>"
        "</span><span id=\"cd-status\"></span></div>"
        "<div id=\"cd-help\">" + help_text + "</div>"
    )

    controls_css = """
<style>
  body { margin: 0; font-family: Arial, Helvetica, sans-serif;
         background: #f4f7fb; }
  #cd-controls { display: flex; align-items: center; flex-wrap: wrap;
    gap: 8px; padding: 10px 16px; background: #ffffff;
    border-bottom: 1px solid #dde5ee;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  #cd-controls .cd-title { font-weight: 600; font-size: 15px;
    color: #0d3b66; margin-right: 8px; }
  #cd-controls button { padding: 6px 12px; font-size: 13px;
    border: 1px solid #c4d3e4; border-radius: 6px; background: #fff;
    color: #1f5f9e; cursor: pointer; transition: background .15s; }
  #cd-controls button:hover { background: #eef4fb; }
  #cd-controls button.cd-active { background: #e3eefb;
    border-color: #2f7fd1; color: #0d3b66; }
  #cd-legend { display: flex; align-items: center; gap: 14px;
    margin-left: 12px; font-size: 12px; color: #456; }
  .cd-chip { display: inline-flex; align-items: center; gap: 6px; }
  .cd-dot { width: 13px; height: 13px; border-radius: 50%;
    display: inline-block; border: 1px solid rgba(0,0,0,0.2); }
  .cd-dot.cd-small { width: 9px; height: 9px; }
  .cd-dot.cd-diamond { border-radius: 2px; transform: rotate(45deg); }
  #cd-status { margin-left: auto; font-size: 12px; color: #789; }
  #cd-help { padding: 7px 16px; font-size: 12px; color: #566;
    background: #fbfdff; border-bottom: 1px solid #eef3f9; }
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
            json.dumps(children_by_separator, ensure_ascii=False),
        )
        .replace(
            "__SEPARATORS_BY_CHILD_JSON__",
            json.dumps(separators_by_child, ensure_ascii=False),
        )
        .replace(
            "__COLLAPSIBLE_NODES_JSON__",
            json.dumps(collapsible_nodes, ensure_ascii=False),
        )
        .replace(
            "__PURE_SEPARATOR_NODES_JSON__",
            json.dumps(pure_separator_nodes, ensure_ascii=False),
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
    print("Robust neighbours:", len(robust))
    print("Separators:", len(separators))
    print("Separator neighbours:", len(associated))

    return output_file



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
