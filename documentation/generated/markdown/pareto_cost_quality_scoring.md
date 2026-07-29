# Cost-Quality Ranking using Multiobjective Optimisation

A normalization-free framework to identify efficient products: low cost, high quality, and strict handling of minimum quality specifications.

> **Recommended idea:** treat quality specification as a constraint, then rank products using Pareto dominance. This avoids forcing cost and quality into one arbitrary normalized scale.

## 1. Problem formulation

For each product i, you observe cost $c_i$ and quality $q_i$. The natural multiobjective problem is:

$$
minimize\ c_i
$$

$$
maximize\ q_i
$$

$$
subject\ to\ q_i \ge q_{min}
$$

This formulation keeps the original units. Cost is cost, quality is quality, and the minimum specification is not treated as something that can be compensated by low cost.

## 2. Feasibility first

Separate products into two groups before ranking:

| Group | Condition | Treatment |
| --- | --- | --- |
| Compliant | $q_i \ge q_{min}$ | Rank using low cost and high quality. |
| Non-compliant | $q_i < q_{min}$ | Place below compliant products; rank by quality shortfall and cost. |

## 3. Pareto dominance

For compliant products, product A dominates product B when A is no more expensive and no worse in quality, with at least one strict improvement:

$$
c_A \le c_B \quad and \quad q_A \ge q_B
$$

The Pareto-optimal products are those not dominated by any other product. They represent the efficient cost-quality trade-offs.

## 4. Example interpretation

| Product | Cost | Quality | Interpretation |
| --- | --- | --- | --- |
| A | 100 | 2.10 | Pareto candidate |
| B | 105 | 2.20 | Pareto candidate |
| C | 110 | 2.05 | Dominated by A: higher cost and lower quality |
| D | 95 | 1.90 | Below specification, therefore infeasible |

## 5. From Pareto set to ranking

Pareto optimisation gives a set of efficient products, not automatically a single best product. If you need an ordered ranking, use non-dominated sorting:

| Layer | Meaning | Priority |
| --- | --- | --- |
| Pareto front 1 | Products not dominated by any other compliant product. | Best group |
| Pareto front 2 | Products dominated only by products in front 1. | Second group |
| Pareto front 3+ | Progressively less efficient products. | Lower groups |

Within each Pareto front, apply a business tie-breaker: cost-first if quality above specification has limited value, or quality-margin-first if extra quality is valuable.

## 6. Non-compliant products

For products below specification, define the quality shortfall:

$$
s_i = q_{min} - q_i
$$

Then rank non-compliant products by minimizing both shortfall and cost:

$$
minimize\ s_i \quad and \quad minimize\ c_i
$$

This ensures the worst products are those with both large quality failure and high cost.

## 7. Recommended operational rule

| Step | Rule |
| --- | --- |
| 1 | Separate compliant and non-compliant products using $q_i \ge q_{min}$. |
| 2 | For compliant products, compute Pareto fronts using cost minimisation and quality maximisation. |
| 3 | Rank front 1 above front 2, front 2 above front 3, and so on. |
| 4 | Within each front, use a clear tie-breaker such as lower cost first. |
| 5 | Place all non-compliant products below compliant products. |
| 6 | For non-compliant products, rank by quality shortfall first, then cost. |

> **Important note:** This gives an ordinal decision ranking, not a smooth economic score. That is often an advantage: it avoids arbitrary weights and makes clear which products are objectively dominated.
