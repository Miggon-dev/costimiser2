# Product Cost-Quality Scoring Framework

A practical method to rank products so that low cost and high quality receive the best scores, while products below specification - especially expensive ones - receive the lowest scores.

> **Recommended design** Combine a normalized cost score with a capped quality score, then apply a strong penalty whenever quality falls below the minimum specification.

## 1. Variables

| Symbol | Meaning |
| --- | --- |
| $c_i$ | Cost of product i |
| $q_i$ | Quality of product i |
| $q_{min}$ | Minimum acceptable quality specification |
| $q_{target}$ | Quality level above which extra quality receives no additional score |
| $c_{min}, c_{max}$ | Reference minimum and maximum cost values |

## 2. Cost score

Normalize cost so that the cheapest product receives 1 and the most expensive receives 0:

$$
CostScore_i = \frac{c_{max} - c_i}{c_{max} - c_{min}}
$$

Lower cost therefore always improves the score.

## 3. Quality score

Reward quality above specification, but cap the benefit at a meaningful target. This prevents excessive quality or overprocessing from being rewarded indefinitely.

$$
QualityScore_i = clip\left(\frac{q_i - q_{min}}{q_{target} - q_{min}}, 0, 1\right)
$$

| Quality position | Quality score |
| --- | --- |
| Below $q_{min}$ | 0 before applying the out-of-specification penalty |
| At $q_{min}$ | 0 |
| Between $q_{min}$ and $q_{target}$ | Increases linearly from 0 to 1 |
| At or above $q_{target}$ | 1 |

## 4. Out-of-specification penalty

Meeting specification is normally non-negotiable. A separate penalty ensures that a cheap but non-compliant product does not outrank a compliant product.

$$
Penalty_i = 0, \quad \text{if } q_i \ge q_{min}
$$

$$
Penalty_i = P_0 + P_1\frac{q_{min} - q_i}{q_{min}}, \quad \text{if } q_i < q_{min}
$$

$P_0$ is a fixed penalty for any specification failure. $P_1$ increases the penalty according to the severity of the quality shortfall.

## 5. Final recommended score

Using equal weights as a starting point:

> **$Score_i = 50 \times CostScore_i + 50 \times QualityScore_i - Penalty_i$**

The weights can be adjusted. For example, use 60% quality and 40% cost when quality should dominate, or 60% cost and 40% quality when all evaluated products are already reliably compliant.

## 6. Expected ranking behaviour

| Product profile | Expected score |
| --- | --- |
| High quality, low cost | Highest |
| Acceptable quality, low cost | High |
| High quality, high cost | Intermediate |
| Below-specification quality, low cost | Low |
| Below-specification quality, high cost | Lowest |

## 7. Suggested starting parameters

| Parameter | Starting recommendation |
| --- | --- |
| Cost weight | 50 |
| Quality weight | 50 |
| $q_{target}$ | A meaningful quality target above specification, not the observed maximum |
| $P_0$ | 50 points, making any specification breach immediately visible |
| $P_1$ | 50 points, distinguishing minor from severe quality failures |
| Cost reference range | Prefer robust limits such as the 5th and 95th percentiles rather than raw extremes |

## 8. Important implementation note

> **Important implementation note:** Use stable reference values for normalization. If $c_{min}$ and $c_{max}$ are recalculated from every small batch, the same product may receive a different score depending on the comparison set. For operational use, define the cost limits from a sufficiently long historical period and review them periodically.

**Interpretation:** The resulting score is a ranking index, not a physical measurement. Its parameters should be validated against business priorities and known examples of good and bad products.
