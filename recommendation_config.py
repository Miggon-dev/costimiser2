# ------------------------------------------------------------
# Recommendation feature selection settings
# ------------------------------------------------------------
ALLOWED_RECOMMENDATION_FEATURE_SOURCES = {"drivers", "model", "auto"}

RECOMMENDATION_FEATURE_SOURCE = "drivers"
# "drivers", "model", "auto"

# Percentage of ranked cost-driver variables to send/use.
# Example: 0.4 = top 40%.
# Use 1.0 for all.
RECOMMENDATION_COST_DRIVER_TOP_FRAC = 0.5

# Percentage of ranked SHAP variables to send/use.
# Example: 0.3 = top 30%.
# Use 1.0 for all.
RECOMMENDATION_SHAP_TOP_FRAC = 0.5

# Number of variables requested from RAG.
# Use None or "all" to request all candidate variables.
RECOMMENDATION_RAG_VARIABLE_LIMIT = "all"

# Number of final recommendation actions.
# "all" or None = all actionable / indirectly actionable RAG variables
# integer = max number of final actions
RECOMMENDATION_ACTION_LIMIT = 5