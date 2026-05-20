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
RECOMMENDATION_ACTION_LIMIT = "all"

RECOMMENDATION_USE_OPTIMIZER = True
# False = current behavior, heuristic ±5%
# True = optimizer proposes final values

RECOMMENDATION_OPTIMIZER_LOWER_Q = 0.05
RECOMMENDATION_OPTIMIZER_UPPER_Q = 0.95
RECOMMENDATION_OPTIMIZER_JOINT_QUANTILE = 0.95

RECOMMENDATION_USE_MANUAL_ACTIONABLE_INPUTS = True

RECOMMENDATION_OPTIMIZER_MAX_INTERVENTIONS = 1
RECOMMENDATION_OPTIMIZER_SELECTION_MODE = "greedy"

RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS_BY_TARGET = {
    "steam": [
        'Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
        'Current_basis_weight',
        'Current_reel_moisture_average(reel)',
        'Moisture_after_SpeedSizer',
        'SpeedSizer_Linepressure_DS',
        'SpeedSizer_Linepressure_FS',
        'Draw_WS-PS',
        'Linepressure_1st_press_FS__bar_',
        'Linepressure_2nd_press_FS__bar_',
        'Linepressure_1st_press_DS__bar_',
        'Linepressure_2nd_press_DS__bar_',
        'Linepressure_shoe_press__bar_',
        'pH_measurement_white_water_B41',
        'Headbox_consistency',
        'Lip_settings',
        'Jet/wire_ratio',
        'Retention_Aid_mass_flow__g/T_',
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'Thick_Stock_Consistency__%_',
        'Short_fibre_flow',
        'Long_fibre_flow'
    ],

    "SCT CD": [
        'Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
        'Current_basis_weight',
        'Current_reel_moisture_average(reel)',
        'Moisture_after_SpeedSizer',
        # 'SpeedSizer_Linepressure_DS',
        # 'SpeedSizer_Linepressure_FS',
        # 'Draw_WS-PS',
        # 'Linepressure_1st_press_FS__bar_',
        # 'Linepressure_2nd_press_FS__bar_',
        # 'Linepressure_1st_press_DS__bar_',
        # 'Linepressure_2nd_press_DS__bar_',
        # 'Linepressure_shoe_press__bar_',
        # 'pH_measurement_white_water_B41',
        'Headbox_consistency',
        'Lip_settings',
        'Jet/wire_ratio',        
        'Short_fibre_flow',
        'Long_fibre_flow'
    ],

    "SCT MD": [
        'Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
        'Current_basis_weight',
        'Current_reel_moisture_average(reel)',
        'Moisture_after_SpeedSizer',
        # 'SpeedSizer_Linepressure_DS',
        # 'SpeedSizer_Linepressure_FS',
        # 'Draw_WS-PS',
        # 'Linepressure_1st_press_FS__bar_',
        # 'Linepressure_2nd_press_FS__bar_',
        # 'Linepressure_1st_press_DS__bar_',
        # 'Linepressure_2nd_press_DS__bar_',
        # 'Linepressure_shoe_press__bar_',
        # 'pH_measurement_white_water_B41',
        'Headbox_consistency',
        'Lip_settings',
        'Jet/wire_ratio',        
        'Short_fibre_flow',
        'Long_fibre_flow'
    ],

    "total": [
       'Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
        'Current_basis_weight',
        'Current_reel_moisture_average(reel)',
        'Moisture_after_SpeedSizer',
        'SpeedSizer_Linepressure_DS',
        'SpeedSizer_Linepressure_FS',
        'Draw_WS-PS',
        'Linepressure_1st_press_FS__bar_',
        'Linepressure_2nd_press_FS__bar_',
        'Linepressure_1st_press_DS__bar_',
        'Linepressure_2nd_press_DS__bar_',
        'Linepressure_shoe_press__bar_',
        'pH_measurement_white_water_B41',
        'Headbox_consistency',
        'Lip_settings',
        'Jet/wire_ratio',
        'Retention_Aid_mass_flow__g/T_',
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'Thick_Stock_Consistency__%_',
        'Short_fibre_flow',
        'Long_fibre_flow'
    ],
}

RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS_DEFAULT = [
    'Starch_uptake_by_paper_Top_Roll__g/m2_',
    'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
    'Current_basis_weight',
    'Current_reel_moisture_average(reel)',
    'Moisture_after_SpeedSizer',
    'SpeedSizer_Linepressure_DS',
    'SpeedSizer_Linepressure_FS',
    'Draw_WS-PS',
    'Linepressure_1st_press_FS__bar_',
    'Linepressure_2nd_press_FS__bar_',
    'Linepressure_1st_press_DS__bar_',
    'Linepressure_2nd_press_DS__bar_',
    'Linepressure_shoe_press__bar_',
    'pH_measurement_white_water_B41',
    'Headbox_consistency',
    'Lip_settings',
    'Jet/wire_ratio',
    'Retention_Aid_mass_flow__g/T_',
    'Bentonite_1_mass_flow__g/T_',
    'Bentonite_2_mass_flow__g/T_',
    'Thick_Stock_Consistency__%_',
    'Short_fibre_flow',
    'Long_fibre_flow'
    ]

RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS = [
    'Starch_uptake_by_paper_Top_Roll__g/m2_',
    'Starch_uptake_by_paper_Bottom_Roll__g/m2_',
    'Current_basis_weight',
    ]


RECOMMENDATION_INVARIANTS = [
    {
        "name": "keep_total_fibre_flow_constant",

        "variables": [
            "Short_fibre_flow",
            "Long_fibre_flow",
        ],

        "fn": lambda row, ref: (
            (
                row["Short_fibre_flow"]
                + row["Long_fibre_flow"]
            )
            -
            (
                ref["Short_fibre_flow"]
                + ref["Long_fibre_flow"]
            )
        ),

        "tolerance": 1e-6,

        # strong penalty
        "weight": 1e8,
    },
]