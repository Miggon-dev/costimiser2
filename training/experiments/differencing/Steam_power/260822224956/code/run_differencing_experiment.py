"""
Differencing experiment: delta_y = f(delta_X)

Models y_t - y_{t-k} = f(X_t - X_{t-k}) with a time-based lag.
EWM is always applied to both X and y before differencing.
The lag is chosen by TimeSeriesSplit CV on train only. Test is scored once.

Usage:
    python run_differencing_experiment.py --y_column "Steam__kWh/T_" \
        --data_path ../data/costimier_turnup.parquet --model ridge --lag_days 1

Parameters:
    --y_column      Target column name
    --data_path     Path to parquet data
    --model         ridge | splines | realmlp
    --lag_days      Differencing lag in days (if omitted, sweeps and picks best)
    --spline_knots  Number of spline knots (default 6, only for --model splines)
    --spline_degree Spline degree (default 3, only for --model splines)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
import shutil
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import CONTROL_VARS
from data_cleaning import unique_in_order, ordered_difference, ordered_intersection
from preprocessing import make_prep_pip, build_pre_estimator
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df, ewm_reset


# =============================================================================
# Configuration
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam_power":["steam_pressure", "exha_mois", "inlet_temp", "vacuum", "linepressure","fabric_tension","gas_decu"],
    "Electrical_power_MW": ["speedsizer_linepressure", "linepressure"],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": ["speedsizer_linepressure", "linepressure","starch2"],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": ["speedsizer_linepressure", "linepressure","starch1"],
    "MBS_SCT_CD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_SCT_MD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_Burst": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_CMT30": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],

}

BLACK_LIST = {
    "Steam_power": [
        "Current_basis_weight", 
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", 
        "CO2_mass_flow__g/T_",
    ],
    "Electrical_power_MW": [
        "Current_basis_weight", 
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", 
        "CO2_mass_flow__g/T_",
    ],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", "CO2_mass_flow__g/T_",
    ],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": [
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", "CO2_mass_flow__g/T_",
    ],
    "MBS_SCT_CD": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
    "MBS_SCT_MD": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
     "MBS_Burst": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
     "MBS_CMT30": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
}

CREATED_VARIABLE_CANDIDATES = {
    "Steam_power": [
        "Water_flow_Predryer",
        "Water_flow_Afterdryer_input",
        "Water_flow_Afterdryer_output",
        "dewatering",
        "fibre_short/long",
    ],
    "Electrical_power_MW": [
        "dewatering",
        "Fibre__g/m2_",
    ],
     "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        "delta_basis_weight",
        "Water_flow_Predryer",            
        "Water_flow",                               
        "inv_Rod_pressure_Top_Roll",
        "square_Rod_pressure_Top_Roll",
     ],
     "Starch_uptake_by_paper_Bottom_Roll__g/m2_":[
        "delta_basis_weight",
        "Water_flow_Predryer",            
        "Water_flow",                               
        "inv_Rod_Pressure_Bottom_Roll",
        "square_Rod_Pressure_Bottom_Roll",
     ],
     "MBS_SCT_CD": [
        "delta_basis_weight",
        "Starch_uptake__g/m2_",
        "Water_flow_Predryer",
        "Water_flow_Afterdryer",
        "Water_flow",
        "flow_diluted_starch",
        "Fibre__g/m2_",
        "Water_flow_Afterdryer_input", 
        "Water_flow_Afterdryer_output",
        "dewatering",
        "fibre_short/long",
        "temperature_starch_working_tank"

     ],
}

GRAMMAGES = [115, 120, 100, 90, 85, 110]
TEST_SIZE = 0.20
ALPHAS = np.logspace(0, 3, 20)
DEFAULT_LAG_SWEEP = [0.25, 0.5, 1, 2, 4, 7, 14]

# RealMLP hyperparameters
REALMLP_HIDDEN_SIZES = [64, 64]
REALMLP_LR = 0.0001
REALMLP_BATCH_SIZE = 32
REALMLP_N_EPOCHS = 256
REALMLP_N_REFIT = 5
REALMLP_VAL_FRACTION = 0.20


# =============================================================================
# Helpers
# =============================================================================

def to_hours(index):
    t = pd.DatetimeIndex(index).asi8.astype(np.float64)
    return (t - t[0]) / 3.6e12


def lag_pairs(t_hours, lag_hours, tolerance=0.35):
    """Find (later, earlier) index pairs separated by ~lag_hours."""
    target = t_hours - lag_hours
    j = np.searchsorted(t_hours, target)
    j = np.clip(j, 0, len(t_hours) - 1)
    j_alt = np.clip(j - 1, 0, len(t_hours) - 1)
    better = np.abs(t_hours[j_alt] - target) < np.abs(t_hours[j] - target)
    j = np.where(better, j_alt, j)
    realised = t_hours - t_hours[j]
    ok = (j < np.arange(len(t_hours))) & (np.abs(realised - lag_hours) <= tolerance * lag_hours)
    later = np.flatnonzero(ok)
    return later, j[later]


def build_ridge_model():
    """Ridge with TimeSeriesSplit CV for alpha."""
    return GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        param_grid={"ridge__alpha": ALPHAS.tolist()},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        refit=True,
    )


def build_splines_ridge_model(n_features, n_knots=6, degree=3):
    """SplineTransformer + Ridge with TimeSeriesSplit CV for alpha."""
    return GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("splines", SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)),
            ("ridge", Ridge()),
        ]),
        param_grid={"ridge__alpha": ALPHAS.tolist()},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        refit=True,
    )


def build_realmlp_model(dX_fit, dy_fit, dX_val, dy_val):
    """RealMLP with chronological early stopping. Returns fitted model."""
    from pytabkit import RealMLP_TD_Regressor

    scaler = StandardScaler()
    dX_fit_s = scaler.fit_transform(dX_fit)
    dX_val_s = scaler.transform(dX_val)

    mlp = RealMLP_TD_Regressor(
        device="cpu",
        random_state=42,
        verbosity=2,
        n_cv=1,
        val_fraction=0.0,
        n_epochs=REALMLP_N_EPOCHS,
        hidden_sizes=REALMLP_HIDDEN_SIZES,
        use_early_stopping=True,
        n_refit=REALMLP_N_REFIT,
        lr=REALMLP_LR,
        batch_size=REALMLP_BATCH_SIZE,
    )
    mlp.fit(dX_fit_s, dy_fit, X_val=dX_val_s, y_val=dy_val)
    return mlp, scaler


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Differencing experiment")
    parser.add_argument("--y_column", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model", choices=["ridge", "splines", "realmlp"], default="ridge")
    parser.add_argument("--spline_knots", type=int, default=6)
    parser.add_argument("--spline_degree", type=int, default=3)
    args = parser.parse_args()

    Y_COLUMN = args.y_column
    DATA_PATH = args.data_path
    MODEL = args.model

    start_time = datetime.now()
    out_dir = Path("experiments") / "differencing" / Y_COLUMN.replace("/", "_") / start_time.strftime("%y%m%d%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # =========================================================================
    # Data Loading & Filtering
    # =========================================================================
    print("\n--- Data Loading & Filtering ---")
    prep_pip, _ = make_prep_pip(prefixes=PIPELINE_PREFIXES[Y_COLUMN])
    turnup_data = pd.read_parquet(DATA_PATH)

    
    turnup_data["Steam_power"] = turnup_data["Steam_flow_from_power_plant_to_PM"] + turnup_data["Waste_steam_flow"]
    turnup_data["Electrical_power_MW"] = (
        turnup_data['Summe_Energieverbrauch_APA_in_kW'] + 
        turnup_data['Mehrmotorenantrieb_Former__-1..10_MW_']*1000 +
        turnup_data['Mehrmotorenantrieb_Presse__-1..10_MW_']*1000 +
        turnup_data['VariSTEP_div._Hauben+Lueftung_Ventilatoren']*1000 +
        turnup_data['VariSTEP_div._Hauben+Lueftung_Ventilatoren']*1000 +
        turnup_data['Varisprint_660V__-10..10_MW_']*1000 +
        turnup_data['Varisprint_400V__-10..10_MW_']*1000 +
        turnup_data['Mehrmotorenantrieb_Trockenpartie__-1..10_MW_']*1000 +
        turnup_data['PM_400V__-1..10_MW_']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler2__-1..10_MW_']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler3__-1..10_MW_']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler3__-1..10_MW_~^0']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler2__-1..10_MW_~^1']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler1__-1..10_MW_~^0']*1000 +
        turnup_data['PM_660V_Festantriebe_Verteiler1__-1..10_MW_~^1']*1000 +
        turnup_data['MCC_Einzel-FU_660V__-10..10_MW_~^0']*1000 +
        turnup_data['MCC_Einzel-FU_660V__-10..10_MW_~^1']*1000 +
        turnup_data['Last_Antrieb_10_1.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_11_1.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_12_2.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_13_2.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_14_2.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_15_3.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_16_3.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_17_3.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_18_4.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_19_4.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_20_4.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_21_4.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_22_5.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_23_5.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_24_5.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_25_5.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_31_6.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_32_6.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_33_6.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_34_6.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_35_6.TG__0..100_kW_'] +
        turnup_data['Last_Antrieb_36_7.TG__0..200_kW_'] +
        turnup_data['Last_Antrieb_37_7.TG__0..200_kW_'] +
        turnup_data['Last_Antrieb_38_7.TG__0..200_kW_'] +
        turnup_data['Last_Antrieb_39_7.TG__0..200_kW_'] 
        
    )/1000

    ctl_vars = unique_in_order(v for v in CONTROL_VARS[Y_COLUMN] if "vacuum" not in v.lower())
    created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[Y_COLUMN], ctl_vars)

    turnup_data = _feature_engineering(turnup_data, setpoint_df, steam_null=False, clip=False)
    turnup_data = turnup_data.set_index("Wedge_Time").sort_index()

    # Time filters
    if Y_COLUMN== "Steam_power":
        #turnup_data = turnup_data[turnup_data.index >   "2026-3-1"]
        turnup_data = turnup_data[turnup_data.index > "2026-1-1"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-11 12:00") & (turnup_data.index < "2026-01-12 11:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-17 12:00") & (turnup_data.index < "2026-01-19 11:00"))]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]
        turnup_data = turnup_data[
            #(turnup_data["Condensate_energy_from_paper_plant_to_power_plant"].between(5, 10))
            #& 
            (turnup_data["DG4_Temperature_Inlet_Air"] > 100)
            & 
            (turnup_data["Vacuum_Zone_1_PickUp"] < -0.5)
        ]
    if Y_COLUMN== "Electrical_power_MW":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2025-11-15"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]

    if Y_COLUMN== "Starch_uptake_by_paper_Top_Roll__g/m2_":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2026-3-1"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]
    
    if Y_COLUMN== "Starch_uptake_by_paper_Bottom_Roll__g/m2_":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2026-3-1"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]

    if Y_COLUMN== "MBS_SCT_CD":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]

    if Y_COLUMN== "MBS_SCT_MD":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]    
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]
    
    if Y_COLUMN== "MBS_Burst":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]

    if Y_COLUMN== "MBS_CMT30":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[(turnup_data["AB_Grade_ID"]=="6010120") |  (turnup_data["AB_Grade_ID"]=="6010100")]
        turnup_data = turnup_data[turnup_data.index < "2026-07-1"]

    turnup_data = turnup_data[turnup_data.grammage.isin([115, 120, 100,  90,  85, 110])]
    print(f"Filtered data: {turnup_data.shape}")

    # =========================================================================
    # Preprocessing
    # =========================================================================
    print("\n--- Preprocessing ---")
    steam_pressure = [v for v in turnup_data.columns if re.search(r"cylinder.*steam_pressure", v, re.I)]
    steam_diff_pressure = [v for v in turnup_data.columns if re.search(r"cylinder.*differential_pressure", v, re.I)]

    exog_vars_reduced = [
        v for v in ctl_vars
        if (v not in BLACK_LIST[Y_COLUMN] and v not in created_vars
            and v not in steam_pressure and v not in steam_diff_pressure
            and "vacuum" not in v.lower())
    ]
    exog_vars_reduced = unique_in_order(["grammage"] + exog_vars_reduced + ["grammage"])

    for _, step in prep_pip.steps:
        if isinstance(step, GroupwisePLSTransformer):
            transformed_names = [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
            exog_vars_reduced = ordered_difference(
                unique_in_order(exog_vars_reduced + transformed_names),
                list(step.pls_columns),
            )
    exog_vars_reduced = unique_in_order(["grammage"] + exog_vars_reduced)

    pre_estimator, feat_list = build_pre_estimator(
        exog_vars_reduced=exog_vars_reduced,
        prep_pip=prep_pip,
        created_vars=created_vars,
        apply_ewm=True,  # always EWM X
    )
    print(f"Pipeline features: {len(feat_list)}")

    # =========================================================================
    # Transform
    # =========================================================================
    print("\n--- Transform ---")
    turnup_ts = turnup_data.copy().sort_index()
    turnup_ts = turnup_ts.dropna(subset=[Y_COLUMN])
    turnup_ts = turnup_ts.dropna(subset=feat_list)

    n_samples = len(turnup_ts)
    split = int(n_samples * (1.0 - TEST_SIZE))

    ts_raw = turnup_ts.loc[:, feat_list]
    pre_estimator.fit(ts_raw.iloc[:split], turnup_ts[Y_COLUMN].iloc[:split])
    ts_transformed = pre_estimator.transform(ts_raw)

    # EWM filter target (always)
    y_raw = turnup_ts[Y_COLUMN].values[:len(ts_transformed)].copy()
    grammage_group = turnup_ts["grammage"].values[:len(ts_transformed)]
    grade_change = pd.Series(grammage_group).ne(pd.Series(grammage_group).shift())
    time_gap = ts_transformed.index.to_series().diff().gt(pd.Timedelta("12h"))
    time_gap.iloc[0] = True
    seg = (grade_change.values | time_gap.values).cumsum()
    y_series = pd.Series(y_raw, index=ts_transformed.index)
    ts_transformed[Y_COLUMN] = y_series.groupby(seg).transform(ewm_reset).values

    feature_cols = [c for c in ts_transformed.columns if c != Y_COLUMN]
    X_all = ts_transformed[feature_cols].values.astype(float)
    y_all = ts_transformed[Y_COLUMN].values.astype(float)
    t_hours = to_hours(ts_transformed.index)

    print(f"Shape: {ts_transformed.shape}, split at {split}")
    print(f"std(y_raw)={np.std(y_raw):.2f}, std(EWM(y))={np.std(y_all):.2f}")

    # =========================================================================
    # Lag selection (train only, always tuned)
    # =========================================================================
    print("\n--- Lag sweep (train only) ---")
    X_train, y_train = X_all[:split], y_all[:split]
    t_hours_train = t_hours[:split]

    sweep_results = []
    for lag_d in DEFAULT_LAG_SWEEP:
        lag_h = lag_d * 24.0
        later, earlier = lag_pairs(t_hours_train, lag_h)
        if len(later) < 200:
            print(f"  lag {lag_d:6.2f}d  -- too few pairs ({len(later)}), skipping")
            continue
        dX_sw = X_train[later] - X_train[earlier]
        dy_sw = y_train[later] - y_train[earlier]
        m = build_ridge_model()
        m.fit(dX_sw, dy_sw)
        cv_rmse = -m.best_score_
        sweep_results.append({"lag_days": lag_d, "cv_rmse": cv_rmse,
                              "n_contrasts": len(later)})
        print(f"  lag {lag_d:6.2f}d  CV RMSE={cv_rmse:8.3f}  contrasts={len(later)}")

    sweep_df = pd.DataFrame(sweep_results)
    best_lag_d = float(sweep_df.loc[sweep_df["cv_rmse"].idxmin(), "lag_days"])
    print(f"  Best lag: {best_lag_d:.2f} days")
    sweep_df.to_csv(out_dir / "lag_sweep.csv", index=False)

    # =========================================================================
    # Build differenced data at the chosen lag
    # =========================================================================
    lag_h = best_lag_d * 24.0
    later, earlier = lag_pairs(t_hours, lag_h)
    dX = X_all[later] - X_all[earlier]
    dy = y_all[later] - y_all[earlier]

    tr_sel = later < split
    te_sel = later >= split
    print(f"\nLag {best_lag_d:.2f}d: {tr_sel.sum()} train, {te_sel.sum()} test contrasts")

    # =========================================================================
    # Fit model
    # =========================================================================
    print(f"\n--- Fitting model: {MODEL} ---")

    if MODEL == "ridge":
        model = build_ridge_model()
        model.fit(dX[tr_sel], dy[tr_sel])
        pred_test = model.predict(dX[te_sel])
        best_alpha = float(model.best_params_["ridge__alpha"])
        print(f"  Best alpha: {best_alpha:.2f}")

    elif MODEL == "splines":
        model = build_splines_ridge_model(
            dX.shape[1], n_knots=args.spline_knots, degree=args.spline_degree)
        model.fit(dX[tr_sel], dy[tr_sel])
        pred_test = model.predict(dX[te_sel])
        best_alpha = float(model.best_params_["ridge__alpha"])
        n_spline_features = model.best_estimator_.named_steps["splines"].n_features_out_
        print(f"  Best alpha: {best_alpha:.2f}")
        print(f"  Spline features: {dX.shape[1]} input -> {n_spline_features} after splines")

    elif MODEL == "realmlp":
        # Chronological val split from train contrasts
        n_train = int(tr_sel.sum())
        n_val = int(n_train * REALMLP_VAL_FRACTION)
        n_fit = n_train - n_val
        tr_indices = np.flatnonzero(tr_sel)
        fit_idx = tr_indices[:n_fit]
        val_idx = tr_indices[n_fit:]
        print(f"  fit: {n_fit}, val: {n_val}")

        mlp_model, mlp_scaler = build_realmlp_model(
            dX[fit_idx], dy[fit_idx], dX[val_idx], dy[val_idx])
        pred_test = mlp_model.predict(mlp_scaler.transform(dX[te_sel])).ravel()
        best_alpha = None

    # =========================================================================
    # Metrics
    # =========================================================================
    print("\n--- Results ---")
    t_anchor = ts_transformed.index[later[te_sel]]

    actual_ewm = y_all[later[te_sel]]
    persistence_ewm = y_all[earlier[te_sel]]
    model_ewm = persistence_ewm + pred_test

    actual_raw_test = y_raw[later[te_sel]]
    persistence_raw = y_raw[earlier[te_sel]]
    model_raw = persistence_ewm + pred_test

    m_dy = metrics(dy[te_sel], pred_test)
    m_ewm_pers = metrics(actual_ewm, persistence_ewm)
    m_ewm_model = metrics(actual_ewm, model_ewm)
    m_raw_pers = metrics(actual_raw_test, persistence_raw)
    m_raw_model = metrics(actual_raw_test, model_raw)

    print(f"  R2(dy):                    {m_dy['r2']:+.4f}  RMSE={m_dy['rmse']:.2f}")
    print(f"  vs EWM(y) persistence:     R2={m_ewm_pers['r2']:+.4f}  RMSE={m_ewm_pers['rmse']:.2f}")
    print(f"  vs EWM(y) model:           R2={m_ewm_model['r2']:+.4f}  RMSE={m_ewm_model['rmse']:.2f}")
    print(f"  gain over persistence:     {m_ewm_model['r2'] - m_ewm_pers['r2']:+.4f}")
    print(f"  vs raw y persistence:      R2={m_raw_pers['r2']:+.4f}  RMSE={m_raw_pers['rmse']:.2f}")
    print(f"  vs raw y model:            R2={m_raw_model['r2']:+.4f}  RMSE={m_raw_model['rmse']:.2f}")
    print(f"  gain over persistence:     {m_raw_model['r2'] - m_raw_pers['r2']:+.4f}")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\n--- Plots ---")

    # Time series
    dt_hours = np.diff(pd.DatetimeIndex(t_anchor).asi8.astype(float) / 3.6e12)
    gap_idx = np.flatnonzero(dt_hours > 12)

    def plot_gapped(ax, t, vals, gap_idx, **kwargs):
        v = np.array(vals, dtype=float).copy()
        v[gap_idx] = np.nan
        ax.plot(t, v, **kwargs)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    plot_gapped(axes[0], t_anchor, actual_ewm, gap_idx, lw=0.8, color="k", label="actual EWM(y)")
    plot_gapped(axes[0], t_anchor, persistence_ewm, gap_idx, lw=0.5, color="C7", alpha=0.7,
                label=f"persistence R2={m_ewm_pers['r2']:.3f}")
    plot_gapped(axes[0], t_anchor, model_ewm, gap_idx, lw=0.7, color="C0", alpha=0.85,
                label=f"model R2={m_ewm_model['r2']:.3f}")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{Y_COLUMN} - {MODEL} differencing lag={best_lag_d:.2f}d, vs EWM(y)")

    plot_gapped(axes[1], t_anchor, actual_raw_test, gap_idx, lw=0.5, color="k", alpha=0.7,
                label="actual raw y")
    plot_gapped(axes[1], t_anchor, model_raw, gap_idx, lw=0.7, color="C0", alpha=0.85,
                label=f"model vs raw R2={m_raw_model['r2']:.3f}")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title(f"{Y_COLUMN} - vs raw y")
    plt.tight_layout()
    plt.savefig(out_dir / "timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(actual_ewm, model_ewm, s=8, alpha=0.3)
    lim = [min(actual_ewm.min(), model_ewm.min()), max(actual_ewm.max(), model_ewm.max())]
    axes[0].plot(lim, lim, "r--", lw=1.3)
    axes[0].set_xlabel("actual EWM(y)"); axes[0].set_ylabel("predicted")
    axes[0].set_title(f"vs EWM(y): R2={m_ewm_model['r2']:.4f}")
    axes[0].set_aspect("equal", adjustable="box"); axes[0].grid(alpha=0.3)

    axes[1].scatter(actual_raw_test, model_raw, s=8, alpha=0.3, color="C1")
    lim = [min(actual_raw_test.min(), model_raw.min()), max(actual_raw_test.max(), model_raw.max())]
    axes[1].plot(lim, lim, "r--", lw=1.3)
    axes[1].set_xlabel("actual raw y"); axes[1].set_ylabel("predicted")
    axes[1].set_title(f"vs raw y: R2={m_raw_model['r2']:.4f}")
    axes[1].set_aspect("equal", adjustable="box"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Save
    # =========================================================================
    print("\n--- Saving ---")
    end_time = datetime.now()

    turnup_ts.to_parquet(out_dir / "turnup_ts.parquet")

    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for src_file in [Path(__file__).name, "config.py", "data_cleaning.py",
                     "feature_selection.py", "preprocessing.py", "state_estimation.py"]:
        src_path = Path(__file__).parent / src_file
        if src_path.exists():
            shutil.copy2(src_path, code_dir / src_file)

    results_json = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "parameters": {
            "y_column": Y_COLUMN,
            "data_path": DATA_PATH,
            "model": MODEL,
            "lag_days": float(best_lag_d),
            "apply_ewm_filter": True,
            "apply_ewm_filter_y": True,
            "spline_knots": args.spline_knots if MODEL == "splines" else None,
            "spline_degree": args.spline_degree if MODEL == "splines" else None,
            "ridge_alpha": best_alpha,
            "realmlp_params": {
                "hidden_sizes": REALMLP_HIDDEN_SIZES,
                "lr": REALMLP_LR,
                "batch_size": REALMLP_BATCH_SIZE,
                "n_epochs": REALMLP_N_EPOCHS,
                "n_refit": REALMLP_N_REFIT,
                "val_fraction": REALMLP_VAL_FRACTION,
            } if MODEL == "realmlp" else None,
        },
        "data": {
            "n_samples": int(n_samples),
            "split": int(split),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "n_train_contrasts": int(tr_sel.sum()),
            "n_test_contrasts": int(te_sel.sum()),
        },
        "metrics": {
            "dy": m_dy,
            "ewm_persistence": m_ewm_pers,
            "ewm_model": m_ewm_model,
            "raw_persistence": m_raw_pers,
            "raw_model": m_raw_model,
            "gain_over_persistence_ewm": m_ewm_model["r2"] - m_ewm_pers["r2"],
            "gain_over_persistence_raw": m_raw_model["r2"] - m_raw_pers["r2"],
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=float)

    with open(out_dir / "model.pkl", "wb") as f:
        if MODEL == "realmlp":
            cloudpickle.dump({"model": mlp_model, "scaler": mlp_scaler}, f)
        else:
            cloudpickle.dump({"model": model}, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
