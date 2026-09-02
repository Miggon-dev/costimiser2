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
    --model         ridge | splines_ridge | realmlp | gp 
    --lag_days      Differencing lag in days (if omitted, sweeps and picks best)
    --spline_knots  Number of spline knots (default 6, only for --model splines_ridge)
    --spline_degree Spline degree (default 3, only for --model splines_ridge)
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
import math
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
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df


# =============================================================================
# Configuration
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam_power":["steam_pressure", "exha_mois", "inlet_temp", "vacuum", "linepressure","fabric_tension","gas_decu"],
    "Steam__kWh/T_":["exha_mois", "inlet_temp", "linepressure","fabric_tension","gas_decu"],
    "Electrical_power_MW": ["speedsizer_linepressure", "linepressure"],
    "Electricity__kWh/T_": ["speedsizer_linepressure", "linepressure"],
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
    "Steam__kWh/T_": [
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
    "Electricity__kWh/T_": [
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
    "Steam__kWh/T_": [
        "Water_Predryer",
        "diluted_starch",
        "Fibre__g/m2_",            
        "Water_Afterdryer_output",
        "dewatering",
        "inv_Production_Rate__T/h_",
        "fibre_short/long"
    ],
    "Electrical_power_MW": [
        "dewatering",
        "Fibre__g/m2_",
    ],
    "Electricity__kWh/T_": [
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
     "MBS_SCT_MD": [
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
     "MBS_Burst": [
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
     "MBS_CMT30": [
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
DEFAULT_LAG_SWEEP = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 7, 14]

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


def ewm_filter(s, halflife="120min"):
    """Grade-aware EWM with configurable smoothness. Drop-in for utility.ewm_reset."""
    out = s.ewm(halflife=halflife, times=s.index, adjust=True).mean()
    if len(s) >= 2:
        out.iloc[0] = (s.iloc[0] + s.iloc[1]) / 2.0
    return out


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


def build_gp_model():
    """Gaussian Process with RBF + White kernel, standardised input."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # noise is in the WhiteKernel
            n_restarts_optimizer=3,
            random_state=42,
        )),
    ])


def build_gam_model(n_features):
    """
    Generalized Additive Model (GAM) using pygam.

    Each feature gets its own smooth spline term: y = sum_i f_i(x_i).
    No interactions -> smooth per-feature response curves.
    Built-in feature selection: lambda penalises each term independently,
    irrelevant features get smoothed to zero.
    """
    from pygam import LinearGAM, s

    # Build sum of spline terms, one per feature
    # n_splines=20 gives enough flexibility, lam will regularise
    terms = s(0, n_splines=20)
    for i in range(1, n_features):
        terms += s(i, n_splines=20)

    gam = LinearGAM(terms)
    return gam


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
    parser.add_argument("--model", choices=["ridge", "splines_ridge", "realmlp", "gp", "gam", "hgbr", "fs_ridge"], default="ridge")
    parser.add_argument("--lag_days", type=float, default=0.25,
                        help="Differencing lag in days (default 0.25 = 6 hours). "
                             "Set based on process knowledge: long enough to see real "
                             "variation (> 2x EWM halflife) but short enough to stay "
                             "within a single grade run.")
    parser.add_argument("--ewm_halflife", type=int, default=0)
    parser.add_argument("--spline_knots", type=int, default=6)
    parser.add_argument("--spline_degree", type=int, default=3)
    parser.add_argument("--fs_k_features", type=int, default=15,
                        help="Number of features to select (for --model fs_ridge)")
    args = parser.parse_args()

    # Derive EWM halflife from the lag: halflife = lag / 3
    # This defines the band-pass: EWM removes noise faster than halflife,
    # differencing removes drift slower than the lag. The ratio of 3 ensures
    # there is meaningful variation in the signal band between them.
    lag_hours = args.lag_days * 24.0
    
    if args.ewm_halflife == 0:
        ewm_halflife_hours = lag_hours / 3.0
    else:
        ewm_halflife_hours = args.ewm_halflife
    if ewm_halflife_hours >= 1.0:
        ewm_halflife = f"{ewm_halflife_hours:.0f}h" if ewm_halflife_hours == int(ewm_halflife_hours) else f"{ewm_halflife_hours:.1f}h"
    else:
        ewm_halflife = f"{ewm_halflife_hours * 60:.0f}min"
    print(f"Lag: {args.lag_days:.2f}d ({lag_hours:.1f}h) -> EWM halflife: {ewm_halflife} (lag/3)")

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

    cond_col = turnup_data['Condensate_energy_from_paper_plant_to_power_plant']
    cond_med = cond_col[(cond_col >= 0) & (cond_col <= 10)].median()
    if  math.isnan(cond_med) :
        cond_med = 5.11
    cond = cond_col.where((cond_col >= 0) & (cond_col <= 10), cond_med)

    turnup_data["Steam_power"] = turnup_data['Steam_flow_to_PM'] + turnup_data['Waste_steam_flow'] 
    turnup_data["Steam__kWh/T_"]  = (((turnup_data["Steam_power"] * 0.788 - cond) * 1.02 - (0.5938 / 24)) / turnup_data["Production_Rate__T/h_"]) * 1000
    #turnup_data["Steam_power"] = turnup_data["Steam_flow_from_power_plant_to_PM"] + turnup_data["Waste_steam_flow"]
    
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
    if Y_COLUMN== "Steam__kWh/T_":
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
        #turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[turnup_data.index>"2025-11-15"]
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
        ewm_halflife=ewm_halflife,
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

    # EWM filter target (same halflife as X)
    y_raw = turnup_ts[Y_COLUMN].values[:len(ts_transformed)].copy()
    grammage_group = turnup_ts["grammage"].values[:len(ts_transformed)]
    grade_change = pd.Series(grammage_group).ne(pd.Series(grammage_group).shift())
    time_gap = ts_transformed.index.to_series().diff().gt(pd.Timedelta("12h"))
    time_gap.iloc[0] = True
    seg = (grade_change.values | time_gap.values).cumsum()
    y_series = pd.Series(y_raw, index=ts_transformed.index)
    ts_transformed[Y_COLUMN] = y_series.groupby(seg).transform(
        lambda s: ewm_filter(s, halflife=ewm_halflife)).values

    feature_cols = [c for c in ts_transformed.columns if c != Y_COLUMN]
    X_all = ts_transformed[feature_cols].values.astype(float)
    y_all = ts_transformed[Y_COLUMN].values.astype(float)
    t_hours = to_hours(ts_transformed.index)

    print(f"Shape: {ts_transformed.shape}, split at {split}")
    print(f"std(y_raw)={np.std(y_raw):.2f}, std(EWM(y))={np.std(y_all):.2f}")

    # =========================================================================
    # Lag sweep (informational, train only)
    # =========================================================================
    # The lag is set by --lag_days based on process knowledge. The sweep is
    # reported for context but does NOT drive the selection, because:
    #   - Raw RMSE favours short lags (EWM guarantees smoothness)
    #   - R2 favours long lags (grade changes dominate)
    #   - Neither correctly identifies the within-run signal band
    # The right lag is: longer than 2x EWM halflife, shorter than a grade run.
    print("\n--- Lag sweep (informational, train only) ---")
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
        persistence_var = float(np.var(dy_sw))
        cv_r2 = 1 - (cv_rmse ** 2) / persistence_var if persistence_var > 0 else 0.0

        sweep_results.append({"lag_days": lag_d, "cv_rmse": cv_rmse,
                              "persistence_rmse": float(np.sqrt(persistence_var)),
                              "cv_r2": cv_r2,
                              "n_contrasts": len(later)})
        marker = "  <-- selected" if abs(lag_d - args.lag_days) < 0.001 else ""
        print(f"  lag {lag_d:6.2f}d  CV RMSE={cv_rmse:8.3f}  "
              f"persist RMSE={np.sqrt(persistence_var):8.3f}  "
              f"CV R2(dy)={cv_r2:+.4f}  contrasts={len(later)}{marker}")

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(out_dir / "lag_sweep.csv", index=False)

    best_lag_d = args.lag_days
    print(f"\n  Using lag: {best_lag_d:.2f} days (set by --lag_days)")

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

    elif MODEL == "splines_ridge":
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

    elif MODEL == "gp":
        # Gaussian Process — fits on train contrasts (can be slow for large N,
        # so subsample if > 2000 train contrasts)
        max_gp_train = 2000
        tr_idx_gp = np.flatnonzero(tr_sel)
        if len(tr_idx_gp) > max_gp_train:
            rng_gp = np.random.default_rng(42)
            tr_idx_gp = np.sort(rng_gp.choice(tr_idx_gp, max_gp_train, replace=False))
            print(f"  GP: subsampled train to {max_gp_train} contrasts (from {tr_sel.sum()})")

        model = build_gp_model()
        model.fit(dX[tr_idx_gp], dy[tr_idx_gp])
        pred_test = model.predict(dX[te_sel])
        best_alpha = None
        gp_step = model.named_steps["gp"]
        print(f"  GP kernel: {gp_step.kernel_}")
        print(f"  GP log-marginal-likelihood: {gp_step.log_marginal_likelihood_value_:.2f}")

    elif MODEL == "gam":
        # GAM: additive smooth splines, one per feature
        # Standardise input first (GAM benefits from normalised features)
        scaler_gam = StandardScaler()
        dX_tr_scaled = scaler_gam.fit_transform(dX[tr_sel])
        dX_te_scaled = scaler_gam.transform(dX[te_sel])

        gam = build_gam_model(dX.shape[1])
        # gridsearch over lambda (regularisation strength)
        gam.gridsearch(dX_tr_scaled, dy[tr_sel], progress=True)
        pred_test = gam.predict(dX_te_scaled)
        best_alpha = None
        model = gam  # for saving

        # Report feature importance (sum of absolute partial effects)
        print(f"  GAM pseudo-R2 (train): {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
        print(f"  Number of significant features (p < 0.05):")
        p_values = gam.statistics_['p_values']
        sig_count = sum(1 for p in p_values if p < 0.05)
        print(f"    {sig_count}/{len(p_values)} terms significant")

    elif MODEL == "hgbr":
        # HistGradientBoosting with max_depth=1: additive stumps (no interactions)
        # Fast, handles many features naturally, effectively does feature selection
        # via boosting (irrelevant features never get split on).
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("hgbr", HistGradientBoostingRegressor(
                max_depth=1,
                max_iter=500,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )),
        ])
        model.fit(dX[tr_sel], dy[tr_sel])
        pred_test = model.predict(dX[te_sel])
        best_alpha = None
        hgbr = model.named_steps["hgbr"]
        print(f"  HGBR iterations used: {hgbr.n_iter_}")
        print(f"  HGBR train score: {hgbr.train_score_[-1]:.4f}")

    elif MODEL == "fs_ridge":
        # Mutual information feature selection + Ridge
        from sklearn.feature_selection import mutual_info_regression, SelectKBest

        # Select top-k features by mutual information (on train only)
        k = min(args.fs_k_features, dX.shape[1])
        selector = SelectKBest(
            score_func=mutual_info_regression, k=k
        )
        dX_tr_selected = selector.fit_transform(dX[tr_sel], dy[tr_sel])
        dX_te_selected = selector.transform(dX[te_sel])

        # Which features were selected?
        selected_mask = selector.get_support()
        selected_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
        mi_scores = selector.scores_

        print(f"  Selected {k}/{dX.shape[1]} features by mutual information")
        print(f"  Selected: {selected_names}")

        # Fit Ridge on the selected features
        model = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            param_grid={"ridge__alpha": ALPHAS.tolist()},
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_root_mean_squared_error",
            refit=True,
        )
        model.fit(dX_tr_selected, dy[tr_sel])
        pred_test = model.predict(dX_te_selected)
        best_alpha = float(model.best_params_["ridge__alpha"])
        print(f"  Best alpha: {best_alpha:.2f}")

        # Report MI scores for all features
        mi_df = pd.DataFrame({
            "feature": feature_cols,
            "mi_score": mi_scores,
            "selected": selected_mask,
        }).sort_values("mi_score", ascending=False)
        mi_df.to_csv(out_dir / "mi_scores.csv", index=False)
        print(f"\n  Top 10 by MI:")
        print(mi_df.head(10).to_string(index=False))

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

    # Save train/test differenced data for downstream analysis (SHAP, etc.)
    dX_train_df = pd.DataFrame(dX[tr_sel], columns=feature_cols)
    dX_test_df = pd.DataFrame(dX[te_sel], columns=feature_cols)
    dX_train_df["AB_Grade_ID"] = turnup_ts["AB_Grade_ID"].values[later[tr_sel]]
    dX_test_df["AB_Grade_ID"] = turnup_ts["AB_Grade_ID"].values[later[te_sel]]
    dX_train_df.to_parquet(out_dir / "dX_train.parquet")
    dX_test_df.to_parquet(out_dir / "dX_test.parquet")
    pd.DataFrame({"dy_train": dy[tr_sel]}).to_parquet(out_dir / "dy_train.parquet")
    pd.DataFrame({"dy_test": dy[te_sel]}).to_parquet(out_dir / "dy_test.parquet")

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
            "ewm_halflife": ewm_halflife,
            "spline_knots": args.spline_knots if MODEL == "splines_ridge" else None,
            "spline_degree": args.spline_degree if MODEL == "splines_ridge" else None,
            "ridge_alpha": best_alpha,
            "fs_k_features": args.fs_k_features if MODEL == "fs_ridge" else None,
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
            cloudpickle.dump({"model": mlp_model, "scaler": mlp_scaler,
                              "pre_estimator": pre_estimator,
                              "feat_list": feat_list}, f)
        elif MODEL == "gam":
            cloudpickle.dump({"model": gam, "scaler": scaler_gam,
                              "pre_estimator": pre_estimator,
                              "feat_list": feat_list}, f)
        elif MODEL == "fs_ridge":
            cloudpickle.dump({"model": model, "pre_estimator": pre_estimator,
                              "feat_list": feat_list,
                              "selected_features": selected_names}, f)
        else:
            cloudpickle.dump({"model": model, "pre_estimator": pre_estimator,
                              "feat_list": feat_list}, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
