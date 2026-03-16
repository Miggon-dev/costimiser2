import pandas as pd
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

logger = logging.getLogger("app")

control_vars = [
    "Current_basis_weight", # Scanner
    #"Fibre__g/m2_",
    "Speed", # Scanner
    "Current_reel_moisture_average(reel)", # Scanner

    "Sizing_Agent__g/T_", # Size Press
    "Starch_uptake__g/m2_", # Size Press
    "Draw_PD5-SS",  # Size Press
    "Defoamer_mass_flow__g/T_", # Size Press
    "Dry_Strength_Agent_mass_flow__kg/T_", # Size Press
    "SpeedSizer_Linepressure_DS", # Size Press
    "SpeedSizer_Linepressure_FS", # Size Press
    "Consistency_starch_main_line", # Size Press
    
    "Draw_PS-PD1", #Pre-dryer
    "Draw_PD2-PD3", #Pre-dryer
    "Draw_PD4-PD5", #Pre-dryer 
    "Draw_PD3-PD4", #Pre-dryer 
    "Draw_PD1-PD2", #Pre-dryer
    "Moisture_out_of_PreDryer", #Pre-dryer

    "Draw_WS-PS", # Press Section
    "PickUp_Tension", # Press Section
    "Vacuum_presszone_of_suction-press_roll", # Press Section
    "Vacuum_uhle-box_Pick-Up", # Press Section
    "Vacuum_uhle-box_bottom_felt", # Press Section
    "Linepressure_1st_press_FS__bar_", # Press Section
    "Linepressure_2nd_press_FS__bar_", # Press Section
    "Linepressure_1st_press_DS__bar_", # Press Section
    "Linepressure_2nd_press_DS__bar_", # Press Section 
    "Linepressure_shoe_press__bar_", # Press Section 

    #"Dewatering_top_wire_suction_box_zone_1", # Forming Wire (BAD)
    #"Dewatering_Jet_channel", # Forming Wire (BAD)
    "Dewatering_top_wire_suction_box_zone_2", # Forming Wire
    "Vacuum_suction_box_9", "Vacuum_wet_suction_box", # Forming Wire
    "Vacuum_sheet_seperator_box", # Forming Wire
    "Vacuum_suction_box_10", # Forming Wire
    "Vacuum_suction_box_11", # Forming Wire
    "Vacuum_wire_suction_box_1", # Forming Wire
    "Vacuum_wire_suction_box_2", # Forming Wire
    "Consistency_white_water", # Forming Wire
    "White_water_temperature", # Forming Wire
    "Conductivity_white_water_B46", # Forming Wire
    "Top_wire_tenstion", # Forming Wire
    "pH_measurement_white_water_B41", # Forming Wire

    "Jet/wire_ratio", # Headbox
    "Lip_settings", # Headbox

    "Rod_Pressure_Bottom_Roll",
    "Rod_pressure_Top_Roll",

    "Retention_Aid_mass_flow__g/T_", # Approach Flow
    "Bentonite_1_mass_flow__g/T_", # Approach Flow
    "Bentonite_2_mass_flow__g/T_", # Approach Flow
    "Thick_Stock_Consistency__%_", # Approach Flow
    "delta_moisture",

    "Vacuum_holding/pre_positions_of_suction-press_roll",
    "Rod_clamping_pressure_Top_Roll",
    "Rod_clamping_pressure_Bottom_Roll",
    "Dewatering_Pick-Up",

    "Headbox_consistency",

    "pH-Messung_Verdünnungswasser__2..12_pH_",
    "fibre_short/long",
    "Dissolved_gas_after_stock_deculator_measurement_1",
    "Dissolved_gas_before_dilution_water_deculator",
    "Defoamer_mass_flow__g/T_",
    "Web_tension_AD6",
    "Top_Felt_Tension",
    "Bottom_wire_tension",
    "Bottom_Felt_Tension",
    "AD7_fabric_tension_bottom",
    "PD4_fabric_tension",
    "Web_tension_AD6",
    "AD6_fabric_tension",
    "PD1_Fabric_tension",
    "PD2_Fabric_tension",
    "PD5_fabric_tension_top",
    "PD5_fabric_tension_bottom",

    "Vacuum_Zone_1_PickUp",
    "Linepressure_1st_press_FS__bar_",
    "Linepressure_1st_press_DS__bar_",
    "Linepressure_2nd_press_DS__bar_",
    "Linepressure_2nd_press_FS__bar_",
    "Act_Deaerator_mass_flow__g/T_",
    
    "Natriumhydroxide_mass_flow__g/T_",

    "Air_pressure_of_rod_clamping_hose_Top_Roll",
    "Free_gas_before_dilution_water_deculator_measurement_2",
    "Free_gas_before_dilution_water_deculator_measurement_1~^0",
    "Free_gas_after_stock_deculator~^0",
    "Storage_tank_temperature",

    "Vacuum_top_wire_suction_box_zone_2",
    "Vacuum_formning_roll",
    "Vacuum_top_wire_suction_box_zone_1",
    "Airturn_pillow_pressure",
    "Dewatering_Suction_Press_Roll",
    "Dewatering_First_Press_Roll",

    "Current_reel_width",
    "Steam_temperature_for_PM",
    "Steam_pressure_for_PM",

    'Dewatering_Shoe_press',
    'Dewatering_Pick-Up',
    'Total_Dewatering_Press',
    'Dewatering_top_wire_suction_box_zone_2',
    'Uhle_box_1_flow___l/min_',
    
    "Stock_deculator_temperature",
    'Stock_deculator_pressure',

    'Multifractor_1_Long_fibre_fraction',
    'Multifractor_2_long_fibre_fraction', 
    'Multifractor_3_long_fibre_fraction',

    'Fixative_2_mass_flow__g/T_',
    'Act_Deaerator_mass_flow__g/T_',
    
    'DG4_Temperature_Inlet_Air',
    'DG5_Temperature_Inlet_Air',
    'DG1_temperature_Inlet_Air',
    'DG2_temperature_Inlet_Air',
    'DG3_temperature_Inlet_Air',

    'DG4_Moisture_content_Outlet_Air',
    'DG5_Moisture_content_Outlet_Air',
    'DG1_Moisture_content_Outlet_Air',
    'DG2_Moisture_content_Outlet_Air',
    'DG3_Moisture_content_Outlet_Air',

    '3200', 
    '3300', 
    '6010', 
    'concentration_starch_working_tank_1', 
    'concentration_starch_working_tank_2',

    'ambient_temp_C'

]

setpoints={
  "3200115": {
    "MBS_SCT_CD": {
      "target": 2.15,
      "min": 1.95
    },
    "MBS_SCT_MD": {
      "target": 3.6,
      #"min": 3.25
    },
    "MBS_Burst": {
      "target": 270.0,
      "min": 240.0
    }
  },
  "3300085": {
    "MBS_SCT_CD": {
      "target": 1.35,
      "min": 1.3
    },
    "MBS_SCT_MD": {
      "target": 2.30
    },
    "MBS_Burst": {
      "target": 180.0,
      "min": 170.0
    }
  },
  "3300090": {
    "MBS_SCT_CD": {
      "target": 1.55,
      "min": 1.4
    },
    "MBS_SCT_MD": {
      "target": 2.45,     
    },
    "MBS_Burst": {
      "target": 195.0,
      "min": 180.0
    }
  },
  "3300095": {
    "MBS_SCT_CD": {
      "target": 1.6,
      "min": 1.45
    },
    "MBS_SCT_MD": {
      "target": 2.7
    },
    "MBS_Burst": {
      "target": 210.0,
      "min": 190.0
    }
  },
  "3300100": {
    "MBS_SCT_CD": {
      "target": 1.7,
      "min": 1.55
    },
    "MBS_SCT_MD": {
      "target": 2.9,
      #"min": 2.7
    },
    "MBS_Burst": {
      "target": 220.0,
      "min": 200.0
    }
  },
  "3300105": {
    "MBS_SCT_CD": {
      "target": 1.7,
      "min": 1.6
    },
    "MBS_SCT_MD": {
      "target": 2.96,
      #"min": 2.76
    },
    "MBS_Burst": {
      "target": 230.0,
      "min": 210.0
    }
  },
  "3300110": {
    "MBS_SCT_CD": {
      "target": 1.9,
      "min": 1.7
    },
    "MBS_SCT_MD": {
      "target": 3.05,
      #"min": 2.95
    },
    "MBS_Burst": {
      "target": 240.0,
      "min": 220.0
    }
  },
  "3300115": {
    "MBS_SCT_CD": {
      "target": 2.0,
      "min": 1.75
    },
    "MBS_SCT_MD": {
      "target": 3.15,
      #"min": 3.05
    },
    "MBS_Burst": {
      "target": 265.0,
      "min": 230.0
    }
  },
  "3300120": {
    "MBS_SCT_CD": {
      "target": 2.1,
      "min": 1.85
    },
    "MBS_SCT_MD": {
      "target": 3.3,
      #"min": 3.15
    },
    "MBS_Burst": {
      "target": 285.0,
      "min": 240.0
    }
  },
  "3300125": {
    "MBS_SCT_CD": {
      "target": 2.15,
      "min": 1.9
    },
    "MBS_SCT_MD": {
      "target": 3.4,
      #"min": 3.3
    },
    "MBS_Burst": {
      "target": 290.0,
      "min": 270.0
    }
  },
  "3300130": {
    "MBS_SCT_CD": {
      "target": 2.25,
      "min": 2.0
    },
    "MBS_SCT_MD": {
      "target": 3.6,
      #"min": 3.4
    },
    "MBS_Burst": {
      "target": 310.0,
      "min": 260.0
    }
  },
  "3300135": {
    "MBS_SCT_CD": {
      "target": 2.3,
      "min": 2.05
    },
    "MBS_SCT_MD": {
      "target": 3.65,
      #"min": 3.6
    },
    "MBS_Burst": {
      "target": 315.0,
      "min": 290.0
    }
  },
  "6010085": {
    "MBS_SCT_CD": {
      "target": 1.6,
      "min": 1.45
    },
    "MBS_SCT_MD": {
      "target": 2.8,
      #"min": 2.65
    },
    "MBS_Burst": {
      "target": 205.0,
      #"min": 195.0
    }
  },
  "6010090": {
    "MBS_SCT_CD": {
      "target": 1.65,
      "min": 1.5
    },
    "MBS_SCT_MD": {
      "target": 3.0,
      #"min": 2.8
    },
    "MBS_Burst": {
      "target": 215.0,
      #"min": 205.0
    }
  },
  "601095": {
    "MBS_SCT_CD": {
      "target": 1.75,
      "min": 1.6
    },
    "MBS_SCT_MD": {
      "target": 3.1,
      #"min": 3.0
    },
    "MBS_Burst": {
      "target": 225.0,
      #"min": 215.0
    }
  },
  "6010100": {
    "MBS_SCT_CD": {
      "target": 1.85,
      "min": 1.7
    },
    "MBS_SCT_MD": {
      "target": 3.2,
      #"min": 3.1
    },
    "MBS_Burst": {
      "target": 250.0,
      #"min": 225.0
    },
    "MBS_CMT30": {
      "target": 160.0,
      "min": 150.0
    }
  },
  "6010110": {
    "MBS_SCT_CD": {
      "target": 2.05,
      "min": 1.9
    },
    "MBS_SCT_MD": {
      "target": 3.5,
      #"min": 3.4
    },
    "MBS_Burst": {
      "target": 275.0,
      #"min": 250.0
    },
    "MBS_CMT30": {
      "target": 185.0,
      "min": 165.0
    }
  },
  "6010120": {
    "MBS_SCT_CD": {
      "target": 2.2,
      "min": 2.05
    },
    "MBS_SCT_MD": {
      "target": 3.8,
      #"min": 3.7
    },
    "MBS_Burst": {
      "target": 300.0,
      #"min": 275.0
    },
    "MBS_CMT30": {
      "target": 195.0,
      "min": 185.0
    }
  },
  "6035105": {
    "MBS_SCT_CD": {
      "target": 2.15,
      "min": 1.95
    },
    "MBS_SCT_MD": {
      "target": 3.55,
      #"min": 3.2
    },
    "MBS_Burst": {
      "target": 285.0,
      #"min": 275.0
    },
    "MBS_CMT30": {
      "target": 205.0,
      "min": 185.0
    }
  },
  "6035110": {
    "MBS_SCT_CD": {
      "target": 2.25,
      "min": 2.05
    },
    "MBS_SCT_MD": {
      "target": 3.75,
      #"min": 3.55
    },
    "MBS_Burst": {
      "target": 295.0,
      #"min": 285.0
    },
    "MBS_CMT30": {
      "target": 215.0,
      "min": 190.0
    }
  },
  "6035115": {
    "MBS_SCT_CD": {
      "target": 2.35,
      "min": 2.15
    },
    "MBS_SCT_MD": {
      "target": 3.9,
      #"min": 3.75
    },
    "MBS_Burst": {
      "target": 310.0,
      #"min": 295.0
    },
    "MBS_CMT30": {
      "target": 225.0,
      "min": 210.0
    }
  },
  "6035120": {
    "MBS_SCT_CD": {
      "target": 2.5,
      "min": 2.25
    },
    "MBS_CMT30": {
      "target": 235.0,
      "min": 220.0
    }
  },
  "360085": {
    "MBS_SCT_CD": {
      "target": 1.6,
      "min": 1.45
    },
    "MBS_SCT_MD": {
      "target": 2.7,
      #"min": 2.4
    },
    "MBS_Burst": {
      "target": 215.0,
      "min": 180.0
    }
  },
  "3600100": {
    "MBS_SCT_CD": {
      "target": 1.85,
      "min": 1.7
    },
    "MBS_SCT_MD": {
      "target": 3.1,
      #"min": 2.8
    },
    "MBS_Burst": {
      "target": 250.0,
      "min": 210.0
    },
    "MBS_CMT30": {
      "target": 170.0,
      "min": 140.0
    }
  },
  "3600110": {
    "MBS_SCT_CD": {
      "target": 2.05,
      "min": 1.9
    },
    "MBS_SCT_MD": {
      "target": 3.4,
      #"min": 3.1
    },
    "MBS_Burst": {
      "target": 275.0,
      "min": 230.0
    },
    "MBS_CMT30": {
      "target": 185.0,
      "min": 155.0
    }
  },
  "3600125": {
    "MBS_SCT_CD": {
      "target": 2.3,
      "min": 2.15
    },
    "MBS_SCT_MD": {
      "target": 3.8,
      #"min": 3.5
    },
    "MBS_Burst": {
      "target": 310.0,
      "min": 260.0
    },
    "MBS_CMT30": {
      "target": 215.0,
      "min": 180.0
    }
  },
  
}

records = []

for item_id, sensors in setpoints.items():
    for sensor_name, svalues in sensors.items():
        for limit_name, lvalue in svalues.items():            
            records.append({
                "AB_Grade_ID": item_id,
                "property": sensor_name,
                "variable": limit_name,
                "value": lvalue,                
            })
setpoint_df = pd.DataFrame(records)


def gmm_with_min_size(df_scores, min_size=10, k_min=2, k_max=None, random_state=1):
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    X = df_scores  # assuming this is an arraylike (n_samples, n_features)
    n = len(X)
    if k_max is None:
        k_max = max(k_min+1, min(n // min_size, n-1))  # reasonable ceiling

    best = {
        "score": -np.inf,
        "k": None,
        "model": None,
        "labels": None,
        "sizes": None
    }

    for k in range(k_min, k_max + 1):
        # Skip impossible k given the min_size constraint
        if k * min_size > n:
            continue

        gm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
        gm.fit(X)
        labels = gm.predict(X)
        sizes = np.bincount(labels, minlength=k)

        # Enforce the minimum size
        if sizes.min() < min_size:
            continue

        sc = silhouette_score(X, labels) if k > 1 else -np.inf

        if sc > best["score"]:
            best = {"score": sc, "k": k, "model": gm, "labels": labels, "sizes": sizes}

    if best["model"] is None:
        raise ValueError(
            f"No viable GMM found that satisfies min_size={min_size}. "
            f"Try reducing min_size, lowering k_max, or using a different algorithm."
        )

    return best["model"], best["labels"], best["k"], best["sizes"]

def weighted_average(df, value_name, n_name):
    return (df[value_name] * df[n_name]).sum() / df[n_name].sum()

def decompose_avg_cost_change(df_prev, df_curr, y_variable_cost):
    import pandas as pd
    import numpy as np

    """
    Decompose change in average cost per unit between two months
    into:
      - mix effect
      - efficiency (cost) effect

    Both df_prev and df_curr must have columns: ['AB_Grade_ID', 'Combined_cost__€/T_', 'n_samples'].

    Assumptions for products not present in one of the months:
      - New products (only in current): previous cost = current cost, previous n = 0
      - Discontinued products (only in previous): current cost = previous cost, current n = 0

    This forces all 'new vs discontinued' impact into the mix effect.
    """

    # Rename columns to mark periods
    prev = df_prev.rename(columns={y_variable_cost: "cost_0", "n_samples": "n_0"})
    curr = df_curr.rename(columns={y_variable_cost: "cost_1", "n_samples": "n_1"})

    # Outer join on product to include new & discontinued products
    df = pd.merge(prev, curr, on="AB_Grade_ID", how="outer")

    # Fill missing quantities with 0 (no production in that month)
    df["n_0"] = df["n_0"].fillna(0)
    df["n_1"] = df["n_1"].fillna(0)

    # For missing costs:
    # - if cost_0 is missing but cost_1 exists (new product), set cost_0 = cost_1
    # - if cost_1 is missing but cost_0 exists (discontinued), set cost_1 = cost_0
    df["cost_0"] = np.where(df["cost_0"].isna(), df["cost_1"], df["cost_0"])
    df["cost_1"] = np.where(df["cost_1"].isna(), df["cost_0"], df["cost_1"])

    # --- Average costs ---

    # Previous month average cost
    Q0 = df["n_0"].sum()
    V0 = (df["cost_0"] * df["n_0"]).sum()
    AC0 = V0 / Q0 if Q0 != 0 else float("nan")

    # Current month average cost
    Q1 = df["n_1"].sum()
    V1 = (df["cost_1"] * df["n_1"]).sum()
    AC1 = V1 / Q1 if Q1 != 0 else float("nan")

    # --- Mix-only average cost (old costs, new quantities) ---
    V_mix = (df["cost_0"] * df["n_1"]).sum()
    AC_mix = V_mix / Q1 if Q1 != 0 else float("nan")

    # --- Effects ---
    mix_effect = AC_mix - AC0            # due to change in mix (incl. new/discontinued)
    efficiency_effect = AC1 - AC_mix     # due to change in unit costs

    total_change = AC1 - AC0
    recon_sum = mix_effect + efficiency_effect

    return {
        "AC0": AC0,
        "AC1": AC1,
        "total_change": total_change,
        "mix_effect": mix_effect,
        "cost_effect": efficiency_effect,
        "reconciliation_sum": recon_sum,
        "detail": df  # optional, to inspect intermediate data
    }

def get_process_grouped(process_data, y_variable_summary, x_variable_summary, color_variable_summary, agg_cost_label, costs_to_consider, overprocessing_vars, y_variable_summary_secondary=None):
    import plotly.express as px
    import numpy as np

    df = process_data.sort_values(["grammage","paper_type"])

    df["Wedge_Date"] = df["Wedge_Time"].dt.date
    df["Wedge_Week"] = df["Wedge_Time"].dt.isocalendar().week
    df["Wedge_Month"] = df["Wedge_Time"].dt.month
    df["Wedge_Year"] = df["Wedge_Time"].dt.year

    x_var="Wedge_Date"
    if x_variable_summary=="grade":
        x_var="AB_Grade_ID"
        grouped_vars=[x_var]
    elif x_variable_summary=="day":
        x_var="Wedge_Date"
        grouped_vars=[x_var]
    elif x_variable_summary=="week":
        x_var="Wedge_Week"
        grouped_vars=["Wedge_Year", x_var]
    elif x_variable_summary=="month":
        x_var="Wedge_Month"
        grouped_vars=["Wedge_Year", x_var]
    elif x_variable_summary=="year":
        x_var="Wedge_Year"
        grouped_vars=[x_var]
    elif x_variable_summary=="target":
        x_var="target"
        grouped_vars=[x_var]

    

    if color_variable_summary=="grade":
        color_var="AB_Grade_ID"
        grouped_vars=grouped_vars+[color_var] 
        unique_vals = df[color_var].unique()  
        n_colors = len(unique_vals)
        colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, n_colors))
        c = dict(zip(unique_vals, colors))   
    elif color_variable_summary=="target":
        color_var="target"
        grouped_vars=grouped_vars+[color_var]
        c = dict(zip(["historic","best","current"], ["blue","green","red"]))        
    elif color_variable_summary=="target_grade":
        color_var="target"
        grouped_vars=grouped_vars+[color_var,"AB_Grade_ID"]
        unique_vals = df[color_var].unique()  
        n_colors = len(unique_vals)
        colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, n_colors))
        c = dict(zip(unique_vals, colors))
    elif  color_variable_summary=="cost":
        vv_group = ['MBS_Current_reel_ID', x_var,"grammage","paper_type"]
        vv_summary = [v for v in [y_variable_summary, y_variable_summary_secondary] if v is not None]
        df = df[costs_to_consider + vv_group + vv_summary]        
        df = pd.melt(df, id_vars= [v for v in  vv_group + [y_variable_summary_secondary] if v is not None], value_vars=costs_to_consider , var_name="cost").reset_index()
        df[agg_cost_label]=df["value"]
        color_var="cost"      
        grouped_vars=grouped_vars+[color_var]
        c = dict(zip(df[color_var].unique(), px.colors.qualitative.Plotly))              
    elif  color_variable_summary=="cost_grade":
        df = df[costs_to_consider + ['MBS_Current_reel_ID',"AB_Grade_ID",y_variable_summary,x_var,"grammage","paper_type"]]        
        df = pd.melt(df, id_vars= ['MBS_Current_reel_ID',"AB_Grade_ID",x_var,"grammage","paper_type"], value_vars=costs_to_consider, var_name="cost").reset_index()
        df[agg_cost_label]=df["value"]
        color_var=["cost","AB_Grade_ID"]
        grouped_vars=grouped_vars+color_var    
        c = dict(zip(df["cost"].unique(), px.colors.qualitative.Plotly)) 
    elif  color_variable_summary=="overprocessing":
        vv_group = ['MBS_Current_reel_ID', x_var,"grammage","paper_type"]
        vv_summary = [v for v in [y_variable_summary_secondary] if v is not None]
        df = df[overprocessing_vars + vv_group + vv_summary]           
        df = pd.melt(df, id_vars= [v for v in  vv_group + [y_variable_summary_secondary] if v is not None], value_vars=overprocessing_vars , var_name="cost", value_name="Overprocessing_percentage").reset_index()  
        df["cost"]=df["cost"].str.replace("Overprocessing_","")       
        color_var="cost"
        grouped_vars=grouped_vars+[color_var]
        y_variable_summary = "Overprocessing_percentage"
        c = dict(zip(df["cost"], px.colors.qualitative.Plotly))       
    elif  color_variable_summary=="overprocessing_grade":
        vv_group = ['MBS_Current_reel_ID', "AB_Grade_ID", x_var,"grammage","paper_type"]
        vv_summary = [v for v in [y_variable_summary_secondary] if v is not None]
        df = df[overprocessing_vars + vv_group + vv_summary]             
        df = pd.melt(df, id_vars= [v for v in  vv_group + [y_variable_summary_secondary] if v is not None], value_vars=overprocessing_vars , var_name="cost", value_name="Overprocessing_percentage").reset_index()  
        df["cost"]=df["cost"].str.replace("Overprocessing_","")       
        color_var=["cost","AB_Grade_ID"]
        grouped_vars=grouped_vars + color_var
        y_variable_summary = "Overprocessing_percentage"
        c = dict(zip(df["cost"], px.colors.qualitative.Plotly))
    else:
        color_var=None
        c = None

    if x_variable_summary=="grade":
        if color_variable_summary=="target":
                dfg=df.groupby(grouped_vars+["grammage","paper_type"]).agg({y_variable_summary:"mean", "MBS_Current_reel_ID":"count"}).rename(columns={"MBS_Current_reel_ID":"n"}).reset_index().sort_values(["target","grammage","paper_type"],ascending=[False, True, True])
        else:
                dfg=df.groupby(grouped_vars+["grammage","paper_type"]).agg({y_variable_summary:"mean", "MBS_Current_reel_ID":"count"}).rename(columns={"MBS_Current_reel_ID":"n"}).reset_index().sort_values(["grammage","paper_type"])            
    else:        
        dfg=df.groupby(grouped_vars).agg({y_variable_summary:"mean", "MBS_Current_reel_ID":"count"}).rename(columns={"MBS_Current_reel_ID":"n"}).reset_index()

    if y_variable_summary=="Overprocessing_percentage":
          df[y_variable_summary] = df[y_variable_summary]*100
          dfg[y_variable_summary] = dfg[y_variable_summary]*100

    return df, dfg, c, x_var, color_var

def plot_cost_breakdown(
    object_drilldown,
    df: pd.DataFrame,
    free_y: bool = False,
    units: str = "€/T",
    title: str = None,
):
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    """
    Bars are colored by CURRENT COST (light scale so black text is readable).
    Facets are sorted by total current_cost (desc).
    """
    df = df.copy()    
 
    # --- facet order by total current cost ---
    cost_order = (
        df.groupby("cost")["current_cost"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
 
    # --- labels ---
    df["cost_text"]  = df["current_cost"].map(lambda v: f"{v:,.2f} {units}")
    arrow = np.where(df["pct_cost"].to_numpy() > 0, "▲",
             np.where(df["pct_cost"].to_numpy() < 0, "▼", "•"))
    df["arrow_text"] = [f"{a}{p:.1f}{units}" for a, p in zip(arrow, df["pct_cost"])]
 
    # --- light colorscale (keeps extremes light for black text) ---
    # white → very light blue; adjust last color if you want slightly darker
    light_scale = [
        (0.00, "#ccffcc"),  # light green (most negative)
        (0.50, "#ffffff"),  # white (zero)
        (1.00, "#ffcccc"),  # light red (most positive)
    ]
 
    # color range based on pct_cost
    if len(df):
        cabs = float(np.nanmax(np.abs(df["pct_cost"])))  # max distance from 0
        if cabs == 0:
            cabs = 1.0  # avoid degenerate range if all zeros
    else:
        cabs = 1.0

    cmin = -cabs
    cmax = cabs
 
    if title is None:
        title = f"Current Cost by AB_Grade_ID and Cost Component ({units})"
 
    # --- base bars (color by current_cost) ---
    fig = px.bar(
        df.rename(columns={"cost":object_drilldown}),
        x="AB_Grade_ID",
        y="current_cost",
        facet_col=object_drilldown,
        facet_col_wrap=3,
        color="pct_cost",                 # << color by COST
        color_continuous_scale=light_scale,   # << light palette
        range_color=[cmin, cmax],
        text="cost_text",
        custom_data=["arrow_text", "previous_cost", "pct_cost", "current_cost"],
        category_orders={"cost": cost_order},
        title=title,
    )
 
    # inside text + thin borders; black text stays readable with light scale
    fig.update_traces(
        selector=dict(type="bar"),
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=11, color="black"),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{facetcol} • ID %{x}</b><br>"
            f"Current: %{{y:,.3f}} {units}<br>"
            f"Previous: %{{customdata[1]:,.3f}} {units}<br>"
            "Δ: %{{customdata[2]:.2%}}<extra></extra>"
        ),
        marker_line_color="rgba(60,60,60,0.5)",
        marker_line_width=0.8,
    )
 
    # arrow text just above each bar (colored by sign of pct change)
    for tr in list(fig.data):
        if tr.type != "bar":
            continue
        cd = getattr(tr, "customdata", None)
        if cd is None:
            continue
        cd_list = cd.tolist() if hasattr(cd, "tolist") else list(cd)
        arrow_text = [row[0] for row in cd_list]
        pct_vals   = [row[2] for row in cd_list]
 
        yvals = np.asarray(tr.y, dtype=float)
        pad = 0.03 * (np.nanmax(yvals) if yvals.size else 1.0)
        y_text = (yvals + pad).tolist()
 
        arrow_colors = ["#2ca02c" if p < 0 else "#d62728" if p > 0 else "#666666" for p in pct_vals]
 
        fig.add_trace(go.Scatter(
            x=tr.x,
            y=y_text,
            mode="text",
            text=arrow_text,
            textposition="top center",
            textfont=dict(size=11, color=arrow_colors),
            hoverinfo="skip",
            showlegend=False,
            xaxis=getattr(tr, "xaxis", "x"),
            yaxis=getattr(tr, "yaxis", "y"),
            cliponaxis=False,
        ))
 
    # whitegrid styling
    base_width=800
    apect_ratio=9/16
    fig.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_colorbar_title= f"Overprocessing ({units})" if object_drilldown=="overprocessing" else f"Current cost ({units})" ,  
        xaxis_title="AB_Grade_ID",
        yaxis_title=f"Overprocessing ({units})" if object_drilldown=="overprocessing" else f"Current cost ({units})" ,
        font=dict(size=12, color="#222"),
        margin=dict(l=50, r=20, t=70, b=40),
        coloraxis1_showscale=False,
        autosize=True,
        width=None,
        height=base_width*apect_ratio
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False, tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False, automargin=True)
 
    # optional free y-scale per facet
    if free_y:
        fig.update_yaxes(matches=None)
 
    return fig

def daterange(start_date, end_date, jump="day"):
    from datetime import date, timedelta
    import pandas as pd

    d = start_date
    while d <= end_date:
        yield d
        if jump=="week":
            d += timedelta(days=7)
        elif jump=="month":
            d += pd.DateOffset(months=1)
        elif jump=="hour":
            d += pd.DateOffset(hours=1)
        else:
            d += timedelta(days=1)

def compute_icl(gmm, X_scores):
    import numpy as np

    """
    ICL = BIC + entropy of posterior responsibilities.
    Lower ICL is better.
    """
    bic = gmm.bic(X_scores)
    # responsibilities: shape (n_samples, n_components)
    resp = gmm.predict_proba(X_scores)
    # numerical stability: add tiny epsilon
    eps = 1e-12
    entropy = -np.sum(resp * np.log(resp + eps))
    return bic + entropy

def cluster_stability_ari(X_scores, k, n_boot=20, subsample_frac=0.8, random_state=0, ):
  import numpy as np
  from sklearn.mixture import GaussianMixture
  from sklearn.metrics import adjusted_rand_score

  """
  For a fixed k:
  - Refit GMM n_boot times on random subsamples of the data
  - Predict cluster labels on *all* samples each time
  - Measure average Adjusted Rand Index across all pairs of labelings
  Returns a scalar stability score in [0, 1].
  """
  rng = np.random.RandomState(random_state)
  n_samples = X_scores.shape[0]
  n_sub = int(subsample_frac * n_samples)

  all_labels = []

  for b in range(n_boot):
      # Subsample indices
      idx = rng.choice(n_samples, size=n_sub, replace=False)
      X_sub = X_scores[idx]

      gmm = GaussianMixture(
          n_components=k,
          covariance_type="full",
          random_state=rng.randint(0, 1_000_000)
      )
      gmm.fit(X_sub)

      # Predict labels on ALL samples to make ARI comparable
      labels_full = gmm.predict(X_scores)
      all_labels.append(labels_full)

  # Compute average pairwise ARI
  stability = 0.0
  count = 0
  for i in range(len(all_labels)):
      for j in range(i + 1, len(all_labels)):
          ari = adjusted_rand_score(all_labels[i], all_labels[j])
          stability += ari
          count += 1

  if count > 0:
      stability /= count
  else:
      stability = 0.0

  return stability

def clustering(
    df,
    ycol,
    n_pls_components=3,
    min_clusters=1,
    max_clusters=10,
    n_boot=20,
    subsample_frac=0.8,
    icl=True,
    stability=True,
    min_cluster_size=5,  
    verbose = False
):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    import numpy as np

    X = df.drop(columns=[ycol]).copy()
    y = df[ycol].copy()
 
    # Drop rows with NaNs in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    pls = PLSRegression(n_components=n_pls_components)
    X_pls_scores, _ = pls.fit_transform(X_scaled, y)
 
    bic_values = []
    icl_values = []
    gmm_models = []
    ks = []  # keep track of which k each metric corresponds to
 
    # ---------- Model selection by BIC / ICL ----------
    for k in range(min_clusters, min(max_clusters, len(X_pls_scores)) + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0
        )
        gmm.fit(X_pls_scores)
 
        # If you want to already enforce min_cluster_size in the selection phase,
        # you can uncomment this block:
        #
        if (min_cluster_size is not None) and (k > min_clusters):
            labels_tmp = gmm.predict(X_pls_scores)
            counts = np.bincount(labels_tmp, minlength=k)
            if (counts < min_cluster_size).any():
                if verbose:
                    print(f"Skipping k={k}: some clusters smaller than {min_cluster_size} "
                      f"({counts})")
                continue
 
        gmm_models.append(gmm)
        ks.append(k)
        bic_values.append(gmm.bic(X_pls_scores))
        icl_values.append(compute_icl(gmm, X_pls_scores))
 
    bic_values = np.array(bic_values)
    icl_values = np.array(icl_values)
 
    if icl:
        best_k_icl = ks[np.argmin(icl_values)]
        if verbose:
            print(f"Initial suggested number of clusters by ICL: k = {best_k_icl}")
    else:
        best_k_icl = ks[np.argmin(bic_values)]
 
    # ---------- Stability-based selection ----------
    if stability:
        stability_scores = []
        ks_stab = []
 
        for k in range(max(min_clusters, 2),
                       min(max_clusters, int(subsample_frac * len(X_pls_scores))) + 1):
            stab_k = cluster_stability_ari(
                X_pls_scores,
                k,
                n_boot=n_boot,
                subsample_frac=subsample_frac,
                random_state=0,
            )
            stability_scores.append(stab_k)
            ks_stab.append(k)
            if verbose:
                print(f"k={k}: stability (ARI) = {stab_k:.3f}")
 
        stability_scores = np.array(stability_scores)
        best_k_stab = ks_stab[np.argmax(stability_scores)]
        if verbose:
            print(f"Best k by stability: k = {best_k_stab}")
    else:
        best_k_stab = np.inf
 
    # ---------- Combine criteria ----------
    final_k = min(best_k_icl, best_k_stab)
    if final_k == np.inf:  # if stability was off and best_k_stab = inf
        final_k = best_k_icl
 
    if verbose:
        print(f"Chosen final number of clusters before size check: k = {final_k}")
 
    # ---------- Enforce minimum cluster size in the final solution ----------
    def fit_with_min_size(k_start):
        """
        Try k = k_start, k_start-1, ..., min_clusters
        Return the first GMM for which all clusters have >= min_cluster_size points.
        """
        if min_cluster_size is None:
            # No constraint requested; just use k_start
            gmm = GaussianMixture(
                n_components=k_start,
                covariance_type="full",
                random_state=0
            )
            gmm.fit(X_pls_scores)
            labels = gmm.predict(X_pls_scores)
            return gmm, labels, k_start
 
        for k in range(k_start, min_clusters - 1, -1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=0
            )
            gmm.fit(X_pls_scores)
            labels = gmm.predict(X_pls_scores)
            counts = np.bincount(labels, minlength=k)
 
            if (counts >= min_cluster_size).all():
                if verbose:
                    print(f"Final k after enforcing min_cluster_size={min_cluster_size}: "
                          f"k = {k} (cluster sizes: {counts})")
                return gmm, labels, k
 
            if verbose:
                print(f"k={k} rejected: some clusters smaller than {min_cluster_size} "
                      f"(sizes: {counts})")
 
        raise ValueError(
            f"Could not find any k between {min_clusters} and {k_start} "
            f"with all clusters >= {min_cluster_size} members."
        )
 
    final_gmm, final_labels, final_k = fit_with_min_size(final_k)
    if verbose:
        print(f"Chosen final number of clusters: k = {final_k}")
 
    return final_labels

def shapley_for_pair(row1, row2, cost, features, cost_variable=None,
                     add_unknown=False, unknown_name="unknown"):
    """
    Faster exact Shapley computation using the coalition formula.
 
    row1, row2: pandas Series (or mapping) with at least `features`
    cost: function(row_like) -> float
    features: list of feature names
    cost_variable: name of column with the *true* cost (optional)
    add_unknown: if True, add an 'unknown' residual contribution so that
                 total = true cost difference instead of model cost diff.
    """
    import itertools
    import math
 
    n = len(features)
    idxs = range(n)
 
    # Prebuild baseline/new values as simple dicts (only once)
    x1 = {f: float(row1[f]) for f in features}
    x2 = {f: float(row2[f]) for f in features}
 
    # 1) Evaluate the cost for ALL subsets S of features switched to x2
    v = {}  # key: frozenset of feature indices, value: cost
 
    for r in range(n + 1):
        for S in itertools.combinations(idxs, r):
            S_set = frozenset(S)
            row = x1.copy()
            # switch features in S from x2
            for j in S:
                f = features[j]
                row[f] = x2[f]
            v[S_set] = cost(row)
 
    # 2) Shapley formula over all subsets S ⊆ N\{i}
    contrib = {f: 0.0 for f in features}
    n_fact = math.factorial(n)
 
    for i in idxs:
        fi = features[i]
        # all subsets of N \ {i}
        others = [j for j in idxs if j != i]
 
        for r in range(n):  # |S| from 0 to n-1
            weight = math.factorial(r) * math.factorial(n - r - 1) / n_fact
            for S in itertools.combinations(others, r):
                S_set = frozenset(S)
                S_with_i = frozenset(S + (i,))
                contrib[fi] += weight * (v[S_with_i] - v[S_set])
 
    # 3) Optionally add the "unknown" residual to match true cost difference
    if add_unknown and cost_variable is not None:
        # model delta: cost(x2) - cost(x1)
        model_delta = v[frozenset(idxs)] - v[frozenset()]
        # true delta: row2[cost_variable] - row1[cost_variable]
        true_delta = float(row2[cost_variable]) - float(row1[cost_variable])
        # residual goes into an extra "feature"
        contrib[unknown_name] = true_delta - model_delta
 
    return contrib

def plot_shapley_contribution(shapley_df):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    id_col = "id"
    value_cols = [c for c in shapley_df.columns if c != id_col]
    total_col = value_cols[-1]           # last column = total change
    contrib_cols = value_cols[:-1]       # all previous columns = contributions

    x_labels = contrib_cols + [total_col]
    measure = ["relative"] * len(contrib_cols) + ["total"]

    # 2. Unique ids for facet-like subplots
    ids = shapley_df[id_col].unique()
    n_ids = len(ids)

    fig = make_subplots(
        rows=1,
        cols=n_ids,
        subplot_titles=[str(i) for i in ids]
    )

    # 3. Add one Waterfall trace per id (one facet per id)
    for col_idx, this_id in enumerate(ids, start=1):
        row = shapley_df[shapley_df[id_col] == this_id].iloc[0]   # assuming one row per id

        y_values = row[contrib_cols + [total_col]].values

        fig.add_trace(
            go.Bar(
                name=str(this_id),
                x=x_labels,
                y=y_values
            ),
            row=1,
            col=col_idx
        )

    # 4. Layout
    fig.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig

def to_rgba(color, alpha=0.25):
  import re
  import plotly.graph_objects as go
  """
  Convert any Plotly color string into 'rgba(r,g,b,a)'.
  Handles:
    - named colors: "red"
    - hex: "#FF0000" or "#F00"
    - rgb: "rgb(255,0,0)"
    - rgba: "rgba(255,0,0,0.5)"
    - (optionally) tuples: (r, g, b)
  """
  # If it's already a tuple like (r, g, b)
  if isinstance(color, tuple) and len(color) >= 3:
      r, g, b = color[:3]
      return f"rgba({r},{g},{b},{alpha})"

  c = str(color).strip()

  # Already rgba(...) → just replace alpha
  if c.lower().startswith("rgba"):
      nums = re.findall(r"[\d.]+", c)
      r, g, b = nums[:3]
      return f"rgba({r},{g},{b},{alpha})"

  # rgb(...) → turn into rgba(...)
  if c.lower().startswith("rgb"):
      nums = re.findall(r"[\d.]+", c)
      r, g, b = nums[:3]
      return f"rgba({r},{g},{b},{alpha})"

  # Hex string "#RRGGBB" or "#RGB"
  if c.startswith("#"):
      hex_value = c.lstrip("#")
      # short hex "#F00" → "#FF0000"
      if len(hex_value) == 3:
          hex_value = "".join(ch * 2 for ch in hex_value)
      r = int(hex_value[0:2], 16)
      g = int(hex_value[2:4], 16)
      b = int(hex_value[4:6], 16)
      return f"rgba({r},{g},{b},{alpha})"
  
  # Otherwise, assume a named color like "red"
  # Let Plotly resolve it (often to "rgb(...)"), then recurse
  dummy = go.Figure(data=[go.Scatter(line=dict(color=c))])
  resolved = dummy.data[0].line.color  # typically "rgb(r,g,b)" or a hex
 
def shapley_for_pair_mc(
    row1, row2, cost, features,
    M=500, random_state=None,
    cost_variable=None,
    add_unknown=False,
    unknown_name="unknown"
):
    import random
    rnd = random.Random(random_state) if random_state is not None else random
 
    n = len(features)
 
    # Extract numeric values once
    x1_vals = {f: float(row1[f]) for f in features}
    x2_vals = {f: float(row2[f]) for f in features}
 
    contrib = {f: 0.0 for f in features}
 
    idxs = list(range(n))        # allocate once
    row = dict(x1_vals)          # allocate once
 
    for _ in range(M):
        rnd.shuffle(idxs)
        changed = []
 
        prev_cost = cost(row)
 
        for j in idxs:
            f = features[j]
            v2 = x2_vals[f]
 
            if row[f] == v2:
                continue
 
            changed.append(f)
            row[f] = v2
 
            new_cost = cost(row)
            contrib[f] += (new_cost - prev_cost)
            prev_cost = new_cost
 
        # restore row back to x1
        for f in changed:
            row[f] = x1_vals[f]
 
    invM = 1.0 / M
    for f in features:
        contrib[f] *= invM
 
    if add_unknown and cost_variable is not None:
        model_delta = cost(x2_vals) - cost(x1_vals)
        true_delta = float(row2[cost_variable]) - float(row1[cost_variable])
        contrib[unknown_name] = true_delta - model_delta
 
    return contrib

def make_model_cost(model, feature_fn):
    cols = feature_fn()
    df = pd.DataFrame([[0.0] * len(cols)], columns=cols)  # allocate ONCE
 
    def cost(row):
        # overwrite values in-place (no allocation)
        df.iloc[0] = [row[c] for c in cols]
        return float(model.predict(df)[0])
 
    return cost

def _prediction_models(models_dir):
  import pickle
  import joblib
  from sklearn.linear_model import LinearRegression
  import numpy as np

  models={}
  features={}
  ref_date=pd.to_datetime("2025-2-5").strftime("%Y_%m_%d")
  for property in ["MBS_SCT_MD","MBS_SCT_CD","MBS_Burst","MBS_CMT30"]:
      with open(models_dir / f"{property}_V{ref_date}_retrained0.pkl", 'rb') as f:
          models[property] = joblib.load(f)
  for property in ["MBS_SCT_MD","MBS_SCT_CD","MBS_Burst","MBS_CMT30"]:
      with open(models_dir /  f"{property}_V{ref_date}_retrained0.pkl", 'rb') as f:        
          features[property] = pickle.load(f)
  LR_models={}

  for model_name in ["MBS_SCT_CD","MBS_SCT_MD","MBS_Burst","MBS_CMT30"]:
      lr = LinearRegression()
      coefs_arr = np.asarray(models[model_name].coef_, dtype=float)
      if coefs_arr.ndim == 1:
          n_features = coefs_arr.shape[0]
          lr.coef_ = coefs_arr
          lr.intercept_ = float(models[model_name].intercept_)
      elif coefs_arr.ndim == 2:
          n_targets, n_features = coefs_arr.shape
          lr.coef_ = coefs_arr
          intercept_arr = np.asarray(models[model_name].intercept_, dtype=float)
          if intercept_arr.shape == ():
              # same scalar intercept for all targets
              intercept_arr = np.full(n_targets, float(models[model_name].intercept_))
          elif intercept_arr.shape != (n_targets,):
              raise ValueError("For multi-output, intercept must be length n_targets.")
          lr.intercept_ = intercept_arr
      else:
          raise ValueError("`coefs` must be 1D (single-output) or 2D (multi-output).")

      if len(features[model_name]) != n_features:
          raise ValueError("Length of feature_names must match number of coefficients/features.")

      # Minimal attributes sklearn checks for during predict()
      lr.n_features_in_ = n_features
      lr.feature_names_in_ = np.array(features[model_name], dtype=object)

      # Optional extras (not required for predict, but nice to have)
      lr.rank_ = n_features
      lr.singular_ = np.ones(n_features, dtype=float)
      LR_models[model_name]=lr

  return LR_models

def _model_performance(LR_models, turnup, start_baseline, end_baseline):
  from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
  
  res=[]
  turnup_targets={}
  for ycol in LR_models.keys():
      turnup_targets[ycol]={}
      turnup_baseline=turnup[(turnup.Wedge_Time.dt.date>=start_baseline) & (turnup.Wedge_Time.dt.date<=end_baseline)][list(LR_models[ycol].feature_names_in_) + [ycol,'MBS_Current_reel_ID']]
      turnup_baseline['MBS_Current_reel_ID']=turnup_baseline['MBS_Current_reel_ID'].astype("str")
      y_pred=LR_models[ycol].predict(turnup_baseline.drop([ycol,'MBS_Current_reel_ID'],axis=1))
      y_test=turnup_baseline[ycol]
      res.append(pd.DataFrame([{"property":ycol,"MAE":round(mean_absolute_error(y_test, y_pred),3),"R2":round(r2_score(y_test, y_pred),2),"RMSE":round(root_mean_squared_error(y_test, y_pred),3),"MAPE":round(mean_absolute_percentage_error(y_test, y_pred)*100,1),"target":"baseline"}]))
      turnup_targets[ycol]["baseline"]=turnup_baseline
      for d in list(daterange(pd.to_datetime("2025-10-1"), pd.to_datetime("2026-1-1"), jump="month"))[:-1]:
          df=turnup[(turnup.Wedge_Time.dt.date>=d.date()) & (turnup.Wedge_Time.dt.date < (d + pd.DateOffset(months=1)).date())][list(LR_models[ycol].feature_names_in_) + [ycol,'MBS_Current_reel_ID']]
          df['MBS_Current_reel_ID']=df['MBS_Current_reel_ID'].astype("str")
          if len(df)>0:
              turnup_targets[ycol][d.strftime("%Y%b")]=df
              y_pred=LR_models[ycol].predict(df.drop([ycol,'MBS_Current_reel_ID'],axis=1))
              y_test=df[ycol]
              res.append(pd.DataFrame([{"property":ycol,"MAE":round(mean_absolute_error(y_test, y_pred),3),"R2":round(r2_score(y_test, y_pred),2),"RMSE":round(root_mean_squared_error(y_test, y_pred),2),"MAPE":round(mean_absolute_percentage_error(y_test, y_pred)*100,1),"target":d.strftime("%Y%b")}]))
  res = pd.concat(res,axis=0)
  
  return res, turnup_targets

def _rolling_lab_performance(turnup, strength_property, window):
  import numpy as np

  if "Wedge_Time" in turnup.columns:     
      turnup = turnup.set_index("Wedge_Time")
  turnup.sort_index(inplace=True)
  perc = 2.5

  def p_low(x):
      return np.percentile(x, perc)
  
  def p_high(x):
      return np.percentile(x, 100-perc)
  
  rolling_stats = (
      turnup
      .groupby("AB_Grade_ID")[[strength_property]]
      .rolling(window=window, min_periods=1)  
      .agg(["mean", "std", "median", p_low, p_high])
  )

  return rolling_stats[strength_property], strength_property

def _lab_performance(LR_models, turnup, start_baseline, end_baseline):
  import pandas as pd

  res=[]
  for grade in ["3200115", "6010085", "6010100", "6010120"]:   
      for ycol in LR_models.keys():
          turnup_baseline=turnup[(turnup.Wedge_Time.dt.date>=start_baseline) & (turnup.Wedge_Time.dt.date<=end_baseline) & (turnup.AB_Grade_ID==grade)][list(LR_models[ycol].feature_names_in_) + [ycol]]                        
          res.append(pd.DataFrame([{"property":ycol,"grade":grade,"mean":turnup_baseline[ycol].mean(),"sd":turnup_baseline[ycol].std(),"target":"baseline"}]))        
      for d in list(daterange(pd.to_datetime("2025-10-1"), pd.to_datetime("2026-1-1"), jump="month"))[:-1]:
          for ycol in LR_models.keys():
              df=turnup[(turnup.Wedge_Time.dt.date>=d.date()) & (turnup.Wedge_Time.dt.date < (d + pd.DateOffset(months=1)).date()) & (turnup.AB_Grade_ID==grade)][list(LR_models[ycol].feature_names_in_) + [ycol]]
              if len(df)>0:                
                  res.append(pd.DataFrame([{"property":ycol,"grade":grade,"mean":df[ycol].mean(),"sd":df[ycol].std(),"target":d.strftime("%Y%b")}]))            
          
  res = pd.concat(res,axis=0)
  return res

def _process_features(raw_data):
    process_features=["Speed","Production_Rate__T/h_", "Current_reel_weight","Current_reel_width","Current_reel_length","Current_basis_weight","Current_reel_moisture_average(reel)","Actual_moisture","Current_reel_dry_average","BSW_2_sigma","Mois_Size_Press_2_sigma", "Mois_2_sigma", "Contact_pressure_reel_holders", "Reel_discharge_pressure", "Draw_meaurement_pope_reel_FS", "Draw_measurement_pope_reel_DS", "Total_draw_measurement_pope_reel", "Contact_pressure_secondary_arm_OS", "Contact_pressure_secondary_arm_DS","Draw_AD7-PR","Draw_SS-AD6", "Draw_AD6-AD7", "DG4_Temperature_Inlet_Air", "Cylinder_36_steam_pressure", "Cylinder_36_differential_pressure", "DG5_Temperature_Inlet_Air", "Cylinder_37_steam_pressure", "Cylinder_38_differential_pressure", "Cylinder_37_differential_pressure", "Cylinder_38_steam_pressure", "Cylinder_39_steam_pressure", "Cylinder_36-39_steam_pressure", "Cylinder_39_differential_pressure", "Cylinder_40-53_differential_pressure", "Cylinder_40-53_steam_pressure", "Web_tension_AD6", "Steam_flow_to_AfterDryers", "Steam_flow_to_AfterDryers_sqm","Steam_flow_to_AfterDryers_index","DG4_Moisture_content_Outlet_Air", "DG5_Moisture_content_Outlet_Air", "DG_4-5_zero_level", "DG_4-5_zero_level_(Pa)", "DG5_Ventilator_Revolution_Output", "DG4_Ventilator_Revolution_Output", "AD7_fabric_tension_bottom", "AD6_fabric_tension", "AD6_speed", "AD7_speed_top", "AD7_speed_bottom", "Draw_PD5-SS", "SpeedSizer_Linepressure_DS", "SpeedSizer_Linepressure_FS", "Starch_uptake_by_paper_Bottom_Roll__g/m2_", "Starch_uptake_by_paper_Top_Roll__g/m2_", "Pressure_of_Starch_flow_Speedsizer_Bottom_Roll~^0", "Starch_consumption_Bottom___m³/h_", "Starch_consumption_Top__m³/h_", "Consistency_starch_main_line", "Pressure_of_Starch_flow_Speedsizer_Top_Roll", "Starch_application_FW_in_ml", "Starch_application_BW_in_ml", "Starch_Top_Roll__ml/m²_", "Starch_Bottom_Roll__ml/m²_", "AirTurn_Temperature", "Airturn_pillow_pressure", "Rod_Pressure_Bottom_Roll", "Rod_clamping_pressure_Bottom_Roll", "Air_pressure_of_rod_clamping_hose_Bottom_Roll", "Rod_clamping_pressure_Top_Roll", "Rod_pressure_Top_Roll", "Speed_Size_Press", "Air_pressure_of_rod_clamping_hose_Top_Roll", "Steam_flow_to_starch_kitchen", "Enzyme_flow_Slurry_1", "Slurry_1_dosing_screw_rotation", "Slurry_1_level", "Dilution_water_flow_slurry_1", "Flow_slurry_1_to_reactor", "Slurry_1_pumping_to_reactor", "Enzyme_flow_Slurry_2", "Slurry_2_dosing_screw_rotation", "Slurry_2_level", "Dilution_water_flow_slurry_2", "Flow_slurry_2_to_reactor", "Slurry_2_pumping_to_reactor", "Enzyme_flow_Slurry_3", "Slurry_3_dosing_screw_rotation", "Slurry_3_level", "Dilution_water_flow_slurry_3", "Flow_slurry_3_to_reactor", "Slurry_3_pumping_to_reactor", "Pressure_slurry_pipe_3", "Pressure_slurry_main_pipe", "Reactor_steam_pressure", "Reactor_Temperature", "Reactor_level", "Starch_flow_to_inactivation", "Pressure_to_inactivation", "Temperature_inactivation", "Pressure_after_inactivation", "Storage_tank_level", "Storage_tank_temperature", "Dilution_water_storage_tank", "Pressure_starch_main_line", "Flow_starch_main_line_to_working_tank_1~^0", "Dilution_water_working_tank_1", "Temperature_starch_working_tank_1", "Level_storage_tank_2~^0", "Differential_pressure_filter_storage_tank_1", "Flow_starch_main_line_to_working_tank_2~^0", "Dilution_water_working_tank_2", "Temperature_starch_working_tank_2", "Level_storage_tank_2", "Differential_pressure_filter_storage_tank_2", "Draw_PS-PD1", "Draw_PD2-PD3", "Draw_PD4-PD5", "Draw_PD3-PD4", "Draw_PD1-PD2", "Cylinder_1_differential_pressure", "Cylinder_1_steam_pressure", "Cylinder_2_steam_pressure", "Cylinder_3_differential_pressure", "Cylinder_2_differential_pressure", "Cylinder_3_steam_pressure", "Cylinder_4_differential_pressure", "Cylinder_5_steam_pressure", "Cylinder_4_steam_pressure", "Cylinder_5_differential_pressure", "Cylinder_1-5_steam_pressure", "Cylinder_1-5_fresh_steam", "Cylinder_6-15_differential_pressure", "Cylinder_6-15_steam_pressure", "Cylinder_1-5_steam_temperature", "Cylinder_14_differential_pressure", "Cylinder_16-24_steam_pressure", "Cylinder_25-35_steam_pressure", "Moisture_out_of_PreDryer", "Cylinder_6-35_differential_pressure", "Inlet_Air_2_Temperature", "Inlet_Air_1_Temperature", "Steam_flow_to_PreDryers", "Steam_flow_to_PreDryers_sqm","Electricity_index","Electricity_sqm","Steam_flow_to_PreDryers_index","DG_1-3_zero_point", "DG1_Ventilator_Revolution_Output", "DG1_Moisture_content_Outlet_Air", "DG1_temperature_Inlet_Air", "DG1_zero_point", "DG2_Moisture_content_Outlet_Air", "DG2_temperature_Inlet_Air", "DG2_Ventilator_Revolution_Output", "DG2-3_zero_point", "DG3_Moisture_content_Outlet_Air", "DG3_temperature_Inlet_Air", "DG3_Ventilator_Revolution_Output", "PD1_Fabric_tension", "PD2_Fabric_tension", "PD5_fabric_tension_bottom", "PD4_fabric_tension", "PD4_fabric_tension_bottom", "PD5_fabric_tension_top", "ProRelease_1_LoVac", "ProRelease_2_LoVac", "ProRelease_3_LoVac", "ProRelease_4_LoVac", "ProRelease_5_LoVac", "ProRelease_6_LoVac", "ProRelease_7_LoVac", "Compressor_1_outgoing_pressure", "Compressor_2_outgoing_pressure", "Speed_PD4_bottom", "Speed_PD4_top", "Speed_PD5_top", "Speed_PD5_bottom", "Speed_PD1", "Speed_PD3", "Speed_PD2", "Draw_WS-PS", "PickUp_Tension", "Vacuum_Zone_1_PickUp", "Vacuum_presszone_of_suction-press_roll", "Vacuum_holding/pre_positions_of_suction-press_roll", "Vacuum_uhle-box_Pick-Up", "Vacuum_uhle-box_bottom_felt", "Dewatering_Suction_Press_Roll", "Dewatering_Shoe_press", "Dewatering_Pick-Up", "Linepressure_1st_press_FS__bar_", "Linepressure_2nd_press_FS__bar_", "Linepressure_1st_press_DS__bar_", "Linepressure_2nd_press_DS__bar_", "Linepressure_shoe_press__bar_", "Uhle_box_1_flow___l/min_", "Uhle_box_2_flow___l/min_", "Top_Felt_Tension", "Bottom_Felt_Tension", "Total_Dewatering_Pick-Up", "Dewatering_First_Press_Roll", "Total_Dewatering_Bottom_Felt", "Bottom_wire_tension", "Vacuum_top_wire_suction_box_zone_1", "Vacuum_top_wire_suction_box_zone_2", "Vacuum_formning_roll", "Dewatering_top_wire_suction_box_zone_1", "Dewatering_Jet_channel", "Dewatering_top_wire_suction_box_zone_2", "Vacuum_suction_box_9", "Vacuum_wet_suction_box", "Vacuum_sheet_seperator_box", "Vacuum_suction_box_10", "Vacuum_suction_box_11", "Vacuum_wire_suction_box_1", "Vacuum_wire_suction_box_2", "Consistency_white_water", "White_water_temperature", "Conductivity_white_water_B46", "Top_wire_tenstion", "pH_measurement_white_water_B41", "Forming_Wire_Speed", "Fresh_water_main_pipe_pressure", "Differential_pressure_retention_aid_filter", "Inlet_pressure_TrumpJet_station_A1-A4", "Differential_pressure_A1-A4_between_stock_and_chemical", "Inlet_pressure_TrumpJet_station_C1-C4", "Differential_pressure_C1-C4_between_stock_and_chemical", "Pre_pressure_TrumpJet", "Stock_to_TrumpJet", "Mixing_water_to_TrumpJet", "Pressure_of_pressure_amplifying_pump_for_retention_aid_injection", "Flow_of_pressure_amplyfing_pump", "Bentonite_filter_differential_pressure", "Bentonite_flow_station_B1_TrumpJet", "Bentonite_flow_station_B2_TrumpJet", "Bentonite_flow_station_B4_TrumpJet", "Bentonite_flow_station_B3_TrumpJet", "Pressure_Bentonite_TrumpJet", "Differential_pressure_B1-B4_between_stock_and_chemical", "Backflow_cross-flow_distributor_dilution_water~^0", "Backflow_cross-flow_distributor_stock~^0", "Headbox_pressure", "Headbox_pressure_DS", "Headbox_pressure_FS", "Headbox_total_flow", "Lip_settings", "Jet/wire_ratio", "Thick_Stock_Consistency__%_", "Thick_Stock_Flow__l/min_", "Machine_chest_consistency", "Dilution_water_deculator_pressure", "Sorter_stock_power", "Dilution_water_deculator_temperature", "Stock_deculator_pressure", "Stock_deculator_temperature", "Dilutionwater_pump_power", "Stock_pump_power", "Short_fibre_flow", "Long_fibre_flow", "Short_fibre_B06_consistency", "Long_fibre_consistency_B07", "Dry_broke_flow", "Wet_broke_flow", "pH-Messung_Verdünnungswasser__2..12_pH_", "Wet_broke_consistency", "Sludge_addition_to_stock", "Free_gas_before_dilution_water_deculator_measurement_1~^0", "Free_gas_before_dilution_water_deculator_measurement_2", "Dissolved_gas_before_dilution_water_deculator", "Free_gas_after_stock_deculator~^0", "Dissolved_gas_after_stock_deculator_measurement_1", "Dissolved_gas_after_stock_deculator_measurement_2", "Ash_measurement_HC-line", "Stock_Valve_Opening_From_Machine_Chest", "Dry_Broke_Consistancy__%_", "Pulper_consistency", "ATS1_power", "ATS1_differential_pressure", "ATS1_light_reject_flow", "ATS2_differential_pressure", "ATS2_light_reject_flow", "Combisorter_1_power", "Combisorter_2_power", "Combisorter_3_power", "Contaminex_1_power", "Contaminex_2_power", "Contaminex_3_power", "Multifractor_1_long_fibre_flow", "Multifractor_1_short_fibre_flow", "Multifractor_1_Long_fibre_fraction", "Multifractor_1_consistency", "Multifractor_2_long_fibre_flow", "Multifractor_2_short_fibre_flow", "Multifractor_2_long_fibre_fraction", "Multifractor_2_consistency", "Multifractor_3_long_fibre_flow", "Multifractor_3_short_fibre_flow", "Multifractor_3_long_fibre_fraction", "Multifractor_3_consistency", "LF_screen_1_power", "LF_screen_1_accept_flow", "LF_screen_2_inlet_consistency", "LF_screen_2_power", "LF_screen_3_inlet_consistency", "LF_screen_3_power", "LF_screen_1_reject_flow", "LF_screen_2_accept_flow", "LF_screen_2_reject_flow", "LF_screen_3_reject_flow", "LF_screen_3_accept_flow", "Steam_flow_from_power_plant_to_PM", "Steam_flow_to_heat_exchangers", "Steam_flow_to_hall_heating", "Steam_flow_to_white_water_heating", "Steam_flow_to_steam_box", "Steam_flow_to_PM", "Pressure_main_steam_line", "Steam_temperature_for_PM", "Steam_pressure_for_PM", "Steam_energy_from_power_plant_to_paper_plant", "Condensate_energy_from_paper_plant_to_power_plant", "Condensate_conductivity", "Total_condensate_flow", "Freshwater_warm__l/min_", "Freshwater_retention__l/min_", "Freshwater_to_Machine_Tank__l/min_", "Freshwater_pressure_from_the_city__bar_","Starch_uptake__g/m2_","ratio_starch","Fibre_usage__T/h_","MC_SF_LF_Demand",'Flow_starch_main_line_to_working_tank',"concentration_starch_working_tank_1","concentration_starch_working_tank_2","SCT_CD_index"]
    return [v for v in process_features if v in raw_data.columns.to_list()]

def _cost_features(raw_data):
    cost_features = [_agg_cost_label(), 'Combined_cost__€/T_','Fibre_cost__€/T_','Steam__€/T_','Electricity__€/T_','Starch__€/T_','Sizing_Agent__€/T_','Combined_cost_current_€/h_',"Steam_current__€/h_","Electricity_current__€/h_","Starch_current__€/h_","Overprocessing_cost__€/T_"]
    return [v for v in cost_features if v in raw_data.columns.to_list()]

def _quality_features():
    return ["MBS_SCT_MD","MBS_SCT_CD","MBS_Burst","MBS_CMT30","Overprocessing_percentage"]

def _component_features(raw_data):
    component_features = ["Retention_Aid_mass_flow__g/T_", "Bentonite_1_mass_flow__g/T_", "Bentonite_2_mass_flow__g/T_", "Fixative_2_mass_flow__g/T_", "Starch_mass_flow__kg/T_", "Act_Deaerator_mass_flow__g/T_", "Defoamer_mass_flow__g/T_", "Sizing_Agent__g/T_", "Dry_Strength_Agent_mass_flow__kg/T_", "Natriumhydroxide_mass_flow__g/T_", "CO2_mass_flow__g/T_", "Electricity__kWh/T_", "Steam__kWh/T_", "Fibre_usage__T/T_","Steam__kW"]
    return [v for v in component_features if v in raw_data.columns.to_list()]

def _costs_to_consider():
    return ['Steam__€/T_','Electricity__€/T_','Starch__€/T_','Sizing_Agent__€/T_']

def _costs_to_consider2():
    return ['Steam__€/T_','Electricity__€/T_','Starch__€/T_','Chemicals__€/T_','Fibre_cost__€/T_']

def _agg_cost_label():
    return "Aggregated_cost__€/T_"

def _agg_cost_label2():
    return "Combined_cost__€/T_"

def _overprocessing_vars():
    return ['Overprocessing_SCT_CD', 'Overprocessing_Burst', 'Overprocessing_CMT30']
    
def _cost_influencers_1():
    return ['Production_Rate__T/h_','Speed','Actual_moisture',
        'Moisture_out_of_PreDryer','Lip_settings','Draw_PS-PD1',
        'Draw_PD4-PD5','Draw_PD3-PD4','Draw_PD1-PD2',
        'Draw_PD5-SS','Jet/wire_ratio','Vacuum_uhle-box_Pick-Up']
    
def _cost_influencers_2():
    return ['Dewatering_top_wire_suction_box_zone_2','Linepressure_1st_press_FS__bar_','Linepressure_1st_press_DS__bar_',
            'Linepressure_2nd_press_FS__bar_','Linepressure_2nd_press_DS__bar_','Linepressure_shoe_press__bar_',
            'Vacuum_suction_box_10','Consistency_white_water','White_water_temperature',
            'Conductivity_white_water_B46','pH_measurement_white_water_B41','Current_basis_weight']
        
def _cost_influencers_3():
    return ["Starch_uptake__g/m2_",'Starch_uptake_by_paper_Bottom_Roll__g/m2_','Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Consistency_starch_main_line',"concentration_starch_working_tank_1","concentration_starch_working_tank_2",
        'Rod_Pressure_Bottom_Roll','Rod_pressure_Top_Roll',
        'Bentonite_1_mass_flow__g/T_','Bentonite_2_mass_flow__g/T_','Retention_Aid_mass_flow__g/T_']

def _cost_influencers_4():
    return ['Steam_flow_to_PreDryers','Steam_flow_to_AfterDryers']

def _strength_influencers():
    return ["Starch_uptake__g/m2_",'Starch_uptake_by_paper_Bottom_Roll__g/m2_','Starch_uptake_by_paper_Top_Roll__g/m2_',
        'Consistency_starch_main_line',"concentration_starch_working_tank_1","concentration_starch_working_tank_2",
        'Rod_Pressure_Bottom_Roll','Rod_pressure_Top_Roll','Current_basis_weight',
        "Jet/wire_ratio"
        ]

def _strength_df(grade_data_process_clustered, setpoint_df, igrade):
  sp=setpoint_df[(setpoint_df.AB_Grade_ID==igrade)].drop("AB_Grade_ID",axis=1).pivot(index=["property"],columns="variable",values="value").reset_index()
  quality_df_=grade_data_process_clustered[['MBS_Current_reel_ID','target'] + _quality_features()].copy()
  quality_df_=quality_df_.rename(columns={"target":"cluster"})
  
  quality_df_res=pd.merge(pd.melt(quality_df_,id_vars=['MBS_Current_reel_ID','cluster'],var_name="property").dropna(),sp, on=["property"],how="left").dropna()
  return quality_df_res

def _acceptable_clusters(grade_data_process_clustered, setpoint_df, igrade, valid_pct):
    import numpy as np

    sp=setpoint_df[(setpoint_df.AB_Grade_ID==igrade)].drop("AB_Grade_ID",axis=1).pivot(index=["property"],columns="variable",values="value").reset_index()
    quality_df_=grade_data_process_clustered[["cluster"] + _quality_features()].copy()

    percentile_name=f"P{round(100-valid_pct)}"
    quality_df_res=pd.merge(pd.melt(quality_df_[_quality_features()+["cluster"]],id_vars=["cluster"],var_name="property").dropna(),sp, on=["property"],how="left").groupby(["cluster", "property"]).agg(percentile=("value", lambda x: np.percentile(x, 100-valid_pct)), min_value=("min", lambda x: min(x)), count=("value", lambda x: len(x))).reset_index().rename(columns={"percentile":percentile_name})
    quality_df_res[percentile_name]=round(quality_df_res[percentile_name],2)
    quality_df_res["valid"]=np.where(quality_df_res[percentile_name]>=quality_df_res["min_value"],"ok","bad")
    quality_df_res.dropna(inplace=True)
    quality_df_res["n_total"]=quality_df_res.groupby(["cluster"])["valid"].transform(lambda x : len(x))

    acceptable_res=quality_df_res[(quality_df_res[percentile_name]>=quality_df_res.min_value) & (quality_df_res["count"]>1)][["cluster","property","n_total"]]
    #acceptable_res=quality_df_res[(quality_df_res[percentile_name]>=quality_df_res.min_value)][["cluster","property","n_total"]]
    acceptable_res_total=acceptable_res.groupby(["cluster"]).agg(n_valid =("n_total", "size"), n_total =("n_total", "first")).reset_index()
    acceptable_res_total=acceptable_res_total[(acceptable_res_total.n_valid>=acceptable_res_total.n_total)][["cluster"]]
    return acceptable_res_total.cluster.to_list(),quality_df_res

def _quality_limits(setpoint_df, igrade, iclustered_quality):
        return setpoint_df[(setpoint_df.AB_Grade_ID==igrade) & (setpoint_df.property==iclustered_quality)]

def _pls_transformation(raw_data, process_data, igrade):
  from sklearn.preprocessing import StandardScaler
  from sklearn.cross_decomposition import PLSRegression

  # ---- config ----
  n_components = 4
  y_col = _agg_cost_label()
  X_cols = _process_features(raw_data)

  # ---- filter data for selected grade ----
  df_all = process_data
  df_features = df_all[df_all["AB_Grade_ID"] == igrade].copy()

  if df_features.empty:
      # Return empty structures gracefully
      return (pd.DataFrame(columns=[f"PLS {i}" for i in range(1, n_components + 1)]),
              pd.DataFrame(index=X_cols, columns=[f"PLS {i}" for i in range(1, n_components + 1)]),
              df_features)

  # ---- X / y preparation (numeric + median impute) ----
  X = df_features[X_cols].apply(pd.to_numeric, errors="coerce")
  y = pd.to_numeric(df_features[y_col], errors="coerce")

  X = X.fillna(X.median(numeric_only=True))
  y = y.fillna(y.median())

  # ---- scale X and y (PLS prefers centered/scaled) ----
  x_scaler = StandardScaler()
  y_scaler = StandardScaler()

  Xs = x_scaler.fit_transform(X.values)
  ys = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

  # ---- choose safe number of components ----
  max_comp = int(min(Xs.shape[0] - 1, Xs.shape[1]))  # <= min(n_samples-1, n_features)
  if max_comp < 1:
      # Not enough data to fit PLS; return empties with correct shapes
      comp_names = [f"PLS {i}" for i in range(1, n_components + 1)]
      return (pd.DataFrame(columns=comp_names),
              pd.DataFrame(index=X_cols, columns=comp_names),
              df_features)

  n = min(n_components, max_comp)

  # ---- fit PLS ----
  pls = PLSRegression(n_components=n, scale=False)  # already scaled
  pls.fit(Xs, ys)

  # ---- outputs to mirror your PCA return signature ----
  comp_names = [f"PLS {i}" for i in range(1, n + 1)]

  # Scores (X-scores): shape (n_samples, n_components)
  df_scores = pd.DataFrame(pls.x_scores_, columns=comp_names, index=df_features.index).reset_index(drop=True)

  # Loadings (X-loadings): shape (n_features, n_components)
  df_loadings = pd.DataFrame(pls.x_loadings_, columns=comp_names, index=X_cols)

  # (Optional) If you want predicted cost back in original units, uncomment:
  # y_pred = y_scaler.inverse_transform(pls.predict(Xs)).ravel()
  # df_features = df_features.copy()
  # df_features["Aggregated_cost_pred"] = y_pred

  return df_scores, df_loadings, df_features

def _principal_component_scoring(raw_data, process_data, igrade, variable_x_scoring):    
    df_scores, df_loadings, df_features = _pls_transformation(raw_data, process_data, igrade)
    pc = df_loadings[variable_x_scoring].to_frame("weight")
    pc.index.names=["feature"]
    pc=pc.reset_index()
    pc["abs_weight"]=abs(pc["weight"])
    return pc[pc["abs_weight"]>pc["abs_weight"].max()*0.8].sort_values("weight",ascending=False)

def _principal_components_features_scoring(raw_data, process_data, igrade, variable_x_scoring):
    pc=_principal_component_scoring(raw_data, process_data,  igrade, variable_x_scoring)
    return pc.feature.to_list()

def _principal_component_oprange(raw_data, process_data, igrade, principal_oprange):
  if principal_oprange=="cost influencers 1":
      return pd.DataFrame({"feature":_cost_influencers_1()})
  elif principal_oprange=="cost influencers 2":
      return pd.DataFrame({"feature":_cost_influencers_2()})
  elif principal_oprange=="cost influencers 3":
      return pd.DataFrame({"feature":_cost_influencers_3()})
  elif principal_oprange=="cost influencers 4":
      return pd.DataFrame({"feature":_cost_influencers_4()})
  elif principal_oprange=="strength influencers":
      return pd.DataFrame({"feature":_strength_influencers()})
  elif principal_oprange=="all cost influencers":
      return pd.DataFrame({"feature":_cost_influencers_1() + _cost_influencers_2() + _cost_influencers_3()})
  else:    
      df_scores, df_loadings, df_features = _pls_transformation(raw_data, process_data, igrade)
      pc = df_loadings[principal_oprange].to_frame("weight")
      pc.index.names=["feature"]
      pc=pc.reset_index()
      pc["abs_weight"]=abs(pc["weight"])
      
      return pc[pc["abs_weight"]>pc["abs_weight"].max()*0.8].sort_values("weight",ascending=False)
  
def _principal_components_features_oprange(raw_data, process_data, igrade, principal_oprange):
  pc=_principal_component_oprange(raw_data, process_data, igrade, principal_oprange)
  return pc.feature.to_list()

def _principal_component_pca(raw_data, process_data, igrade, principal_pca):
  df_scores, df_loadings, df_features = _pls_transformation(raw_data, process_data, igrade)
  pc = df_loadings[principal_pca].to_frame("weight")
  pc.index.names=["feature"]
  pc=pc.reset_index()
  pc["abs_weight"]=abs(pc["weight"])
  return pc[pc["abs_weight"]>pc["abs_weight"].max()*0.8].sort_values("weight",ascending=False)

def _principal_components_features_pca(raw_data, process_data, igrade, principal_pca):
  pc=_principal_component_pca(raw_data, process_data, igrade, principal_pca)
  return pc.feature.to_list()

def _principal_components(raw_data, process_data, igrade):
  df_scores, df_loadings, df_features = _pls_transformation(raw_data, process_data, igrade)
  return df_loadings.columns.to_list()

def impute_outside_limits_with_grade_median(
    df: pd.DataFrame,
    var: str,
    low: float,
    high: float,
    grade_col: str = "AB_Grade_ID",
    treat_na_as_bad: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Impute values of `var` that fall outside [low, high] (and optionally NaNs)
    with the median of `var` within each grade group.
 
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    var : str
        Name of the variable/column to check and impute.
    low, high : float
        Inclusive bounds. Values < low or > high are imputed.
    grade_col : str
        Column used to group grades (default: "AB_Grade_ID").
    treat_na_as_bad : bool
        If True, NaNs are also imputed with grade median.
    inplace : bool
        If True, modifies df in place and returns it. Otherwise returns a copy.
 
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values in `var`.
    """
    if var not in df.columns:
        raise KeyError(f"Column '{var}' not found in df.")
    if grade_col not in df.columns:
        raise KeyError(f"Grade column '{grade_col}' not found in df.")
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high}).")
 
    out = df if inplace else df.copy()
 
    # Ensure numeric; non-numeric become NaN (and may be imputed if treat_na_as_bad=True)
    out[var] = pd.to_numeric(out[var], errors="coerce")
 
    # Grade-level median (ignores NaN by default)
    grade_median = out.groupby(grade_col)[var].transform("median")
 
    mask_bad = (out[var] < low) | (out[var] > high)
    if treat_na_as_bad:
        mask_bad = mask_bad | out[var].isna()
 
    out.loc[mask_bad, var] = grade_median[mask_bad]
    return out

def _turnup_data(local, bucket, fs, setpoint_df, steam_null):
  from datetime import datetime
  import numpy as np

  if local:

      turnup=[]

      for time_ref in ["2025-06-01","2025-07-01","2025-08-01","2025-09-01","2025-10-01","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01"]:
          #print(time_ref)
          turnup.append(pd.read_parquet(f"./data/costimier_turnup_{time_ref}.parquet",engine="fastparquet"))
      turnup=pd.concat(turnup,axis=0).copy()
  else:            
      prefix = "turnup"
      paths = [
          f"s3://{bucket}/{prefix}/costimier_turnup_{d:%Y-%m-%d}.parquet"
          for d in daterange(datetime(2025, 6, 1), datetime.now(), jump="month")
      
      ]
      existing_paths = [p for p in paths if fs.exists(p)]
      
      turnup = pd.concat(
          (                    
              pd.read_parquet(p, filesystem=fs)   
              .assign(**{
                  c: lambda df, c=c: df[c].astype("float32")
                  for c in pd.read_parquet(p, filesystem=fs).select_dtypes(include=["number"]).columns
              })
              for p in existing_paths
          ),
          ignore_index=True,
          sort=False
      )


  for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
      turnup[v]=turnup[v].astype("float32")

  turnup['MBS_Current_reel_ID']=turnup['MBS_Current_reel_ID'].astype(int)
  turnup['AB_Grade_ID']=turnup['AB_Grade_ID'].astype(int)
  turnup['Wedge_Time']=pd.to_datetime(turnup['Wedge_Time'])

  
  turnup = turnup.assign(
      grammage = turnup["AB_Grade_ID"].mod(1000),
      paper_type = (turnup["AB_Grade_ID"] // 1000).astype("string")
  )

  turnup = turnup[turnup.grammage.isin([85,90,95,100,110,115,120,125,130,135])]
  turnup = turnup[turnup.paper_type.isin(["3200", "6010", "3300"])]

  paper_type = pd.get_dummies(turnup["paper_type"])
  turnup = pd.concat([turnup, paper_type], axis=1)

  for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
      if any(e in v.lower() for e in ["€","g/m2","g/t","mbs_","t/t","level","consistency","temperature","l/min","fibre_fraction"]):
          turnup.loc[turnup[v] <= 0, v] = np.nan

  for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
      if any(e in v.lower() for e in ["power","flow"]):
          turnup.loc[turnup[v] < 0, v] = np.nan

  for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
      if any(e in v.lower() for e in ["vacuum"]):
          turnup.loc[turnup[v] > 0, v] = np.nan

  #for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
  #    if any(e in v.lower() for e in ["mbs_sct"]):
  #        turnup.loc[turnup[v] > 3, v] = np.nan

  for v in turnup.drop(['MBS_Current_reel_ID',"AB_Grade_ID","Wedge_Time"],axis=1).columns.to_list():
      if any(e in v.lower() for e in ["moisture"]):
          turnup.loc[turnup[v] < 5, v] = np.nan

  turnup.loc[turnup["Starch_uptake_by_paper_Bottom_Roll__g/m2_"] > 4, "Starch_uptake_by_paper_Bottom_Roll__g/m2_"] = np.nan
  turnup.loc[turnup["Starch_uptake_by_paper_Top_Roll__g/m2_"] > 4, "Starch_uptake_by_paper_Top_Roll__g/m2_"] = np.nan
  turnup.loc[turnup["LF_screen_1_accept_flow"] < 600, "LF_screen_1_accept_flow"] = np.nan
  turnup.loc[turnup["LF_screen_1_accept_flow"] > 1000, "LF_screen_1_accept_flow"] = np.nan
  turnup.loc[turnup["Steam__€/T_"] > 100, "Steam__€/T_"] = np.nan
  turnup.loc[turnup["Steam__kWh/T_"] > 2000, "Steam__kWh/T_"] = np.nan
  turnup.loc[turnup["Electricity__kWh/T_"] > 500, "Electricity__kWh/T_"] = np.nan
  turnup.loc[turnup["Starch_consumption_Top__m³/h_"] > 2, "Starch_consumption_Top__m³/h_"] = np.nan
  turnup.loc[turnup["Starch_application_BW_in_ml"] > 30, "Starch_application_BW_in_ml"] = np.nan
  turnup.loc[turnup["Starch_flow_to_inactivation"] > 1e8, "Starch_flow_to_inactivation"] = np.nan
  turnup.loc[turnup["Flow_starch_main_line_to_working_tank_2~^0"] > 8, "Flow_starch_main_line_to_working_tank_2~^0"] = np.nan
  turnup.loc[turnup["Dilution_water_working_tank_2"] > 6, "Dilution_water_working_tank_2"] = np.nan
  turnup.loc[turnup["Pulper_consistency"] > 20, "Pulper_consistency"] = np.nan
  turnup.loc[turnup["Combisorter_1_power"] > 150, "Combisorter_1_power"] = np.nan
  turnup.loc[turnup["Combisorter_2_power"] > 100, "Combisorter_2_power"] = np.nan
  turnup.loc[turnup["Contaminex_1_power"] > 100, "Contaminex_1_power"] = np.nan
  turnup.loc[turnup["Contaminex_2_power"] > 100, "Contaminex_2_power"] = np.nan
  turnup.loc[turnup["Contaminex_3_power"] > 100, "Contaminex_3_power"] = np.nan
  turnup.loc[turnup["LF_screen_1_power"] > 150, "LF_screen_1_power"] = np.nan
  turnup.loc[turnup["LF_screen_1_power"] < 100, "LF_screen_1_power"] = np.nan
  turnup.loc[turnup["LF_screen_2_power"] > 100, "LF_screen_2_power"] = np.nan
  turnup.loc[turnup["LF_screen_2_power"] < 85, "LF_screen_2_power"] = np.nan
  turnup.loc[turnup["LF_screen_3_power"] > 50, "LF_screen_3_power"] = np.nan
  turnup.loc[turnup["LF_screen_3_power"] < 40, "LF_screen_3_power"] = np.nan
  turnup.loc[turnup["Multifractor_1_Long_fibre_fraction"] > 50, "Multifractor_1_Long_fibre_fraction"] = np.nan
  turnup.loc[turnup["Multifractor_2_long_fibre_fraction"] > 50, "Multifractor_2_long_fibre_fraction"] = np.nan
  turnup.loc[turnup["LF_screen_2_inlet_consistency"] > 2, "LF_screen_2_inlet_consistency"] = np.nan
  turnup.loc[turnup["Cylinder_14_differential_pressure"] > 0.22, "Cylinder_14_differential_pressure"] = np.nan
  turnup.loc[turnup["Multifractor_1_long_fibre_flow"] > 400, "Multifractor_1_long_fibre_flow"] = np.nan
  turnup.loc[turnup["Short_fibre_B06_consistency"] < 4, "Short_fibre_B06_consistency"] = np.nan
  turnup.loc[turnup["Fibre_usage__T/T_"] < .06, "Fibre_usage__T/T_"] = np.nan
  turnup.loc[turnup['Combined_cost__€/T_'] > 300, 'Combined_cost__€/T_'] = np.nan
  turnup.loc[turnup["LF_screen_1_accept_flow"] < 600, "LF_screen_1_accept_flow"] = np.nan
  turnup.loc[turnup["LF_screen_1_accept_flow"] > 1000, "LF_screen_1_accept_flow"] = np.nan
  turnup.loc[turnup["Jet/wire_ratio"] > -10, "LF_screen_1_accept_flow"] = np.nan
  turnup.loc[turnup["MBS_SCT_CD"] > 2.5, "MBS_SCT_CD"] = np.nan
  turnup.loc[turnup["Draw_AD7-PR"] < 0, "Draw_AD7-PR"] = np.nan
  turnup.loc[turnup["Draw_SS-AD6"] > 0.25, "Draw_SS-AD6"] = np.nan
  turnup.loc[turnup["Draw_PD5-SS"] < -0.25, "Draw_PD5-SS"] = np.nan
  turnup.loc[turnup["Draw_PD3-PD4"] > 0.35, "Draw_PD3-PD4"] = np.nan
  turnup.loc[turnup["White_water_temperature"] > 60, "White_water_temperature"] = np.nan
  turnup.loc[turnup["Bentonite_filter_differential_pressure"] > 0, "Bentonite_filter_differential_pressure"] = np.nan
  turnup.loc[turnup["Jet/wire_ratio"] > -10, "Jet/wire_ratio"] = np.nan
  turnup.loc[turnup["Free_gas_before_dilution_water_deculator_measurement_1~^0"] > 2, "Free_gas_before_dilution_water_deculator_measurement_1~^0"] = np.nan
  turnup.loc[turnup["Free_gas_before_dilution_water_deculator_measurement_2"] > 2, "Free_gas_before_dilution_water_deculator_measurement_2"] = np.nan
  turnup.loc[turnup["Dissolved_gas_before_dilution_water_deculator"] > 2,  "Dissolved_gas_before_dilution_water_deculator"] = np.nan
  turnup.loc[turnup["Dissolved_gas_before_dilution_water_deculator"] > 2,  "Dissolved_gas_before_dilution_water_deculator"] = np.nan
  turnup.loc[turnup["Free_gas_after_stock_deculator~^0"] > 0.8,  "Free_gas_after_stock_deculator~^0"] = np.nan
  turnup.loc[turnup["Dissolved_gas_after_stock_deculator_measurement_1"] > 1,  "Dissolved_gas_after_stock_deculator_measurement_1"] = np.nan
  turnup.loc[turnup["Dissolved_gas_after_stock_deculator_measurement_2"] > 1,  "Dissolved_gas_after_stock_deculator_measurement_2"] = np.nan
  turnup.loc[turnup["Condensate_energy_from_paper_plant_to_power_plant"] < 4.5,  "Condensate_energy_from_paper_plant_to_power_plant"] = np.nan
  turnup.loc[(turnup["Act_Deaerator_mass_flow__g/T_"] > 250) & (turnup["Act_Deaerator_mass_flow__g/T_"] < 70),  "Act_Deaerator_mass_flow__g/T_"] = np.nan
  turnup.loc[(turnup["Defoamer_mass_flow__g/T_"] > 250) & (turnup["Defoamer_mass_flow__g/T_"] < 30),  "Defoamer_mass_flow__g/T_"] = np.nan
  turnup.loc[(turnup["Sizing_Agent__g/T_"] > 6000) & (turnup["Sizing_Agent__g/T_"] < 3500),  "Sizing_Agent__g/T_"] = np.nan
  turnup.loc[(turnup["Dry_Strength_Agent_mass_flow__kg/T_"] > 16) & (turnup["Dry_Strength_Agent_mass_flow__kg/T_"] < 10),  "Dry_Strength_Agent_mass_flow__kg/T_"] = np.nan
  turnup.loc[(turnup["Natriumhydroxide_mass_flow__g/T_"] > 3000) & (turnup["Natriumhydroxide_mass_flow__g/T_"] < 340),  "Natriumhydroxide_mass_flow__g/T_"] = np.nan
  turnup.loc[(turnup["Long_fibre_consistency_B07"] <= 0) & (turnup["Long_fibre_consistency_B07"] >= 100),  "Long_fibre_consistency_B07"] = np.nan
  turnup.loc[(turnup["Short_fibre_B06_consistency"] <= 0) & (turnup["Short_fibre_B06_consistency"] >= 100),  "Short_fibre_B06_consistency"] = np.nan
  turnup.loc[turnup["Long_fibre_flow"] <= 0,  "Long_fibre_flow"] = np.nan

  impute_outside_limits_with_grade_median(turnup, "Steam__kWh/T_", 800, 1220, inplace=True)
  impute_outside_limits_with_grade_median(turnup, "Steam__€/T_", 75, 110, inplace=True)

  for v in ["Pressure_to_inactivation","Current_reel_moisture_average(reel)","Current_reel_dry_average"]:
      turnup.loc[turnup[v] <= 2, v] = np.nan

  for v in ["Multifractor_1_long_fibre_flow"]:
      turnup.loc[turnup[v] <= 250, v] = np.nan

  for v in ["Starch_flow_to_inactivation"]:
      turnup.loc[turnup[v] <= 10000, v] = np.nan

  turnup.loc[(turnup["AB_Grade_ID"]==6010100) &  ( turnup["MBS_CMT30"] > 180), "MBS_CMT30"] = turnup[turnup["AB_Grade_ID"]==6010100]["MBS_CMT30"].mean()
  turnup.loc[(turnup["AB_Grade_ID"]==6010120) &  ( turnup["MBS_CMT30"] < 180), "MBS_CMT30"] = turnup[turnup["AB_Grade_ID"]==6010120]["MBS_CMT30"].mean()
  turnup.loc[turnup["MBS_SCT_CD"] > 2.5, "MBS_SCT_CD"] = np.nan

  turnup = turnup.ffill().bfill().copy()

  # TO ADD
  if not steam_null:
      turnup = turnup[turnup["Steam_flow_to_PreDryers"]>42] 
      turnup = turnup[~((turnup.Wedge_Time > "2025-10-23 11:56") & (turnup.Wedge_Time <"2025-11-16 10:00"))]
  # END TO ADD

  turnup["Fibre_usage__T/T_"]=(turnup["Current_basis_weight"]*(1-turnup["Current_reel_moisture_average(reel)"]/100)-turnup["Starch_uptake_by_paper_Bottom_Roll__g/m2_"]-turnup["Starch_uptake_by_paper_Top_Roll__g/m2_"])/turnup["Current_basis_weight"]
  turnup['Combined_cost__€/T_'] = turnup['Combined_cost__€/T_'] - turnup['Fibre_cost__€/T_'] + 146.46 * turnup["Fibre_usage__T/T_"] - turnup['Electricity__€/T_'] + turnup["Electricity__kWh/T_"] * 113.66 / 1000
  turnup['Fibre_cost__€/T_'] = 146.46*turnup["Fibre_usage__T/T_"]
  turnup['Electricity__€/T_'] = turnup["Electricity__kWh/T_"] * 113.66 / 1000

  turnup.loc[:,"Starch_uptake__g/m2_"]=turnup["Starch_uptake_by_paper_Bottom_Roll__g/m2_"]+turnup["Starch_uptake_by_paper_Top_Roll__g/m2_"]
  turnup.loc[:,"ratio_starch"]=turnup["Starch_uptake__g/m2_"]/turnup["Current_basis_weight"]
  turnup.loc[:,'MC_SF_LF_Demand'] = ((turnup['Short_fibre_flow'] * (turnup['Short_fibre_B06_consistency']/100))+(turnup['Long_fibre_flow']*(turnup['Long_fibre_consistency_B07']/100)))*(60/1000)

  turnup.loc[:,"Steam_current__€/h_"] = turnup["Steam__€/T_"] * turnup["Production_Rate__T/h_"]
  turnup.loc[:,"Electricity_current__€/h_"] = turnup["Electricity__€/T_"] * turnup["Production_Rate__T/h_"]
  turnup.loc[:,"Starch_current__€/h_"] = turnup['Starch__€/T_'] * turnup["Production_Rate__T/h_"]
  turnup['Flow_starch_main_line_to_working_tank']=turnup['Flow_starch_main_line_to_working_tank_2~^0']+turnup["Flow_starch_main_line_to_working_tank_1~^0"]
  
  turnup["concentration_starch_working_tank_1"]=turnup["Flow_starch_main_line_to_working_tank_1~^0"]/(turnup["Dilution_water_working_tank_1"]+turnup["Flow_starch_main_line_to_working_tank_1~^0"])
  turnup["concentration_starch_working_tank_2"]=turnup["Flow_starch_main_line_to_working_tank_2~^0"]/(turnup["Dilution_water_working_tank_2"]+turnup["Flow_starch_main_line_to_working_tank_2~^0"])

  turnup["fibre_short/long"] = turnup['Short_fibre_flow']*turnup['Short_fibre_B06_consistency']/(turnup['Long_fibre_flow']*turnup['Long_fibre_consistency_B07'])
  turnup["delta_moisture"]=turnup["Current_reel_moisture_average(reel)"]-turnup["Moisture_out_of_PreDryer"]
  turnup["Fibre_sqm"]=(turnup["Current_reel_dry_average"]-turnup["Starch_uptake_by_paper_Bottom_Roll__g/m2_"]-turnup["Starch_uptake_by_paper_Top_Roll__g/m2_"])

  turnup["AB_Grade_ID"] = turnup["AB_Grade_ID"].astype(str)

  turnup[_agg_cost_label()] = turnup[_costs_to_consider()].sum(axis=1)

  turnup['Combined_cost__€/T_'] = np.where(turnup['Combined_cost__€/T_'] < turnup[_agg_cost_label()] + turnup['Fibre_cost__€/T_'] ,
                                            turnup[_agg_cost_label()] + turnup['Fibre_cost__€/T_'],
                                            turnup['Combined_cost__€/T_'])        
  turnup['Chemicals__€/T_'] = turnup['Combined_cost__€/T_'] - turnup[['Steam__€/T_','Electricity__€/T_','Starch__€/T_','Fibre_cost__€/T_']].sum(axis=1)

  turnup["Combined_cost_current_€/h_"] = turnup[_agg_cost_label()] * turnup["Production_Rate__T/h_"]

  turnup["Steam_flow_to_PreDryers_sqm"]=turnup["Steam_flow_to_PreDryers"]*1e5/(turnup["Speed"]*turnup["Current_reel_width"]*60) # Kg/m2
  turnup["Steam_flow_to_PreDryers_index"]=turnup["Steam_flow_to_PreDryers_sqm"]*1000/turnup["Current_basis_weight"]   # Kg/Kg
  turnup["Steam_flow_to_AfterDryers_sqm"]=turnup["Steam_flow_to_AfterDryers"]*1e5/(turnup["Speed"]*turnup["Current_reel_width"]*60)
  turnup["Steam_flow_to_AfterDryers_index"]=turnup["Steam_flow_to_AfterDryers_sqm"]*1000/turnup["Current_basis_weight"]
  turnup["Steam__kW"]=turnup["Steam__kWh/T_"]*turnup["Production_Rate__T/h_"]

  turnup["Electricity_index"]=turnup["Electricity__kWh/T_"]
  turnup["Electricity_sqm"]=turnup["Electricity__kWh/T_"]*turnup["Production_Rate__T/h_"]*100/(turnup["Speed"]*turnup["Current_reel_width"]*60)
  turnup["SCT_CD_index"]=turnup["MBS_SCT_CD"]/turnup["Current_basis_weight"]
  turnup["retention"]=1-turnup['Consistency_white_water']/(10*turnup['Headbox_consistency'])

  coef_df = pd.DataFrame({
      "property": ["MBS_SCT_MD", "MBS_SCT_CD", "MBS_Burst", "MBS_CMT30"],
      "starch_coef": [0.1356332061139627, 0.15738688100760012, 10.8090692825425, 3.701831560389852]
  })

  turnup["Wedge_Date"] = turnup["Wedge_Time"].dt.date

  setpnts=setpoint_df[setpoint_df.variable=="min"]
  #setpnts["AB_Grade_ID"]=setpnts["AB_Grade_ID"].astype(int)
  setpnts=setpnts.rename(columns={"value":"minimum"})

  overprocessing=pd.melt(turnup[["Wedge_Time",'AB_Grade_ID','MBS_Current_reel_ID','Current_basis_weight','Starch_uptake__g/m2_','MBS_SCT_MD', 'MBS_SCT_CD', 'MBS_Burst', 'MBS_CMT30']], id_vars=['Wedge_Time','AB_Grade_ID','MBS_Current_reel_ID','Current_basis_weight','Starch_uptake__g/m2_'], value_vars=['MBS_SCT_MD', 'MBS_SCT_CD', 'MBS_Burst', 'MBS_CMT30'],var_name="property").merge(setpnts, on=["AB_Grade_ID","property"], how="left").drop("variable",axis=1).dropna()
  overprocessing=pd.merge(overprocessing,coef_df,on=["property"],how="left")
  overprocessing["property_diff"]=overprocessing["value"]-overprocessing["minimum"]
  overprocessing["property_pct"]=overprocessing["property_diff"]/overprocessing["minimum"]
  overprocessing["starch_uptake_diff"]=overprocessing["property_diff"]*overprocessing["starch_coef"]
  overprocessing["starch_mass_flow_diff_avg"]=overprocessing["starch_uptake_diff"]*1000/overprocessing['Current_basis_weight'] #kg/T
  overprocessing["starch_cost_diff"]=overprocessing["starch_mass_flow_diff_avg"]* 434.22 / 1000
  overprocessing = overprocessing[["MBS_Current_reel_ID","property","property_pct","starch_cost_diff"]]
  overprocessingT=overprocessing.groupby('MBS_Current_reel_ID').agg({"property_pct":"mean","starch_cost_diff":"mean"}).reset_index()
  overprocessingT["property"]="ALL"
  overprocessing=pd.concat([overprocessing,overprocessingT],axis=0)
  overprocessing=overprocessing.pivot(index=["MBS_Current_reel_ID"],columns="property",values=["property_pct","starch_cost_diff"])
  overprocessing1=overprocessing["property_pct"].rename(columns={"ALL":"Overprocessing_percentage","MBS_SCT_CD":"Overprocessing_SCT_CD","MBS_Burst":"Overprocessing_Burst","MBS_CMT30":"Overprocessing_CMT30"})
  overprocessing2=overprocessing["starch_cost_diff"][["ALL"]].rename(columns={"ALL":"Overprocessing_cost__€/T_"})
  overprocessing=pd.concat([overprocessing1,overprocessing2],axis=1)
  overprocessing=overprocessing.reset_index()[["MBS_Current_reel_ID","Overprocessing_percentage","Overprocessing_SCT_CD","Overprocessing_Burst","Overprocessing_CMT30","Overprocessing_cost__€/T_"]]
  turnup=pd.merge(turnup,overprocessing,on='MBS_Current_reel_ID',how="left")

  return turnup

def _raw_data(turnup_data):
  turnup = turnup_data
  turnup=turnup.drop(["Sheet_Off","Current_reel_weight","Current_reel_length","Dilution_water_storage_tank","ProRelease_1_LoVac","ProRelease_2_LoVac","ProRelease_3_LoVac","ProRelease_4_LoVac","ProRelease_5_LoVac","ProRelease_6_LoVac","ProRelease_7_LoVac","Ash_measurement_HC-line","Uhle_box_2_flow___l/min_",'Enzyme_flow_Slurry_1','Dilution_water_flow_slurry_1','Flow_slurry_1_to_reactor','Slurry_1_pumping_to_reactor','Enzyme_flow_Slurry_2','Enzyme_flow_Slurry_3','Slurry_3_level','Dilution_water_flow_slurry_3','Flow_slurry_3_to_reactor','Slurry_3_pumping_to_reactor',"DG_4-5_zero_level","Total_SQM_cost__€/hm2_","LF_screen_3_inlet_consistency", "LF_screen_3_power","LF_screen_3_reject_flow", "LF_screen_3_accept_flow"],axis=1)
  turnup=turnup.drop(["Headbox_total_flow"],axis=1)
  return turnup.sort_values("Wedge_Time").dropna(axis=1, how='all')

def shapley_contribution(df1, df2, cost_component, fibre_cost, steam_cost, electricity_cost, starch_cost, steam_features, electricity_features, starch_features, fibre_features):
  steam_feats = steam_features()
  elec_feats = electricity_features()
  starch_feats = starch_features()
  fibre_feats = fibre_features()

  results = []
  for idx in list(set(df1.index) & set(df2.index)):
      row1 = df1.loc[idx]
      row2 = df2.loc[idx]
      if cost_component=='Fibre_cost__€/T_':
          logger.info(cost_component)
          contrib = shapley_for_pair(row1, row2, fibre_cost, fibre_feats,cost_variable=cost_component)
      elif cost_component=='Steam__€/T_':
          logger.info(cost_component)
          contrib = shapley_for_pair_mc(row1, row2, steam_cost, steam_feats,cost_variable=cost_component)
      elif cost_component=='Electricity__€/T_':
          logger.info(cost_component)
          contrib = shapley_for_pair_mc(row1, row2, electricity_cost, elec_feats,cost_variable=cost_component)
      elif cost_component=='Starch__€/T_':
          logger.info(cost_component)
          contrib = shapley_for_pair_mc(row1, row2, starch_cost, starch_feats,cost_variable=cost_component)
      else:
          return None            
      delta_cost = sum(contrib.values())  # should equal cost2 - cost1
      contrib["id"] = idx
      contrib["delta_cost"] = delta_cost
      results.append(contrib)

  shapley_df = pd.DataFrame(results).set_index("id")
  return shapley_df

def _shapley_contrib(data_version, df1, df2, cost_component, grade, fibre_cost, steam_cost, electricity_cost, starch_cost, steam_features, electricity_features, starch_features, fibre_features):
  df1 = df1.groupby("AB_Grade_ID").mean()
  df2 = df2.groupby("AB_Grade_ID").mean()
    
  shapley_df = shapley_contribution(df1, df2, cost_component, fibre_cost, steam_cost, electricity_cost, starch_cost, steam_features, electricity_features, starch_features, fibre_features)
  
  df = df2.subtract(df1, axis=0).reset_index()
  df = df[df.AB_Grade_ID.isin(shapley_df.index.unique())]

  df = pd.melt(shapley_df.drop("delta_cost",axis=1).div(shapley_df["delta_cost"].abs(),axis=0).reset_index(), id_vars="id", value_name="contribution").rename(columns={"id":"AB_Grade_ID"}).merge(pd.melt(df, id_vars="AB_Grade_ID", value_name="value_change"), on=["AB_Grade_ID","variable"], how="left")
  return df

def _process_data_clustered(data_version, process_data, cost_component, grade, cost_component_features):
  turnup_process = process_data
  turnup_main = []

  for grade in [grade]:#["3200115", "6010085", "6010100", "6010120"]:
      for target in ["historic","current"]:
          tu = turnup_process[(turnup_process.AB_Grade_ID==grade) & (turnup_process.target==target)][cost_component_features + [cost_component, "AB_Grade_ID","target"]].copy()
          if len(tu) > 0:
          
              clusters = clustering(tu.drop(["AB_Grade_ID","target"], axis=1),cost_component, stability = True)
              tu["cluster"] = clusters
          else:
              tu["cluster"] = -1
          turnup_main.append(tu)
  turnup_main=pd.concat(turnup_main, axis=0)
  return turnup_main

def _process_data_clustered_summary(process_data_clustered, cost_component, grade, fibre_cost, steam_cost, electricity_cost, starch_cost, steam_features, electricity_features, starch_features, fibre_features):
    turnup_main = process_data_clustered
    turnup_main = turnup_main[turnup_main["AB_Grade_ID"]==grade]
    
    #turnup_main["ratio_starch"]=turnup_main["Starch_uptake__g/m2_"]/turnup_main["Current_basis_weight"]
    turnup_results= turnup_main.groupby(["AB_Grade_ID","target","cluster"]).agg({cost_component:"mean"}).reset_index()
    ycol = cost_component
    
    turnup_summary =  turnup_main.groupby(["AB_Grade_ID","target"]).agg({cost_component:"mean"}).reset_index().pivot(index="AB_Grade_ID", columns="target", values=cost_component)
    turnup_summary["delta"]=turnup_summary["current"]-turnup_summary["historic"]
    
    turnup_results= turnup_main.groupby(["AB_Grade_ID","target","cluster"]).agg({cost_component:"mean"}).reset_index()
    best_historic=(turnup_results[turnup_results.target=="historic"].sort_values(["AB_Grade_ID",cost_component],ascending=[True, True]).groupby("AB_Grade_ID").head(1)[["AB_Grade_ID","target","cluster"]])
    worse_historic=(turnup_results[turnup_results.target=="historic"].sort_values(["AB_Grade_ID",cost_component],ascending=[True, False]).groupby("AB_Grade_ID").head(1)[["AB_Grade_ID","target","cluster"]])
    best_current=(turnup_results[turnup_results.target=="current"].sort_values(["AB_Grade_ID",cost_component],ascending=[True, True]).groupby("AB_Grade_ID").head(1)[["AB_Grade_ID","target","cluster"]])
    worse_current=(turnup_results[turnup_results.target=="current"].sort_values(["AB_Grade_ID",cost_component],ascending=[True, False]).groupby("AB_Grade_ID").head(1)[["AB_Grade_ID","target","cluster"]])

    baseline = []
    current = []
    for g in turnup_summary.index:
        tm = turnup_main[turnup_main.AB_Grade_ID==g]
        if(turnup_summary.loc[g]["delta"]) > 0:
            baseline.append(tm.merge(best_historic[best_historic.AB_Grade_ID==g], on=["AB_Grade_ID","target","cluster"], how="inner"))
            current.append(tm.merge(worse_current[worse_current.AB_Grade_ID==g], on=["AB_Grade_ID","target","cluster"], how="inner"))
        else:
            baseline.append(turnup_main.merge(worse_historic[worse_historic.AB_Grade_ID==g], on=["AB_Grade_ID","target","cluster"], how="inner"))
            current.append(turnup_main.merge(best_current[best_current.AB_Grade_ID==g], on=["AB_Grade_ID","target","cluster"], how="inner"))
    baseline = pd.concat(baseline, axis=0)
    current = pd.concat(current, axis=0)

    fibre_feats  = fibre_features()
    steam_feats  = steam_features()
    elec_feats   = electricity_features()
    starch_feats = starch_features()

    if ycol == 'Fibre_cost__€/T_':
        df1 = baseline[fibre_feats + ["AB_Grade_ID", ycol]].copy()
        df2 = current[fibre_feats + ["AB_Grade_ID", ycol]].copy()
    
        # fibre_cost expects mapping-like with names -> use the row Series directly
        df1["unknown"] = df1[ycol] - df1[fibre_feats].apply(fibre_cost, axis=1)
        df2["unknown"] = df2[ycol] - df2[fibre_feats].apply(fibre_cost, axis=1)
    
    elif ycol == 'Steam__€/T_':
        df1 = baseline[steam_feats + ["AB_Grade_ID", ycol]].copy()
        df2 = current[steam_feats + ["AB_Grade_ID", ycol]].copy()
    
        # steam_cost now expects dict-like keyed by feature name
        df1["unknown"] = df1[ycol] - df1[steam_feats].apply(steam_cost, axis=1)
        df2["unknown"] = df2[ycol] - df2[steam_feats].apply(steam_cost, axis=1)
    
    elif ycol == 'Electricity__€/T_':
        df1 = baseline[elec_feats + ["AB_Grade_ID", ycol]].copy()
        df2 = current[elec_feats + ["AB_Grade_ID", ycol]].copy()
    
        df1["unknown"] = df1[ycol] - df1[elec_feats].apply(electricity_cost, axis=1)
        df2["unknown"] = df2[ycol] - df2[elec_feats].apply(electricity_cost, axis=1)
    
    elif ycol == 'Starch__€/T_':
        df1 = baseline[starch_feats + ["AB_Grade_ID", ycol]].copy()
        df2 = current[starch_feats + ["AB_Grade_ID", ycol]].copy()
    
        df1["unknown"] = df1[ycol] - df1[starch_feats].apply(starch_cost, axis=1)
        df2["unknown"] = df2[ycol] - df2[starch_feats].apply(starch_cost, axis=1)
    
    else:
        df1 = None
        df2 = None
    
    return df1, df2



def drilldown_df(dfp, level, object_drilldown, reference_drilldown):
  import numpy as np

  if reference_drilldown=="week":
      x_variable_summary="week"
  else:
      x_variable_summary="target"

  if object_drilldown=="cost":
      y_variable_summary = "Combined_cost__€/T_"
  elif object_drilldown=="overprocessing":
      y_variable_summary = "Overprocessing_percentage"    

  if level==1:
      color_variable_summary ="none"                
      df, dfg, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary,_agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
      if object_drilldown=="cost":            
          dfg["cost"]="TOTAL"
      elif object_drilldown=="overprocessing":
          dfg["cost"]="AVERAGE"

      if reference_drilldown=="week":
          t = dfg.Wedge_Week.max() - 1                                
          C=dfg[(dfg.Wedge_Week==(t+1))].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.Wedge_Week==(t))].rename(columns={y_variable_summary:"previous_cost"})[["cost","previous_cost" ]], on= ["cost"], how="left" )
          
      else:
          C=dfg[(dfg.target=="current")].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.target=="historic")].rename(columns={y_variable_summary:"previous_cost"})[["cost","previous_cost" ]], on= ["cost"], how="left" )
      C["AB_Grade_ID"]="ALL" 
      
  elif level==0:
      color_variable_summary ="cost"                             
      df, dfg, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary,_agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
      if reference_drilldown=="week":
          t = dfg.Wedge_Week.max() - 1
          C=dfg[(dfg.Wedge_Week==(t+1))].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.Wedge_Week==(t))].rename(columns={y_variable_summary:"previous_cost"})[["cost","previous_cost" ]], on= ["cost"], how="left" )
      else:
          C=dfg[(dfg.target=="current")].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.target=="historic")].rename(columns={y_variable_summary:"previous_cost"})[["cost","previous_cost" ]], on= ["cost"], how="left" )

      C["AB_Grade_ID"]="ALL" 
  elif level==2:
      color_variable_summary ="grade"                             
      df, dfg, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary,_agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
      if object_drilldown=="cost":            
          dfg["cost"]="TOTAL"
      elif object_drilldown=="overprocessing":
          dfg["cost"]="AVERAGE"
      if reference_drilldown=="week":
          t = dfg.Wedge_Week.max() - 1                
          C=dfg[(dfg.Wedge_Week==(t+1))].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.Wedge_Week==(t))].rename(columns={y_variable_summary:"previous_cost"})[["AB_Grade_ID","cost","previous_cost" ]], on= ["AB_Grade_ID","cost"], how="left" )
      else:
          C=dfg[(dfg.target=="current")].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.target=="historic")].rename(columns={y_variable_summary:"previous_cost"})[["AB_Grade_ID","cost","previous_cost" ]], on= ["AB_Grade_ID","cost"], how="left" )
  elif level==3:
      color_variable_summary ="grade"                       
      df, dfg1, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary,_agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
      
      if object_drilldown=="cost":
          color_variable_summary ="cost_grade"
          dfg1["cost"]="TOTAL"
      elif object_drilldown=="overprocessing":
          color_variable_summary ="overprocessing_grade"        
          dfg1["cost"]="AVERAGE"

      df, dfg2, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary,_agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
      dfg = pd.concat([dfg1,dfg2],axis=0).dropna()
      
      if reference_drilldown=="week":
          t = dfg.Wedge_Week.max() - 1
          C=dfg[(dfg.Wedge_Week==(t+1))].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.Wedge_Week==(t))].rename(columns={y_variable_summary:"previous_cost"})[["AB_Grade_ID","cost","previous_cost" ]], on= ["AB_Grade_ID","cost"], how="left" )
      else:
          C=dfg[(dfg.target=="current")].rename(columns={y_variable_summary:"current_cost"}).merge(dfg[(dfg.target=="historic")].rename(columns={y_variable_summary:"previous_cost"})[["AB_Grade_ID","cost","previous_cost" ]], on= ["AB_Grade_ID","cost"], how="left" )
  C["pct_cost"]=(C["current_cost"]-C["previous_cost"])
  C["pct_cost"]=C["pct_cost"].replace(np.inf,1)
  return C

def _process_data(turnup_data, target_range, baseline_range, steam_null):
  df = _raw_data(turnup_data).copy()
  df["target"]="none"
  df.loc[(df.Wedge_Time.dt.date >= target_range[0]) & (df.Wedge_Time.dt.date <= target_range[1]),"target"]="current"
  df.loc[(df.Wedge_Time.dt.date >= baseline_range[0]) & (df.Wedge_Time.dt.date <= baseline_range[1]),"target"]="historic"
  
  if steam_null:
      df_temp=df.groupby("AB_Grade_ID").agg({"Steam__€/T_":"mean"}).reset_index()
      df["Aggregated_cost__€/T_"]=df["Aggregated_cost__€/T_"]-df["Steam__€/T_"]
      df["Combined_cost__€/T_"]=df["Combined_cost__€/T_"]-df["Steam__€/T_"]
      df.drop("Steam__€/T_",axis=1,inplace=True)
      df=df.merge(df_temp,on="AB_Grade_ID", how="left")
      df["Aggregated_cost__€/T_"]=df["Aggregated_cost__€/T_"]+df["Steam__€/T_"]
      df["Combined_cost__€/T_"]=df["Combined_cost__€/T_"]+df["Steam__€/T_"]
  # END TO REMOVE
  df=df[df.target!="none"]        
  return df

def _clusters(df_scores, num_samples):
  gmm, clusters, nclusters, sizes = gmm_with_min_size(
      df_scores, 
      min_size=num_samples,       # <-- set your minimum samples per cluster here
      k_min=2, 
      k_max=None,        # or an explicit upper bound
      random_state=1
  )
  
  return clusters

def _grade_data_general(process_data, igrade_general):    
  return process_data[process_data["AB_Grade_ID"].isin(igrade_general)]

def _best_cluster(grade_data_process_clustered, acceptable_clusters):
  df = grade_data_process_clustered.copy()        
  best_df=df[(df.cluster.isin(acceptable_clusters))]
  if len(best_df)>0:
      bc=best_df.groupby("cluster").agg({_agg_cost_label():"median"}).sort_values(_agg_cost_label(),ascending=True).reset_index().iloc[0].cluster
      best_df  = best_df[best_df.cluster==bc].copy()
      best_df["target"]="best"
      best_df=best_df.reset_index(drop=True)            
  else:
      bc=-1
      best_df = None
  return bc,best_df

def _data_benchmark(i_variable_benchmark, grade_data_process_clustered, best_cluster, best_cluster_dataframe):
  from scipy.stats import ks_2samp
  import numpy as np

  if i_variable_benchmark==_agg_cost_label():
      df=grade_data_process_clustered[["cluster","target",i_variable_benchmark]].copy()
  else:
      df=grade_data_process_clustered[["cluster","target",i_variable_benchmark,_agg_cost_label()]].copy()
  
  historic_df=df[df.target=="historic"][[i_variable_benchmark]].copy().reset_index(drop=True)
  historic_df["target"]="historic"
  current_df=df[df.target=="current"][[i_variable_benchmark]].copy().reset_index(drop=True)
  current_df["target"]="current"
  
  potential_savings=0
  true_savings=0

  if best_cluster!=-1:
      best_df=best_cluster_dataframe[[i_variable_benchmark,"target"]]
      res=pd.concat([historic_df,best_df,current_df],axis=0)
      
      best_values  = best_df[i_variable_benchmark]
      current_values  = current_df[i_variable_benchmark]
      statistic, p_value_bc = ks_2samp(best_values, current_values)
      if p_value_bc<0.05:
          potential_savings=np.mean(best_values)-np.mean(current_values)                    
  else:
      res=pd.concat([historic_df,current_df],axis=0)            
      p_value_bc = np.inf
  
  historic_values  = historic_df[i_variable_benchmark]
  current_values  = current_df[i_variable_benchmark]
  statistic, p_value_hc = ks_2samp(historic_values, current_values)
  if p_value_hc<0.05:
      true_savings = np.mean(historic_values)-np.mean(current_values)
  
  return res,true_savings,potential_savings, p_value_hc,p_value_bc

def mix_effect(object_drilldown,reference_drilldown,grades_drilldown, process_data):
    if object_drilldown=="overprocessing":           
        y_variable_summary = "Overprocessing_percentage"
    elif object_drilldown=="cost":
        y_variable_summary = "Combined_cost__€/T_"    
    else:
        return None        

    color_variable_summary ="grade"        
    if reference_drilldown=="week":
        x_variable_summary="week"
    else:
        x_variable_summary="target"

    dfp = process_data
    dfp = dfp[dfp.AB_Grade_ID.isin(grades_drilldown)]

    df, dfg, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary,x_variable_summary,color_variable_summary, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())       
    
    A=dfg.rename(columns={"n": "n_samples"}).reset_index()

    if reference_drilldown=="week":
        t = dfg.Wedge_Week.max() - 1
        mix_comp= decompose_avg_cost_change(A[A.Wedge_Week==(t)], A[A.Wedge_Week==(t+1)],y_variable_summary)
    else:
        mix_comp= decompose_avg_cost_change(A[A.target=="historic"], A[A.target=="current"],y_variable_summary)

    return mix_comp["mix_effect"], mix_comp["cost_effect"]

def plot_KDE(df, variable, color, name):
  from scipy.stats import gaussian_kde
  import numpy as np
  import plotly.graph_objects as go

  data = df[variable].dropna().values
  kde = gaussian_kde(data, bw_method=0.2)
  xmin, xmax = data.min(), data.max()
  x = np.linspace(xmin, xmax, 500)
  y = kde(x)

  p025 = np.percentile(data, 2.5)
  p975 = np.percentile(data, 97.5)
  p500 = np.percentile(data, 50)

  traces = []
  shapes = []
  
  traces.append(
      go.Scatter(
          x=x,
          y=y,
          mode="lines",
          line=dict(color=color, width=3),
          fill="tozeroy",
          fillcolor=to_rgba(color, 0.25),
          name=name
      )
  )
  # shapes.append(
  #     dict(
  #         type="line",
  #         x0=p025,
  #         x1=p025,
  #         y0=0,
  #         y1=max(y),
  #         line=dict(
  #             color=color,
  #             width=2,
  #             dash="dash"
  #         ),
  #         name="2.5th percentile"
  #     )
  # )

  shapes.append(
      dict(
          type="line",
          x0=p500,
          x1=p500,
          y0=0,
          y1=max(y),
          line=dict(
              color=color,
              width=3,
              dash="dash"
          ),
          name="median"
      )
  )
  
  # Vertical dashed line at 97.5 percentile
  # shapes.append(
  #     dict(
  #         type="line",
  #         x0=p975,
  #         x1=p975,
  #         y0=0,
  #         y1=max(y),
  #         line=dict(
  #             color=color,
  #             width=2,
  #             dash="dash"
  #         ),
  #         name="97.5th percentile"
  #     )
  # )
  return traces, shapes, x, y

def _grade_data_process_clustered(process_data, igrade, clusters):
  df = process_data
  df = df[df["AB_Grade_ID"] == igrade]
  df["cluster"]=clusters
  return df

def _scores_clustered(df_scores, df_features, clusters):
  df=df_scores.copy()
  df["cluster"]= clusters
  return pd.concat([df.reset_index(drop=True),df_features.reset_index(drop=True)],axis=1)

def _model_performance_table(model_metric, model_performance):
  res = model_performance
  res["grade"]="TOTAL"
  tt = res.sort_values(["property"],ascending=[False])[["grade","property",model_metric,"target"]].pivot(index=["grade","target"],columns="property", values=model_metric).reset_index()
  for q in _quality_features():
      if "overprocessing" not in q.lower():
          tt[q]=tt[q].astype("double")
          tt[q]=round(tt[q],1)
  return tt

def _strength_variability_table(model_grade, lab_performance):
  metric_var = lab_performance  
  metric_var = metric_var.sort_values(["property","grade"],ascending=[False,True])
  tt = metric_var[metric_var.grade==model_grade][["grade","property","sd","target"]].pivot(index=["grade","target"],columns="property",values="sd").reset_index()
  for q in _quality_features():
      if "overprocessing" not in q.lower():
          tt[q]=tt[q].astype("double")
          tt[q]=round(tt[q],1)
  return tt

def _lab_performance_plot(turnup_data, strength_property, start_baseline, end_baseline, reference_time, model_grade):
  import plotly.graph_objects as go
  import pandas as pd

  turnup = turnup_data.copy()        
  strength_property = strength_property
  turnup = turnup[(turnup.Wedge_Time.dt.date >= start_baseline) & (turnup.Wedge_Time.dt.date <= end_baseline)  & (turnup.AB_Grade_ID == model_grade)]
  
  fig = go.Figure()
  tt,ss,x,y = plot_KDE(turnup, strength_property, "red", "baseline")
  
  fig.add_traces(tt)
  for s in ss:
      fig.add_shape(s)

  t0 = pd.to_datetime(reference_time)
  start = t0 - pd.Timedelta("30D")
  turnup = turnup_data.copy()                
  turnup = turnup[(turnup.Wedge_Time >= start) & (turnup.Wedge_Time <= t0)  & (turnup.AB_Grade_ID == model_grade)]
  tt,ss,x,y = plot_KDE(turnup, strength_property,"green", "current")
  fig.add_traces(tt)
  for s in ss:
      fig.add_shape(s)

  min_df=setpoint_df.query(f"AB_Grade_ID == '{model_grade}' & property == '{strength_property}' & variable == 'min'")
  if len(min_df) > 0:
      fig.add_shape(
          type="line",
          x0=min_df.value.values[0],
          x1=min_df.value.values[0],
          y0=0,
          y1=max(y),
          line=dict(
              color="red",
              width=4,
              dash="dash"
          ),
          name="Minimum"
      )

  fig.update_layout(template="plotly_white")

  return fig

def _rolling_lab_performance_plot(rolling_stats, strength_property, start_baseline, end_baseline, model_grade, setpoint_df):   
  import plotly.graph_objects as go

  df_plot, strength_property = rolling_stats, strength_property
  
  df_plot = df_plot.loc[model_grade].reset_index()
  
  fig = go.Figure()

  if strength_property=="MBS_CMT30" and (model_grade!="6010120" or model_grade != "6010100"):
      fig.update_layout(                
          template="plotly_white",                                
      )
          
      return fig

  # Upper bound (95th percentile)
  fig.add_trace(
      go.Scatter(
          x=df_plot["Wedge_Time"],
          y=df_plot["p_high"],
          mode="lines",
          line=dict(width=0),
          name="95th percentile",
          showlegend=False,   # we’ll label the band once
      )
  )
  
  # Lower bound (5th percentile), filled up to previous trace
  fig.add_trace(
      go.Scatter(
          x=df_plot["Wedge_Time"],
          y=df_plot["p_low"],
          mode="lines",
          line=dict(width=0),
          fill="tonexty",             # fills to previous trace (95th)
          fillcolor="rgba(0, 100, 200, 0.2)",
          name="5–95% band",
          hoverinfo="skip",           # optional: cleaner hover
          opacity=0.02,
          showlegend=False
      )
  )
  
  # Median line
  fig.add_trace(
      go.Scatter(
          x=df_plot["Wedge_Time"],
          y=df_plot["median"],
          mode="lines",
          name="Median",
          showlegend=False
      )
  )

  shapes = []
  min_df=setpoint_df.query(f"AB_Grade_ID == '{model_grade}' & property == '{strength_property}' & variable == 'min'")
  
  if len(min_df) > 0:                        
      shapes.append(
          dict(
              type="line",
              xref="x",
              yref="y",
              x0=df_plot["Wedge_Time"].min(),
              x1=df_plot["Wedge_Time"].max(),
              y0=min_df.value.values[0],
              y1=min_df.value.values[0],
              line=dict(
                  color="red",
                  width=3,
                  dash="dash",
              ),
              layer="above",
          )
      )
  shapes.append(
      dict(
          type="rect",
          xref="x",
          yref="paper",
          x0=start_baseline,
          x1=end_baseline,
          y0=0,
          y1=1,
          fillcolor="rgba(255, 0, 0, 0.15)",  # low alpha
          line=dict(width=0),
          layer="below",
      )
  )        
  
  fig.update_layout(
      title=f"{strength_property} – Median with 5–95% Band (Grade {model_grade})",
      xaxis_title="Time",
      yaxis_title=strength_property,
      template="plotly_white",
      hovermode="x",
      xaxis=dict(
          showspikes=True,
          spikemode="across",
          spikecolor="red",
          spikethickness=2,
          spikedash="solid",   # you can remove this if not needed
      ),
      shapes=shapes
      
  )

  return fig

def _model_performance_plot(model_metrics, model_targets, prediction_models, strength_property, target_model):      
  from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
  import plotly.graph_objects as go
  import numpy as np

  res,turnup_targets = model_metrics, model_targets
  LR_models = prediction_models

  ycol = strength_property
  target = target_model

  y_pred = LR_models[ycol].predict(
      turnup_targets[ycol][target].drop([ycol, 'MBS_Current_reel_ID'], axis=1)
  )
  y_test = turnup_targets[ycol][target][ycol]

  rmse = root_mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  # print(f"{ycol} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

  # Axis limits
  lims = [
      np.nanmin([y_test.min(), y_pred.min()]),
      np.nanmax([y_test.max(), y_pred.max()])
  ]

  target_name = ycol

  fig = go.Figure()

  # Scatter: observed vs predicted
  fig.add_trace(
      go.Scatter(
          x=y_test,
          y=y_pred,
          mode="markers",
          opacity=0.5,
          name="Observed vs Predicted"
      )
  )

  # 45-degree reference line: y = x
  fig.add_trace(
      go.Scatter(
          x=lims,
          y=lims,
          mode="lines",
          line=dict(dash="dash", width=1),
          name="y = x"
      )
  )
  base_width=800
  aspect_ratio=8/16
  fig.update_layout(
      template="simple_white",
      plot_bgcolor="white",
      paper_bgcolor="white",
      autosize=True,
      #width=None,
      #height=base_width*aspect_ratio,
      title=dict(
          text=(
              "Observed vs. Predicted<br>"
              f"<sup>{ycol} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}</sup>"
          ),
          x=0.5
      ),
      xaxis_title=f"Observed {target_name}",
      yaxis_title=f"Predicted {target_name}",
      legend=dict(x=0.01, y=0.99)
  )

  # Match axes to same range & aspect
  #fig.update_xaxes(range=lims)
  #fig.update_yaxes(range=lims, scaleanchor="x", scaleratio=1)
  return fig

def _working_plot(shapley_contrib, df1, df2, cost_component_drilldown):
  import plotly.graph_objects as go
  import plotly.express as px
  import pandas as pd
  from collections import defaultdict
  import numpy as np

  df = shapley_contrib.copy()
  contrib_vars = list(df.groupby("variable").agg(contribution_abs=("contribution", lambda s: s.abs().max())).sort_values("contribution_abs", ascending=False).head(10).index.values)

  df1 = df1[df1.AB_Grade_ID.isin(list(set(df1.AB_Grade_ID) & set(df2.AB_Grade_ID)))]
  df2 = df2[df2.AB_Grade_ID.isin(list(set(df1.AB_Grade_ID) & set(df2.AB_Grade_ID)))]

  df1 = df1[contrib_vars + ["AB_Grade_ID", cost_component_drilldown]]
  df2 = df2[contrib_vars + ["AB_Grade_ID", cost_component_drilldown]]

  dfb = pd.melt(
      df1,
      id_vars=["AB_Grade_ID"],
  )
  dfb["target"] = "baseline"
  
  dfc = pd.melt(
      df2,
      id_vars=["AB_Grade_ID"],
  )
  dfc["target"] = "current"
  
  df = pd.concat([dfb, dfc], axis=0)
  
  # --- Aggregate to mean + std + count + median (median unused now) ---
  agg = (
      df.groupby(["AB_Grade_ID", "variable", "target"])["value"]
      .agg(mean="mean", std="std", count="count", median="median")
      .reset_index()
  )
  
  # standard error
  agg["stderr"] = agg["std"] / np.sqrt(agg["count"])

  vars_all = list(agg["variable"].unique())
  vars_cost = sorted([v for v in vars_all if "€" in v.lower()])
  vars_other = sorted([v for v in vars_all if "€" not in v.lower()])
  if vars_cost:
      variable_order = vars_cost + vars_other
  else:
      variable_order = vars_all
  
  n_grades = agg["AB_Grade_ID"].nunique()
  row_height = 120  # adjust if needed
  
  # --- Bar plot with error bars ---
  fig = px.bar(
      agg,
      x="target",
      y="mean",
      facet_row="AB_Grade_ID",
      facet_col="variable",
      facet_col_wrap=1,
      color="target",
      error_y="stderr",
      facet_row_spacing=0.02,
      height=row_height * n_grades,
      custom_data=["median"],   # median is passed but not used anymore
      category_orders={"variable": variable_order},
  )
  
  # Clean facet labels: "AB_Grade_ID=123" -> "123"
  fig.for_each_annotation(
      lambda a: a.update(
          text=a.text.split("=")[-1].split("__")[0],
          font=dict(size=11),
      )
  )
  
  # Independent y-axes + visible labels everywhere
  fig.update_yaxes(matches=None, showticklabels=True)
  
  # --- Per-axis range based on mean ± stderr (without forcing zero) ---
  axis_extents = defaultdict(lambda: {"low": np.inf, "high": -np.inf})
  
  for trace in fig.data:
      axis_name = trace.yaxis if hasattr(trace, "yaxis") else "y"
      y = np.array(trace.y, dtype=float)
  
      # error bars
      err_obj = getattr(trace, "error_y", None)
      if err_obj is not None:
          if getattr(err_obj, "array", None) is not None:
              e = np.array(err_obj.array, dtype=float)
          elif getattr(err_obj, "value", None) is not None:
              e = np.full_like(y, float(err_obj.value))
          else:
              e = np.zeros_like(y)
      else:
          e = np.zeros_like(y)
  
      low = np.min(y - e)
      high = np.max(y + e)
  
      axis_extents[axis_name]["low"] = min(axis_extents[axis_name]["low"], low)
      axis_extents[axis_name]["high"] = max(axis_extents[axis_name]["high"], high)
  
  # Apply per-axis ranges
  for axis_name, ext in axis_extents.items():
      lo, hi = ext["low"], ext["high"]
  
      # padding
      if lo == hi:
          pad = abs(lo) * 0.1 if lo != 0 else 1.0
          lo -= pad
          hi += pad
      else:
          pad = (hi - lo) * 0.1
          lo -= pad
          hi += pad
  
      layout_axis = "yaxis" if axis_name == "y" else "yaxis" + axis_name[1:]
      fig.layout[layout_axis].update(range=[lo, hi])
  
  # --- Add mean labels (top of bars, shifted right) ---
  for trace in fig.data:
      if trace.type != "bar":
          continue
  
      for x, y, cd in zip(trace.x, trace.y, trace.customdata):
          mean_val = y
  
          fig.add_annotation(
              x=x,
              y=y,
              xref=trace.xaxis,
              yref=trace.yaxis,
              text=f"{mean_val:.2f}",   # ⬅ show mean
              showarrow=False,
              xanchor="left",
              yanchor="bottom",
              xshift=6,                # slight right shift
              yshift=2,                # small lift above bar
              font=dict(size=10),
          )
  
  # Layout
  base_width=800
  aspect_ratio=16/16
  fig.update_layout(
      template="simple_white",
      plot_bgcolor="white",
      paper_bgcolor="white",
      autosize=True,
      width=None,
      height=base_width*aspect_ratio,
      yaxis_title="mean(value)",
      margin=dict(l=80, r=20, t=40, b=40),
      showlegend=False,
  )        
  return fig

def _cost_driver_plot(shapley_contrib):
  import plotly.graph_objects as go
  import plotly.express as px
  import pandas as pd
  import numpy as np

  df = shapley_contrib.copy()

  contrib_vars = df.groupby("variable").agg(contribution_abs=("contribution", lambda s: s.abs().max())).sort_values("contribution_abs", ascending=False).head(10).index.values
  df = df[df.variable.isin(contrib_vars)]

  light_scale = [
      (0.00, "#ccffcc"),  # light green (most negative)
      (0.50, "#ffffff"),  # white (zero)
      (1.00, "#ffcccc"),  # light red (most positive)
  ]

  units = ""
  df["value_text"]  = df["contribution"].map(lambda v: f"{v:,.2f} {units}")
  arrow = np.where(df["value_change"].to_numpy() > 0, "▲",
              np.where(df["value_change"].to_numpy() < 0, "▼", "•"))

  df["arrow_text"] = [f"{a}{p:.2f}{units}" for a, p in zip(arrow, df["value_change"])]

  # color range based on pct_cost
  if len(df):
      cabs = float(np.nanmax(np.abs(df["contribution"])))  # max distance from 0
      if cabs == 0:
          cabs = 1.0  # avoid degenerate range if all zeros
  else:
      cabs = 1.0

  cmin = -cabs
  cmax = cabs

  # --- base bars (color by current_cost) ---
  fig = px.bar(
      df,
      x="variable",
      y="contribution",
      facet_col="AB_Grade_ID",
      facet_col_wrap=1,
      text="value_text",
      color="contribution",                 # << color by COST
      color_continuous_scale=light_scale,   # << light palette
      range_color=[cmin, cmax],   
      custom_data=["arrow_text","value_change","contribution"], 
  )

  for tr in list(fig.data):
      if tr.type != "bar":
          continue
      cd = getattr(tr, "customdata", None)
      if cd is None:
          continue
      cd_list = cd.tolist() if hasattr(cd, "tolist") else list(cd)
      arrow_text = [row[0] for row in cd_list]
      pct_vals   = [row[1] for row in cd_list]
      contrib_vals   = [row[2] for row in cd_list]
  
      yvals = np.asarray(tr.y, dtype=float)
      xvals = np.asarray(tr.x)
  
      # single symmetric pad based on magnitude
      if yvals.size:
          pad = 0.03 * np.nanmax(np.abs(yvals))
      else:
          pad = 0.03
  
      # colors for all points
      arrow_colors = np.array(
          ["#2ca02c" if p < 0 else "#d62728" if p > 0 else "#666666" for p in pct_vals]
      )
  
      # --- positive (and zero) bars: text ABOVE bar ---
      pos_mask = yvals >= 0
      if np.any(pos_mask):
          fig.add_trace(go.Scatter(
              x=xvals[pos_mask],
              y=yvals[pos_mask] + pad,          # slightly above bar
              mode="text",
              text=[arrow_text[i] for i in np.where(pos_mask)[0]],
              textposition="top center",
              textfont=dict(
                  size=11,
                  color=arrow_colors[pos_mask],
              ),
              hoverinfo="skip",
              showlegend=False,
              xaxis=getattr(tr, "xaxis", "x"),
              yaxis=getattr(tr, "yaxis", "y"),
              cliponaxis=False,
          ))
  
      # --- negative bars: text BELOW bar ---
      neg_mask = yvals < 0
      if np.any(neg_mask):
          fig.add_trace(go.Scatter(
              x=xvals[neg_mask],
              y=yvals[neg_mask] - pad,          # slightly below bar end
              mode="text",
              text=[arrow_text[i] for i in np.where(neg_mask)[0]],
              textposition="bottom center",
              textfont=dict(
                  size=11,
                  color=arrow_colors[neg_mask],
              ),
              hoverinfo="skip",
              showlegend=False,
              xaxis=getattr(tr, "xaxis", "x"),
              yaxis=getattr(tr, "yaxis", "y"),
              cliponaxis=False,
          ))

  # inside text + thin borders; black text stays readable with light scale
  fig.update_traces(
      selector=dict(type="bar"),
      textposition="inside",
      insidetextanchor="middle",
      textfont=dict(size=11, color="black"),
      cliponaxis=False,
  
      marker_line_color="rgba(60,60,60,0.5)",
      marker_line_width=0.8,
  )
  fig.update_yaxes(matches=None)


  base_width=800
  aspect_ratio=16/16
  fig.update_layout(
      autosize=True,
      width=None,
      height=base_width*aspect_ratio,
      template="simple_white",
      plot_bgcolor="white",
      paper_bgcolor="white"
      )

  return fig

def _overall_analysis_table(process_data, grades_drilldown, reference_drilldown):
  dfp = process_data
  dfp = dfp[dfp.AB_Grade_ID.isin(grades_drilldown)]

  tt1 = drilldown_df(dfp, 1, "cost", reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())  

  color_variable_summary ="grade"   
  if reference_drilldown=="week":
      x_variable_summary="week"
  else:
      x_variable_summary="target"

  y_variable_summary = "Combined_cost__€/T_" 
  df, dfg, c, x_var, color_var = get_process_grouped(dfp, y_variable_summary, x_variable_summary, color_variable_summary, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())       
  A=dfg.rename(columns={"n": "n_samples"}).reset_index()

  if reference_drilldown=="week":
      t = dfg.Wedge_Week.max() - 1
      mix_comp= decompose_avg_cost_change(A[A.Wedge_Week==(t)], A[A.Wedge_Week==(t+1)],y_variable_summary)
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ","Wedge_Week":"week"})
      tt1=tt1[["week","AB_Grade_ID","component","cost","n","Δ"]]   
  else:
      mix_comp= decompose_avg_cost_change(A[A.target=="historic"], A[A.target=="current"],y_variable_summary)
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ"})
      tt1=tt1[["target","AB_Grade_ID","component","cost","n","Δ"]]   
  
  
  tt1["cost"]=round(tt1["cost"],1)
  tt1["Δ"]=round(tt1["Δ"],1)
  tt1["Δ Mix Effect"] = round(mix_comp["mix_effect"],1)
  tt1["Δ Cost Effect"] = round(mix_comp["cost_effect"],1)

  tt2 = drilldown_df(dfp, 1,"overprocessing",reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
  tt1["Overprocessing"]=round(tt2["current_cost"],1)

  return tt1

def _grade_analysis_table(process_data, grades_drilldown, reference_drilldown):
  dfp = process_data
  dfp = dfp[dfp.AB_Grade_ID.isin(grades_drilldown)]

  tt1 = drilldown_df(dfp, 2, "cost", reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())  

  if reference_drilldown=="week":
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ","Wedge_Week":"week"})
      tt1=tt1[["week","AB_Grade_ID","component","cost","n","Δ"]]   
  else:
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ"})
      tt1=tt1[["target","AB_Grade_ID","component","cost","n","Δ"]]   

  tt1["cost"]=round(tt1["cost"],1)
  tt1["Δ"]=round(tt1["Δ"],1)

  tt2 = drilldown_df(dfp, 2,"overprocessing", reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
  tt1["Overprocessing"]=round(tt2["current_cost"],1)

  return tt1

def _component_analysis_table(process_data, grades_drilldown, reference_drilldown):
  dfp = process_data
  dfp = dfp[dfp.AB_Grade_ID.isin(grades_drilldown)]

  tt1 = drilldown_df(dfp, 3,"cost",reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())  

  if reference_drilldown=="week":
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ","Wedge_Week":"week"})
      tt1=tt1[["week","AB_Grade_ID","component","cost","n","Δ"]] 
  else:
      tt1=tt1.rename(columns={"cost":"component","current_cost":"cost","pct_cost":"Δ"})
      tt1=tt1[["target","AB_Grade_ID","component","cost","n","Δ"]]
  
  tt1["cost"]=round(tt1["cost"],1)
  tt1["Δ"]=round(tt1["Δ"],1)

  tt1["component"]=tt1["component"].str.replace("__€/T_","")
  tt1["component"]=tt1["component"].str.replace("_cost","")
                        
  if reference_drilldown=="week":
      pivot = tt1.pivot_table(
          index=["week","AB_Grade_ID","n"],
          columns="component",
          values=["cost", "Δ"],            
      )
  else:
      pivot = tt1.pivot_table(
          index=["target","AB_Grade_ID","n"],
          columns="component",
          values=["cost", "Δ"],            
      )

  tt2 = drilldown_df(dfp, 2,"overprocessing", reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())        

  
  df_grid = pivot.copy()
  if isinstance(df_grid.index, pd.MultiIndex):
      df_grid = df_grid.reset_index()

  # 2. Flatten MultiIndex columns into single strings
  def flatten_col(c):
      # c will be a tuple for MultiIndex; leave non-tuples as is
      if isinstance(c, tuple):
          # join non-empty levels with "__"
          parts = [str(p) for p in c if p not in (None, "")]
          return "__".join(parts)
      return c

  df_grid.columns = [flatten_col(c) for c in df_grid.columns]
  df_grid["Overprocessing"]=round(tt2["current_cost"],1)

  return df_grid

def drilldown_plot(process_data, level, object_drilldown, reference_drilldown, grades_drilldown):
  dfp = process_data
  dfp = dfp[dfp.AB_Grade_ID.isin(grades_drilldown)]

  C = drilldown_df(dfp, level, object_drilldown, reference_drilldown, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars())
  if object_drilldown=="overprocessing":
      units="%"            
  elif object_drilldown=="cost":
      units="€/t"
  else:
      units=""
  fig = plot_cost_breakdown(C, free_y=True, units=units, title=f"Comparison of {object_drilldown} by grade and {reference_drilldown}")
  if level==1:
      mix, efficiency = mix_effect(object_drilldown, reference_drilldown, grades_drilldown, process_data)
      for tr in fig.data:
          if tr.type == "scatter" and tr.mode == "text" and "top center" in (tr.textposition or ""):                
              if mix > 0:
                  mix_text = f"▲ {round(mix,1)} {units}"
              elif mix < 0:
                  mix_text = f"▼ {round(mix,1)} {units}"
              else:
                  mix_text = f"• 0"
              if efficiency > 0:
                  mix_efficiency = f"▲ {round(efficiency,1)} {units}"
              elif efficiency < 0:
                  mix_efficiency = f"▼ {round(efficiency,1)} {units}"
              else:
                  mix_efficiency = f"• 0"
              tr.text = f"{tr.text[0]} (Mix {mix_text}, Process {mix_efficiency})",
  return fig

def _drilldown_analysis_plot(drilldown, mix_contribution, level, object_drilldown, reference_drilldown):
  C = drilldown

  if object_drilldown=="overprocessing":
      units="%"            
  elif object_drilldown=="cost":
      units="€/t"
  else:
      units=""
  
  fig = plot_cost_breakdown(object_drilldown, C, free_y=True, units=units, title=f"Comparison of {object_drilldown} by grade and {reference_drilldown}")
  if level==1:
      mix, efficiency = mix_contribution
      for tr in fig.data:
          if tr.type == "scatter" and tr.mode == "text" and "top center" in (tr.textposition or ""):                
              if mix > 0:
                  mix_text = f"▲ {round(mix,1)} {units}"
              elif mix < 0:
                  mix_text = f"▼ {round(mix,1)} {units}"
              else:
                  mix_text = f"• 0"
              if efficiency > 0:
                  mix_efficiency = f"▲ {round(efficiency,1)} {units}"
              elif efficiency < 0:
                  mix_efficiency = f"▼ {round(efficiency,1)} {units}"
              else:
                  mix_efficiency = f"• 0"
              tr.text = f"{tr.text[0]} (Mix {mix_text}, Process {mix_efficiency})",
  return fig

def _strengthplot(strength_df, plotly_theme, igrade):
  import plotly.express as px
  import plotly.graph_objects as go
  import numpy as np

  df=strength_df
  df["cluster"]=df["cluster"].astype(str)
  template = plotly_theme
  nbins_hist = 10
  nbins2d = 10

  if df.empty:
      raise ValueError("Input dataframe is empty.")
  req_cols = {"MBS_Current_reel_ID", "property", "value", "min"}
  missing = req_cols - set(df.columns)
  if missing:
      raise ValueError(f"Missing required columns: {missing}")

  # Ensure cluster is string when present
  if "cluster" in df.columns:
      df = df.copy()
      df["cluster"] = df["cluster"].astype(str)

  props = df["property"].dropna().unique().tolist()
  if len(props) == 0:
      raise ValueError("Column 'property' has no values.")
  if len(props) > 2:
      raise ValueError("This helper supports up to 2 properties.")

  # Build a consistent color map from clusters -> Alphabet
  palette = px.colors.qualitative.Alphabet
  if "cluster" in df.columns:
      clusters = sorted(df["target"].dropna().unique().tolist())
      #color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(clusters)}
      color_map =  dict(zip(["historic","best","current","none"], ["blue","green","red","gray"]))
      category_orders = {"cluster": clusters}
      color_arg = "cluster"
  else:
      color_map = None
      category_orders = None
      color_arg = None

  # ---------- CASE A: ONE PROPERTY -> mirror (value vs value) ----------
  if len(props) == 1:
      p = props[0]
      dP = df[df["property"] == p].copy()

      # Global min for this property (prefer unique, fallback to median)
      mins = dP["min"].dropna().unique()
      p_min = float(mins[0]) if len(mins) == 1 else float(dP["min"].median())

      # Build a "wide" dataframe with identical x/y = value
      # Keep cluster for coloring (if present)
      pivot = dP[["MBS_Current_reel_ID", "value"] + ([ "cluster"] if "cluster" in dP.columns else [])].copy()
      pivot.rename(columns={"value": f"value_{p}"}, inplace=True)
      # Duplicate as y column
      pivot[f"value_{p}_y"] = pivot[f"value_{p}"]

      xcol = f"value_{p}"
      ycol = f"value_{p}_y"

      jitter_frac = 0.10
      rng = np.random.default_rng(42)
      val_std = float(np.nanstd(pivot[xcol])) or 1.0
      jitter_scale = jitter_frac * val_std
      pivot[ycol] = pivot[ycol] + rng.normal(0.0, jitter_scale, size=len(pivot))
      pivot[xcol] = pivot[xcol] + rng.normal(0.0, jitter_scale, size=len(pivot))

      # Scatter + marginals (PDFs)
      fig = px.scatter(
          pivot,
          x=xcol,
          y=ycol,
          color=color_arg,
          opacity=0.8,
          marginal_x="histogram",
          marginal_y="histogram",
          template=template,
          color_discrete_map=color_map,
          category_orders=category_orders
      )

      # Make marginal histograms show PDF
      fig.update_traces(
          selector=dict(type="histogram"),
          histnorm="probability density",
          nbinsx=nbins_hist,
          nbinsy=nbins_hist,
          opacity=0.5
      )

      # 2D density contours (on the mirrored data)
      d2 = px.density_contour(
          pivot,
          x=xcol,
          y=ycol,
          color=None,
          nbinsx=nbins2d,
          nbinsy=nbins2d,
          template=template
      )
      for tr in d2.data:
          tr.update(line=dict(color="rgba(0,0,0,0.35)", width=1), showscale=False, hoverinfo="skip")
          fig.add_trace(tr)

      # y=x reference (main panel only)
      # Determine range
      x_min_data = np.nanmin(pivot[xcol])
      x_max_data = np.nanmax(pivot[xcol])
      fig.add_trace(go.Scatter(
          x=[x_min_data, x_max_data],
          y=[x_min_data, x_max_data],
          mode="lines",
          line=dict(color="rgba(0,0,0,0.25)", dash="dot"),
          name="y=x",
          showlegend=False
      ))

      # Red min lines (same threshold on both axes), main panel only
      fig.add_shape(
          type="line",
          x0=p_min, x1=p_min, y0=0, y1=1,
          xref="x", yref="y domain",
          line=dict(color="red", width=2, dash="dash"),
          layer="above"
      )
      fig.add_shape(
          type="line",
          x0=0, x1=1, y0=p_min, y1=p_min,
          xref="x domain", yref="y",
          line=dict(color="red", width=2, dash="dash"),
          layer="above"
      )

      fig.update_layout(
          xaxis_title=p,
          yaxis_title=p,
          legend_title="cluster" if color_arg else None,
          title=f"Strength {igrade}"
      )
      return fig

  # ---------- CASE B: TWO PROPERTIES ----------
  p1, p2 = props

  # Wide pivot: one row per reel (+ cluster), value/min per property
  pivot = df.pivot_table(
      index=["MBS_Current_reel_ID"] + (["cluster"] if "cluster" in df.columns else []),
      columns="property",
      values=["value", "min"],
      aggfunc="first"
  )
  pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]  # ('value','P1')->'value_P1'
  pivot = pivot.reset_index()

  xcol, ycol = f"value_{p1}", f"value_{p2}"

  # Min thresholds (prefer unique per property; else median)
  x_min_vals = df.loc[df["property"] == p1, "min"].dropna().unique()
  y_min_vals = df.loc[df["property"] == p2, "min"].dropna().unique()
  x_min = float(x_min_vals[0]) if len(x_min_vals) == 1 else float(df.loc[df["property"] == p1, "min"].median())
  y_min = float(y_min_vals[0]) if len(y_min_vals) == 1 else float(df.loc[df["property"] == p2, "min"].median())

  # Scatter + marginals with consistent colors
  fig = px.scatter(
      pivot,
      x=xcol,
      y=ycol,
      color=color_arg,
      opacity=0.8,
      marginal_x="histogram",
      marginal_y="histogram",
      template=template,
      color_discrete_map=color_map,
      category_orders=category_orders
  )

  #Marginals as PDFs
  fig.update_traces(
      selector=dict(type="histogram"),
      histnorm="probability density",
      nbinsx=nbins_hist,
      nbinsy=nbins_hist,
      opacity=0.5
  )        

  # 2D density contours
  d2 = px.density_contour(
      pivot,
      x=xcol,
      y=ycol,
      color=None,
      nbinsx=nbins2d,
      nbinsy=nbins2d,
      template=template
  )
  for tr in d2.data:
      tr.update(line=dict(color="rgba(0,0,0,0.35)", width=1), showscale=False, hoverinfo="skip")
      fig.add_trace(tr)

  # Red min lines (main panel only)
  fig.add_shape(
      type="line",
      x0=x_min, x1=x_min, y0=0, y1=1,
      xref="x", yref="y domain",
      line=dict(color="red", width=2, dash="dash"),
      layer="above"
  )
  fig.add_shape(
      type="line",
      x0=0, x1=1, y0=y_min, y1=y_min,
      xref="x domain", yref="y",
      line=dict(color="red", width=2, dash="dash"),
      layer="above"
  )

  fig.update_layout(
      xaxis_title=p1,
      yaxis_title=p2,
      legend_title="cluster" if color_arg else None,
      title=f"Strength {igrade}"
  )
  return fig

def _scoringplot(grade_data_process_clustered, scores_clustered, best_cluster, variable_x_scoring, variable_y_scoring, feature_scoring, samples_pca, centroids_pca, plotly_theme):
  import plotly.graph_objects as go
  import numpy as np
  import plotly.express as px

  bc = best_cluster

  if feature_scoring == "score":
      datax = scores_clustered.copy()
      datay = scores_clustered.copy()
  else:
      datax = grade_data_process_clustered.copy()
      datay = scores_clustered.copy()

  # ensure consistent types
  datax["cluster"] = datax["cluster"].astype(str)
  datay["cluster"] = datay["cluster"].astype(str)

  varx = variable_x_scoring if feature_scoring == "score" else feature_scoring
  vary = variable_y_scoring

  pcgx = datax[[varx, "cluster"] + (["target"] if "target" in datax.columns else [])]
  pcgy = datay[[vary, "cluster"] + (["target"] if "target" in datay.columns else [])]

  pcgx_stat = pcgx.groupby(["cluster"]).agg(avg=(varx, "mean"), std=(varx, "std")).reset_index()
  pcgy_stat = pcgy.groupby(["cluster"]).agg(avg=(vary, "mean"), std=(vary, "std")).reset_index()

  fig = go.Figure()

  if samples_pca:
      for cluster in sorted(datax["cluster"].unique()):
          selx = datax["cluster"] == cluster
          sely = datay["cluster"] == cluster

          cluster_datax = datax.loc[selx, :]
          cluster_datay = datay.loc[sely, :]

          x_vals = cluster_datax[varx].to_numpy()
          y_vals = cluster_datay[vary].to_numpy()

          # Determine the 'target' per point; prefer datax if available, else datay; else None
          if "target" in cluster_datax.columns:
              target_vals = cluster_datax["target"].astype(str).to_numpy()
          elif "target" in cluster_datay.columns:
              target_vals = cluster_datay["target"].astype(str).to_numpy()
          else:
              target_vals = None

          color = px.colors.qualitative.Alphabet[int(cluster)]
          size = 12 if int(cluster) == bc else 8

          # Build per-point outline styling
          if target_vals is not None:
              is_current = (target_vals == "current")
              line_colors = np.where(is_current, "red", color)
              line_widths = np.where(is_current, 2, 1)
          else:
              # fallback (no 'target' column)
              line_colors = color
              line_widths = 2

          trace = go.Scatter(
              x=x_vals,
              y=y_vals,
              mode="markers",
              name=f"Best {cluster}" if int(cluster) == bc else f"Cluster {cluster}",
              marker=dict(
                  symbol="circle",
                  color=color,
                  size=size,
                  line=dict(
                      color=line_colors,  # supports array-like for per-point outline color
                      width=line_widths   # supports array-like for per-point outline width
                  )
              ),
              showlegend=True
          )
          fig.add_trace(trace)

  if centroids_pca:
      for i, row in pcgx_stat.iterrows():
          cluster = int(row["cluster"])
          avg_x = row["avg"]
          avg_y = pcgy_stat.loc[i, "avg"]
          color = px.colors.qualitative.Alphabet[cluster]

          trace = go.Scatter(
              x=[avg_x],
              y=[avg_y],
              mode="markers",
              name=f"Best {cluster}" if int(cluster) == bc else f"Cluster {cluster}",
              marker=dict(
                  symbol="x",
                  color=color,
                  size=20 if int(cluster) == bc else 15,
                  line=dict(width=2)
              ),
              showlegend=False if samples_pca else True
          )
          fig.add_trace(trace)

          std_x = row["std"]
          std_y = pcgy_stat.loc[i, "std"]
          if (pd.notna(std_x) and pd.notna(std_y) and (std_x > 0) and (std_y > 0)):
              fig.add_shape(
                  type="circle",
                  xref="x",
                  yref="y",
                  x0=avg_x - 2 * std_x,
                  x1=avg_x + 2 * std_x,
                  y0=avg_y - 2 * std_y,
                  y1=avg_y + 2 * std_y,
                  fillcolor=color,
                  line=dict(color=color),
                  opacity=0.3 if int(cluster) == bc else 0.2,
                  layer="below",
              )

  fig.update_layout(
      showlegend=True,
      margin=dict(t=40, b=40, l=40, r=40),
      autosize=True,
      template=plotly_theme
  )

  return fig

def _loadingsplot(principal_component, plotly_theme):
  import plotly.graph_objects as go

  pc = principal_component
  fig = go.Figure(go.Bar(
      x=pc.weight,
      y=pc.feature,
      orientation='h'))

  fig.update_layout(yaxis={'categoryorder': 'total ascending'},margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)

  return fig

def _clusteredcostplot(grade_data_process_clustered, acceptable_clusters, best_cluster, iclustered_cost, plotly_theme):
  import plotly.express as px

  aa = acceptable_clusters
  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  gdpc=grade_data_process_clustered.copy()

  bc = best_cluster
  gdpc.loc[gdpc["cluster"] == bc, "target"] = "best"
  
  gdpc=gdpc.sort_values("cluster")

  fig = px.box(
      gdpc, 
      x="cluster", 
      y=iclustered_cost, 
      color="cluster",
      color_discrete_sequence=px.colors.qualitative.Alphabet
  )
  
  quantile = gdpc.groupby("cluster")[iclustered_cost].quantile(0.65)
  for cluster, quantile in quantile.items():
      if len(gdpc[(gdpc.target!="current") & (gdpc.cluster==cluster)])==0:
          label = "CURRENT"
      else:
          if cluster==bc:
              label = "BEST"
          elif cluster in aa:
              label = "OK"
          else:
              label = "BAD"
      fig.add_annotation(
      x=cluster,
          y=quantile,  # slightly above median
          text=label,
          showarrow=False,
          font=dict(color="red" if label == "BAD" else "green", size=12),
          xanchor="center"
      )

  # Optional: set figure layout to auto-size
  fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  #fig.update_traces(boxpoints=False) 
  return fig

def _costevolutionplot(raw_data, igrade, i_variable_benchmark, target_range, baseline_range, box_evolution, plotly_theme):
  import plotly.express as px
  import plotly.graph_objects as go

  c = dict(zip(["historic","best","current","none"], ["blue","green","red","gray"]))
  
  
  df = raw_data.copy()
  df["Wedge_Time"] = pd.to_datetime(df["Wedge_Time"])
  df["Wedge_Date"]=df["Wedge_Time"].dt.date
  df["target"]="none"
  df.loc[(df.Wedge_Time.dt.date>=target_range[0]) & (df.Wedge_Time.dt.date<=target_range[1]),"target"]="current"
  df.loc[(df.Wedge_Time.dt.date>=baseline_range[0]) & (df.Wedge_Time.dt.date<=baseline_range[1]),"target"]="historic"   
  df = df[df.Wedge_Time.dt.date <= target_range[1]]     
  
  # TO REMOVE
  df_temp=df[df.target=="historic"].groupby("AB_Grade_ID").agg({"Steam__€/T_":"mean"}).reset_index()
  df["Aggregated_cost__€/T_"]=df["Aggregated_cost__€/T_"]-df["Steam__€/T_"]
  df["Combined_cost__€/T_"]=df["Aggregated_cost__€/T_"]-df["Steam__€/T_"]
  df.drop("Steam__€/T_",axis=1,inplace=True)
  df=df.merge(df_temp,on="AB_Grade_ID", how="left")
  df["Aggregated_cost__€/T_"]=df["Aggregated_cost__€/T_"]+df["Steam__€/T_"]
  df["Combined_cost__€/T_"]=df["Aggregated_cost__€/T_"]+df["Steam__€/T_"]
  # END TO REMOVE

  
  df=df[df["AB_Grade_ID"] == igrade][["target","Wedge_Date",i_variable_benchmark]]

  if box_evolution:
      fig = px.box(
          df, 
          x="Wedge_Date", 
          y=i_variable_benchmark, 
          color="target",
          color_discrete_map=c
      )
      fig.update_traces(marker=dict(opacity=0)) 
  else:            
      fig = px.line(                 
          x=df["Wedge_Date"], 
          y=df.groupby("Wedge_Date")[i_variable_benchmark].transform("mean"), 
          color=df["target"],
          color_discrete_map=c,
          markers=True              
      )
      
  
  fig.update_layout(
      title=go.layout.Title(
              text=f"{igrade}: {i_variable_benchmark}",
              xref="paper",
              x=0
          ),
          template=plotly_theme,
          margin=dict(t=40, b=40, l=40, r=40), 
          autosize=True
      )
  
  return fig

def _benchmarkplot(data, true_savings, potential_savings, p_value_hc, p_value_bc, i_variable_benchmark, igrade, plotly_theme):    
  import plotly.express as px
  import numpy as np
  import plotly.graph_objects as go

  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  if "€" in i_variable_benchmark:
      if p_value_bc is np.inf or p_value_bc > 0.05:
          msg=f"Current savings {round(true_savings,1)} €/T (p-value:{round(p_value_hc,4)})"
      else:
          msg=f"Current savings {round(true_savings,1)} €/T (p-value:{round(p_value_hc,4)}), Potential savings {round(-potential_savings, 1)} €/T (p-value:{round(p_value_bc,4)})"
      
  else:
      if p_value_bc is np.inf:
          msg=f"Current improvement {round(-true_savings,1)} (p-value:{round(p_value_hc,4)})"
      else:
          msg=f"Current improvement {round(-true_savings,1)} (p-value:{round(p_value_hc,4)}), Potential improvement {round(-potential_savings, 1)} (p-value:{round(p_value_bc,4)})"

  
  
  fig = px.box(data, x="target", y = i_variable_benchmark, color="target",color_discrete_map=c)
  fig.update_layout(
      title=go.layout.Title(
              text=f"{igrade}: {i_variable_benchmark}",
              xref="paper",
              x=0
          ),
          xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
                  text=msg
                  )
          ),
          template=plotly_theme,
          margin=dict(t=40, b=40, l=40, r=40), 
          autosize=True
      )
  
  return fig

def _clusteredqualityplot(grade_data_process_clustered, acceptable_clusters, best_cluster, quality_limits, iclustered_quality, box_quality, plotly_theme):
  import plotly.express as px

  aa = acceptable_clusters
  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  sp=quality_limits

  if "min" in sp.variable.unique():
      min_value=sp[sp.variable=="min"].value.values[0]
  else:
      min_value=None

  if "target" in sp.variable.unique():
      target_value=sp[sp.variable=="target"].value.values[0]
  else:
      target_value=None

  gdpc=grade_data_process_clustered.copy()
  

  bc = best_cluster
  gdpc.loc[gdpc["cluster"] == bc, "target"] = "best"
  
  gdpc=gdpc.sort_values("cluster")
  
  if box_quality:
      fig = px.box(
          gdpc, 
          x="cluster", 
          y=iclustered_quality, 
          color="cluster",
          color_discrete_sequence=px.colors.qualitative.Alphabet
      )
  else:
      fig = px.strip(
          gdpc, 
          x="cluster", 
          y=iclustered_quality, 
          color="target",
          color_discrete_map=c
      )


  quantiles = gdpc.groupby("cluster")[iclustered_quality].quantile(0.65)
  for cluster, quantile in quantiles.items():
      if len(gdpc[(gdpc.target!="current") & (gdpc.cluster==cluster)])==0:
          label = "CURRENT"
      else:
          if cluster==bc:
              label = "BEST"
          elif cluster in aa:
              label = "OK"
          else:
              label = "BAD"

      fig.add_annotation(
          x=cluster,
          y=quantile,  # slightly above median
          text=label,
          showarrow=False,
          font=dict(color="red" if label == "BAD" else "green", size=12),
          xanchor="center"
      )

  if min_value is not None:
      fig.add_shape(
          type="line",
          x0=gdpc.cluster.min(),
          x1=gdpc.cluster.max(),
          y0=min_value,
          y1=min_value,
          xref="paper",  # spans the entire x-axis
          yref="y",      # in data coordinates
          line=dict(color="Red", width=2, dash="dash")  # customize color/style
      )

  if target_value is not None:
      fig.add_shape(
          type="line",
          x0=gdpc.cluster.min(),
          x1=gdpc.cluster.max(),
          y0=target_value,
          y1=target_value,
          xref="paper",  # spans the entire x-axis
          yref="y",      # in data coordinates
          line=dict(color="Green", width=2, dash="dash")  # customize color/style
      )


  # Optional: set figure layout to auto-size
  fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  #fig.update_traces(boxpoints=False) 
  return fig

def _oprangeplot(grade_data_process_clustered, best_cluster, principalfeature_oprange, principal_oprange, filter_oprange, box_oprange, igrade, plotly_theme):
  import plotly.graph_objects as go
  import plotly.express as px

  if "all" in principal_oprange:
      gdpc=grade_data_process_clustered.copy()
      bc = best_cluster
      gdpc.loc[(gdpc["cluster"] == bc) & (gdpc["target"] == "historic"), "target"] = "best"
      gdpc=gdpc[gdpc.target=="best"]
      gdpc=pd.melt(gdpc[_cost_influencers_1() + _cost_influencers_2() + _cost_influencers_3()], var_name="feature").reset_index(drop=True).groupby("feature")["value"].agg("mean").reset_index().sort_values("feature")
      gdpc["value"]=round(gdpc["value"],2)
      fig = go.Figure(
          data=[go.Table(
              header=dict(values=list(gdpc.columns)),
              cells=dict(values=[gdpc[c] for c in gdpc.columns])
          )]
      )
      return fig
  else:
      gdpc=grade_data_process_clustered.copy()
      bc = best_cluster
      gdpc.loc[(gdpc["cluster"] == bc) & (gdpc["target"] == "historic"), "target"] = "best"
      c = dict(zip(["historic","best","current"], ["blue","green","red"]))
      
      if filter_oprange:
          gdpc=gdpc[gdpc.target=="best"]

      gdpc=pd.melt(gdpc[["cluster","target"]+list(principalfeature_oprange)], id_vars=["cluster","target"],var_name="feature").reset_index()

      if len(gdpc)>0:
          if box_oprange:
              fig = px.box(gdpc, 
                      x="cluster", 
                      y="value", 
                      color="target",
                      facet_col="feature",
                      facet_col_wrap=3,
                      color_discrete_map=c
                  )
          else:
              fig = px.strip(gdpc, 
                      x="cluster", 
                      y="value", 
                      color="target",
                      facet_col="feature",
                      facet_col_wrap=3,
                      color_discrete_map=c
                  )
              

          fig.update_yaxes(matches=None, showticklabels=True)

          fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme,title=f"Cost effective Operating range {igrade}")
          return fig
      else:
          fig= go.Figure()
          fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
          return fig

def _featureclusteredplot(grade_data_process_clustered, principalfeature_pca, plotly_theme):
  import plotly.express as px

  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  fig = px.strip(grade_data_process_clustered, 
          x="cluster", 
          y=principalfeature_pca, 
          color="target",
          color_discrete_map=c,
          stripmode = "overlay"   # Select between "group" or "overlay" mode
      )

  fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  return fig

def _correlationplot(grade_data_general, itarget, plotly_theme):
  import plotly.express as px

  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  fig = px.box(
      grade_data_general(), 
      x="AB_Grade_ID", 
      y=itarget, 
      color="target",
      color_discrete_map=c
  )
  # Optional: set figure layout to auto-size
  fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  return fig

def _summary_cost_plot(process_data, x_variable_summary, y_variable_summary, y_variable_summary_secondary, color_variable_summary, trim_summary, grades_summary, type_summary, summary_bar_lines, plotly_theme):      
  import plotly.express as px
  import numpy as np
  from plotly.subplots import make_subplots
  import plotly.graph_objects as go

  if trim_summary:
      df = process_data.sort_values(["grammage","paper_type"])
      df[y_variable_summary]=np.where(df[y_variable_summary]>df[y_variable_summary].mean(),df[y_variable_summary].mean(),df[y_variable_summary])
  else:    
      df = process_data.sort_values(["grammage","paper_type"])

  df = df[df.AB_Grade_ID.isin(grades_summary)]


  df, dfg, c, x_var, color_var =  get_process_grouped(df, y_variable_summary, x_variable_summary, color_variable_summary, _agg_cost_label2(), _costs_to_consider2(), _overprocessing_vars(), y_variable_summary_secondary)

  
  if type_summary=="box":
      fig = px.box(
          df.astype({x_var: 'str'}),
          x=x_var, 
          y=y_variable_summary, 
          color=color_var,
          color_discrete_map=c,
          category_orders=  {x_var:[str(v) for v in dfg[x_var].unique()]}
      )
      # Optional: set figure layout to auto-size
      fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  elif type_summary=="line":            
      fig = px.line(
          dfg.astype({x_var: 'str'}), 
          x=x_var, 
          y=y_variable_summary, 
          color=color_var,
          color_discrete_map=c,
          markers=True,
          category_orders=  {x_var:[str(v) for v in dfg[x_var].unique()]}
      )
      # Optional: set figure layout to auto-size
      fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  elif type_summary=="violin":
      fig = px.violin(
          df.astype({x_var: 'str'}),
          x=x_var, 
          y=y_variable_summary, 
          color=color_var,
          color_discrete_map=c,
          points="all",
          category_orders=  {x_var:[str(v) for v in dfg[x_var].unique()]}
          #jitter=0.35
      )

      fig.update_traces(
          selector=dict(type="violin"),
          points="all",
          pointpos=0,           # 0=center of the violin; [-1,1] moves left/right
          jitter=0.3,           # horizontal jitter so they spread within the violin
          marker=dict(size=4, opacity=0.55),
          width=0.9             # (optional) wider violins
      )
      
      # key: overlay the color traces instead of grouping side-by-side
      fig.update_layout(violinmode="overlay")

      fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  elif type_summary=="bar":
      fig = make_subplots(specs=[[{"secondary_y": True}]])
      
      bar_fig = []
      if  color_var is None:
          group_df = dfg    
          if x_variable_summary=="week" and color_variable_summary=="grade":
              
              width_base=(dfg[x_var].max()-dfg[x_var].min())/(len(dfg[x_var].unique())*dfg["n"].max())
              bar_fig.append(go.Bar(
                  x=group_df[x_var],
                  y=group_df[y_variable_summary],            
                  width=width_base*group_df["n"]
              ))
          else:
              bar_fig.append(go.Bar(
                  x=group_df[x_var],
                  y=group_df[y_variable_summary],
              ))

          for tr in bar_fig:
              fig.add_trace(tr, secondary_y=False)
      else:
          for group in dfg[color_var].unique():
              group_df = dfg[dfg[color_var] == group]
              
      
              if x_variable_summary=="week" and color_variable_summary=="grade":
                  
                  weeks = dfg[x_var].unique()                  
                  weeks_dif = int(weeks[-1])-int(weeks[0])

                  if weeks_dif<=0:
                      weeks_dif += 52

                  width_base=(weeks_dif)/(len(dfg[x_var].unique())*len(dfg[color_var].unique())*dfg["n"].max())
                  bar_fig.append(go.Bar(
                      x=group_df[x_var],
                      y=group_df[y_variable_summary],                                
                      name=f"{group}",
                      marker=dict(color=c[group]),
                      legendgroup=group,
                      width=width_base*group_df["n"]
                  ))
              else:
                  bar_fig.append(go.Bar(
                      x=group_df[x_var],
                      y=group_df[y_variable_summary], 
                      name=f"{group}",
                      marker=dict(color=c[group]),
                      legendgroup=group,                    
                  ))                

          for tr in bar_fig:
              fig.add_trace(tr, secondary_y=False)

      if summary_bar_lines:
          if x_variable_summary=="grade":
              if color_variable_summary=="cost":
                  df_1= df.groupby([x_var, "grammage","paper_type","cost"])[y_variable_summary].agg("mean").reset_index().sort_values(["grammage","paper_type"])
                  df_1= df_1.groupby([x_var, "grammage","paper_type"])[y_variable_summary].agg("sum").reset_index().sort_values(["grammage","paper_type"])

              else:
                  df_1= df.groupby([x_var, "grammage","paper_type"])[y_variable_summary].agg("mean").reset_index().sort_values(["grammage","paper_type"])
              df_2= df.groupby([x_var, "grammage","paper_type"])[y_variable_summary_secondary].agg("mean").reset_index().sort_values(["grammage","paper_type"])

          else:              
              if x_var=="Wedge_Week":
                  x_var_  = ["Wedge_Year","Wedge_Week"]
              else:
                  x_var_  = ["Wedge_Week"]
              if color_variable_summary=="cost":
                  df_1= df.groupby(x_var_ + ["cost"])[y_variable_summary].agg("mean").reset_index().sort_values(x_var_)
                  df_1= df_1.groupby(x_var_)[y_variable_summary].agg("sum").reset_index().sort_values(x_var_)
              else:
                  df_1= df.groupby(x_var_)[y_variable_summary].agg("mean").reset_index().sort_values(x_var_)
              
              df_2= df.groupby(x_var_)[y_variable_summary_secondary].agg("mean").reset_index().sort_values(x_var_)

              

          fig.add_scatter(
              x=df_1[x_var], 
              y=df_1[y_variable_summary], 
              marker=dict(size=10, color='blue'), 
              mode="lines+markers",
              line=dict(color='blue', width=2),
              name="avg",
              secondary_y=False,
          )
          
          fig.add_scatter(
              x=df_2[x_var], 
              y=df_2[y_variable_summary_secondary], 
              marker=dict(size=10, color='blue'), 
              mode="lines+markers",
              line=dict(color='red', width=2),
              name=y_variable_summary_secondary,
              secondary_y=True,
          )

          y1 = y_variable_summary
          y2 = y_variable_summary_secondary
          primary_vals, secondary_vals = [], []
          for tr in fig.data:
              if tr.type in ("bar", "scatter") and tr.y is not None:
                  vals = [v for v in tr.y if v is not None]
                  if getattr(tr, "yaxis", "y") in ("y", "y1"):
                      primary_vals.extend(vals)
                  elif getattr(tr, "yaxis", None) == "y2":
                      secondary_vals.extend(vals)
          
          def padded_range(vals):
              vmin, vmax = float(np.min(vals)), float(np.max(vals))
              if vmin == vmax:
                  pad = 0.05 * (abs(vmax) + 1.0)
              else:
                  pad = 0.05 * (vmax - vmin)
              return [vmin - pad, vmax + pad]
          
          if primary_vals:
              fig.update_yaxes(autorange=False, range=padded_range(primary_vals), secondary_y=False)
          if secondary_vals:
              fig.update_yaxes(autorange=False, range=padded_range(secondary_vals), secondary_y=True)
      
          # (optional) axis titles
          order = dfg[x_var].astype(str).unique().tolist()
          fig.update_xaxes(
              type="category",
              categoryorder="array",
              categoryarray=order
          )
          fig.update_yaxes(title_text=y1, secondary_y=False)
          fig.update_yaxes(title_text=y2, secondary_y=True)
          # Optional: set figure layout to auto-size
      else:
          y1 = y_variable_summary
          y2 = y_variable_summary_secondary
          primary_vals, secondary_vals = [], []
          for tr in fig.data:
              if tr.type in ("bar") and tr.y is not None:
                  vals = [v for v in tr.y if v is not None]
                  if getattr(tr, "yaxis", "y") in ("y", "y1"):
                      primary_vals.extend(vals)
                  
          
          def padded_range(vals):
              vmin, vmax = float(np.min(vals)), float(np.max(vals))
              if vmin == vmax:
                  pad = 0.05 * (abs(vmax) + 1.0)
              else:
                  pad = 0.05 * (vmax - vmin)
              return [vmin - pad, vmax + pad]
          
          order = dfg[x_var].astype(str).unique().tolist()
          fig.update_xaxes(
              type="category",
              categoryorder="array",
              categoryarray=order
          )

          if primary_vals:
              fig.update_yaxes(autorange=False, range=padded_range(primary_vals), secondary_y=False)
          if secondary_vals:
              fig.update_yaxes(autorange=False, range=padded_range(secondary_vals), secondary_y=True)

      fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme,barmode="group")
          
  return fig

def _targetplot(grade_data_general, itarget, plotly_theme):
  import plotly.express as px

  c = dict(zip(["historic","best","current"], ["blue","green","red"]))

  fig = px.box(
      grade_data_general, 
      x="AB_Grade_ID", 
      y=itarget, 
      color="target",
      color_discrete_map=c
  )
  # Optional: set figure layout to auto-size
  fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
  return fig

def _generalplot(grade_data_general, diff, ivariable_x_general, ivariable_y_general, ivariable_y2_general, ivariable_group_general, plotly_theme):
    import plotly.graph_objects as go
    import plotly.express as px

    df = grade_data_general.copy()
    df["Wedge_Time"] = pd.to_datetime(df["Wedge_Time"])

    if diff:
        num = df.select_dtypes(include="number").diff()  # default axis=0, periods=1            
        df[num.columns] = num

        
    x_col = ivariable_x_general
    y1_col = ivariable_y_general
    y2_col = ivariable_y2_general
    group_col = ivariable_group_general

    traces = []

    # Flag for whether secondary Y should be included
    use_secondary_y = y2_col != ""

    if group_col == "":
        # No grouping
        traces.append(go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            mode='markers',
            name=y1_col,
            marker=dict(color='blue')
        ))

        if use_secondary_y:
            traces.append(go.Scatter(
                x=df[x_col],
                y=df[y2_col],
                mode='markers',
                name=y2_col,
                marker=dict(color='red', symbol='circle-open'),
                yaxis='y2'
            ))

    elif group_col in ["AB_Grade_ID", "target"]:
        # Grouping enabled
        unique_groups = df[group_col].unique()
        color_map = dict(zip(unique_groups, px.colors.qualitative.Plotly))

        for group in unique_groups:
            group_df = df[df[group_col] == group]

            # Primary Y-axis
            traces.append(go.Scatter(
                x=group_df[x_col],
                y=group_df[y1_col],
                mode='markers',
                name=f"{group} - {y1_col}",
                marker=dict(color=color_map[group]),
                legendgroup=group,
            ))

            # Secondary Y-axis (only if specified)
            if use_secondary_y:
                traces.append(go.Scatter(
                    x=group_df[x_col],
                    y=group_df[y2_col],
                    mode='markers',
                    name=f"{group} - {y2_col}",
                    marker=dict(color=color_map[group], symbol='circle-open'),
                    legendgroup=group,
                    yaxis='y2'
                ))
    else:
        
        # Primary Y-axis
        traces.append(go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            mode='markers',
            marker=dict(
                color=df[group_col],                 # Continuous variable for color
                colorscale='Viridis',               # Any continuous colorscale: Viridis, Plasma, Cividis, etc.
                colorbar=dict(title=group_col),     # Colorbar legend
                showscale=True                      # Show the color scale
            ),
            name=f"{y1_col}",
            showlegend=False                       # Legend would be redundant with continuous colorbar
        ))

        # Secondary Y-axis (only if specified)
        if use_secondary_y:
            traces.append(go.Scatter(
                x=df[x_col],
                y=df[y2_col],
                mode='markers',
                marker=dict(
                    color=df[group_col],
                    colorscale='Viridis',
                    colorbar=dict(title=group_col),
                    showscale=False,               # Only one colorbar is needed
                    symbol='circle-open'
                ),
                name=f"{y2_col}",
                showlegend=False,
                yaxis='y2'
            ))

    # Layout
    layout = go.Layout(
            title='Scatter Plot',
            xaxis=dict(title=x_col),
            yaxis=dict(title=y1_col),
            legend=dict(title=group_col if group_col else "Legend"),
        )

    if x_col=="Wedge_Time":
        layout.xaxis=dict(
            title=x_col,
            type='date',   # <-- important
            tickformat='%Y-%m-%d'  # customize as you like
        )

    # Only add secondary y-axis if needed
    if use_secondary_y:
        layout.yaxis2 = dict(
            title=y2_col,
            overlaying='y',
            side='right'
        )

    fig = go.Figure(data=traces, layout=layout)

    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
    return fig

def _processplot(grade_data_process_clustered, ivariable_x_process, ivariable_y_process, ivariable_group_process, plotly_theme):
    import plotly.express as px

    gdpc=grade_data_process_clustered.copy()        
    if ivariable_group_process != "":
        if ivariable_group_process=="cluster":
            gdpc["cluster"]=gdpc["cluster"].astype(str)
            fig = px.scatter(
                gdpc, 
                x=ivariable_x_process, 
                y=ivariable_y_process, 
                color="cluster",
                color_discrete_sequence=px.colors.qualitative.Alphabet
            )
        else:    
                fig = px.scatter(
                gdpc, 
                x=ivariable_x_process, 
                y=ivariable_y_process, 
                color=ivariable_group_process
            )
    else:
        fig = px.scatter(
            gdpc, 
            x=ivariable_x_process, 
            y=ivariable_y_process
        )
    # Optional: set figure layout to auto-size
    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True, template=plotly_theme)
    return fig

def describe_drilldown_row(row, decimals=2):
    import pandas as pd

    # --- grade text
    grade = row.get("AB_Grade_ID", None)
    grade_txt = "all considered grades" if str(grade).upper() == "ALL" else f"grade {grade}"
 
    # --- amount + direction
    pct = row.get("pct_cost", None)
    if pd.isna(pct):
        change_txt = "an unknown change"
        amount_txt = ""
    else:
        if pct > 0:
            change_txt = "an increase"
        elif pct < 0:
            change_txt = "a decrease"
        else:
            change_txt = "no change"
        amount_txt = f"{abs(pct):.{decimals}f}"
 
    # --- cost component (your example: TOTAL cost)
    component = row.get("cost", "TOTAL")
    component_txt = f"{component} cost"
 
    # --- resulting cost
    curr = row.get("current_cost", None)
    curr_txt = "an unknown value" if pd.isna(curr) else f"{curr:.{decimals}f}"
 
    # Optional extras if you want them
    prev = row.get("previous_cost", None)
    prev_part = "" if pd.isna(prev) else f" (previously {prev:.{decimals}f})"
 
    if pd.isna(pct):
        return f"For {grade_txt}, {component_txt} resulted in {curr_txt} €/t{prev_part}."
    if pct == 0:
        return f"For {grade_txt}, {component_txt} was unchanged at {curr_txt} €/t{prev_part}."
    return (
        f"For {grade_txt}, {component_txt} saw {change_txt} of {amount_txt} €/t, "
        f"resulting in {curr_txt} €/t."
    )


def build_drilldown_text(drilldown, mix_contribution=None, lang="en"):
    """
    Multiline Markdown-ready report text.

    Rules:
      - Normal case (no component breakdown):
          * print row-level grade descriptions
          * include 'Looking across...' summary for strongest grade increase/decrease (if applicable)
          * include ALL-only mix/process sentence (if mix_contribution provided)
      - Component breakdown case (and ONLY then):
          * ignore row-level grade descriptions entirely
          * ignore the 'Looking across...' grade summary entirely
          * report only the component breakdown analysis (excluding TOTAL / AVERAGE)
          * output order:
              1) Average change by component
              2) A more detailed breakdown... (largest avg increase/decrease + peak/trough grades)
          * still includes ALL-only mix/process sentence if applicable

    Units:
      - cost -> €/t
      - overprocessing -> %
    """
    import pandas as pd
    import re

    # ----------------------------
    # Language helpers
    # ----------------------------
    lang = (lang or "en").lower()

    def _t(en, de):
        return en if lang == "en" else de

    def _translate_row_text_de(text: str) -> str:
        """
        Conservative EN->DE phrase replacement for describe_drilldown_row() output.
        (No external translation service; only patterns we know.)
        """
        if not isinstance(text, str):
            text = str(text)

        # Ordered: longer/more specific first (IMPORTANT!)
        patterns = [
            # New items you reported
            (r"\bsaw a\b", "verzeichnete einen"),
            (r"\bresulting in\b", "was zu"),
            (r"\bresulting\b", "wodurch"),

            # "For ..." cases (more specific first!)
            (r"\bFor all considered grades\b", "Für alle betrachteten Sorten"),
            (r"\bFor the considered grades\b", "Für die betrachteten Sorten"),
            (r"\bFor all grades\b", "Für alle Sorten"),
            (r"\bFor the grade\b", "Für die Sorte"),
            (r"\bFor grade\b", "Für Sorte"),
            (r"\bFor\b", "Für"),

            # Existing items
            (r"\bconsidered grades\b", "betrachtete Sorten"),

            (r"\bLooking across individual grades\b", "Über alle einzelnen Sorten betrachtet"),
            (r"\bLooking across grades\b", "Über alle Sorten betrachtet"),
            (r"\bindividual grades\b", "einzelne Sorten"),
            (r"\bThese movements stand out and may require further investigation\b",
             "Diese Bewegungen stechen hervor und sollten ggf. weiter untersucht werden"),
            (r"\bmay require further investigation\b", "sollten ggf. weiter untersucht werden"),
            (r"\bfurther investigation\b", "weitere Untersuchung"),

            # cost / overprocessing words (in case they appear)
            (r"\boverprocessing\b", "Überverarbeitung"),
            (r"\bcost\b", "Kosten"),

            # common verbs/adjectives
            (r"\bwas observed for\b", "wurde beobachtet bei"),
            (r"\bwas driven by\b", "wurde verursacht durch"),
            (r"\bdriven by\b", "verursacht durch"),
            (r"\bdue to\b", "aufgrund von"),
            (r"\bexplained by\b", "erklärt durch"),
            (r"\bassociated with\b", "verbunden mit"),

            (r"\bstrongest\b", "stärkste"),
            (r"\bhigher\b", "höher"),
            (r"\blower\b", "niedriger"),
            (r"\bincrease\b", "Anstieg"),
            (r"\bdecrease\b", "Rückgang"),
            (r"\bimprovement\b", "Verbesserung"),
            (r"\bdeterioration\b", "Verschlechterung"),
            (r"\bmore expensive\b", "teurer"),
            (r"\bcheaper\b", "günstiger"),
            (r"\bunchanged\b", "unverändert"),
            (r"\bincreased\b", "gestiegen"),
            (r"\bdecreased\b", "gesunken"),

            # grade phrasing (plural first!)
            (r"\bgrades\b", "Sorten"),
            (r"\bgrade\b", "Sorte"),
        ]

        out = text
        for pat, sub in patterns:
            out = re.sub(pat, sub, out, flags=re.IGNORECASE)
        return out

    # ----------------------------
    # Detect metric + unit + component column
    # ----------------------------
    metric = "cost"
    if (
        "overprocessing" in drilldown.columns
        or drilldown.astype(str).apply(lambda s: s.str.lower().eq("overprocessing")).any().any()
    ):
        metric = "overprocessing"

    is_overprocessing = metric == "overprocessing"
    metric_noun_en = "overprocessing" if is_overprocessing else "cost"
    metric_noun_de = "Überverarbeitung" if is_overprocessing else "Kosten"
    metric_noun = _t(metric_noun_en, metric_noun_de)
    unit = "%" if is_overprocessing else "€/t"

    # Column that holds component names (TOTAL / AVERAGE / component labels)
    component_col = "overprocessing" if is_overprocessing else "cost"
    has_component_col = component_col in drilldown.columns

    # Breakdown detection:
    # - cost: any component != TOTAL
    # - overprocessing: any component != AVERAGE
    breakdown_flag_value = "AVERAGE" if is_overprocessing else "TOTAL"
    breakdown_exists = False
    if has_component_col:
        breakdown_exists = (~drilldown[component_col].astype(str).str.upper().eq(breakdown_flag_value)).any()

    # ----------------------------
    # Formatting helpers
    # ----------------------------
    def _fmt_signed(x, decimals=2):
        x = float(x)
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.{decimals}f} {unit}"

    def _fmt_abs(x, decimals=2):
        return f"{abs(float(x)):.{decimals}f} {unit}"

    def _cheaper_or_more_expensive(x):
        x = float(x)
        if x < 0:
            return _t("lower" if is_overprocessing else "cheaper",
                      "niedriger" if is_overprocessing else "günstiger")
        if x > 0:
            return _t("higher" if is_overprocessing else "more expensive",
                      "höher" if is_overprocessing else "teurer")
        return _t("unchanged", "unverändert")

    def _improvement_or_deterioration(x):
        x = float(x)
        if x < 0:
            return _t("an improvement in the process", "einer Verbesserung im Prozess")
        if x > 0:
            return _t("a deterioration in the process", "einer Verschlechterung im Prozess")
        return _t("no change in the process", "keiner Veränderung im Prozess")

    def _avg_direction_word(x):
        x = float(x)
        if x > 0:
            return _t("increased", "gestiegen")
        if x < 0:
            return _t("decreased", "gesunken")
        return _t("was unchanged", "unverändert geblieben")

    # ----------------------------
    # Detect ALL-only case (for mix/process line)
    # ----------------------------
    has_grade = "AB_Grade_ID" in drilldown.columns
    grades = drilldown["AB_Grade_ID"].astype(str).str.upper().unique().tolist() if has_grade else []
    all_only = has_grade and len(drilldown) == 1 and len(grades) == 1 and grades[0] == "ALL"

    extra_lines = []

    # Kept even in breakdown mode
    if all_only and mix_contribution is not None:
        mix, process = mix_contribution[0], mix_contribution[1]
        mix_dir = _cheaper_or_more_expensive(mix)
        proc_phrase = _improvement_or_deterioration(process)

        extra_lines.append(
            _t(
                f"Breaking down the overall {metric_noun_en} change, {_fmt_abs(mix)} was driven by a {mix_dir} mix of grades, "
                f"and {_fmt_abs(process)} was due to {proc_phrase}.",
                f"Bei der Aufschlüsselung der gesamten Veränderung der {metric_noun_de} ergibt sich, dass {_fmt_abs(mix)} durch eine {mix_dir} Sortenmischung verursacht wurden "
                f"und {_fmt_abs(process)} auf {proc_phrase} zurückzuführen sind."
            )
        )

    # ----------------------------
    # Component breakdown mode (ONLY when breakdown exists)
    # ----------------------------
    if breakdown_exists:
        required = {"pct_cost", "AB_Grade_ID", component_col}
        if not required.issubset(drilldown.columns):
            return "\n\n".join(extra_lines) if extra_lines else ""

        dd_comp = drilldown.copy()
        dd_comp[component_col] = dd_comp[component_col].astype(str)

        # Keep only component rows (exclude TOTAL/AVERAGE) and valid changes
        dd_comp = dd_comp[
            ~dd_comp[component_col].str.upper().eq(breakdown_flag_value)
        ].dropna(subset=["pct_cost"])

        if dd_comp.empty:
            return "\n\n".join(extra_lines) if extra_lines else ""

        # Average change per component
        avg_by_comp = dd_comp.groupby(component_col, as_index=False)["pct_cost"].mean()

        lines = []

        # 1) Average change by component
        lines.append(_t("Average change by component:", "Durchschnittliche Veränderung je Komponente:"))
        avg_sorted = avg_by_comp.copy()
        avg_sorted["abs_avg"] = avg_sorted["pct_cost"].abs()
        avg_sorted = avg_sorted.sort_values(["abs_avg", component_col], ascending=[False, True])

        for _, r in avg_sorted.iterrows():
            comp = r[component_col]
            avgv = float(r["pct_cost"])
            lines.append(
                _t(
                    f"- **{comp}** {_avg_direction_word(avgv)} on average ({_fmt_signed(avgv)}).",
                    f"- **{comp}** ist im Durchschnitt {_avg_direction_word(avgv)} ({_fmt_signed(avgv)})."
                )
            )

        # 2) A more detailed breakdown...
        avg_inc = avg_by_comp[avg_by_comp["pct_cost"] > 0]
        avg_dec = avg_by_comp[avg_by_comp["pct_cost"] < 0]

        detail_parts = []

        if not avg_inc.empty:
            row = avg_inc.loc[avg_inc["pct_cost"].idxmax()]
            comp = row[component_col]
            peak = dd_comp[dd_comp[component_col] == comp].loc[
                dd_comp[dd_comp[component_col] == comp]["pct_cost"].idxmax()
            ]
            detail_parts.append(
                _t(
                    f"the largest average increase comes from **{comp}** ({_fmt_signed(row['pct_cost'])} on average), "
                    f"with the highest increase in grade {peak['AB_Grade_ID']} ({_fmt_signed(peak['pct_cost'])})",
                    f"die größte durchschnittliche Zunahme stammt von **{comp}** (im Mittel {_fmt_signed(row['pct_cost'])}), "
                    f"mit dem höchsten Anstieg in Sorte {peak['AB_Grade_ID']} ({_fmt_signed(peak['pct_cost'])})"
                )
            )

        if not avg_dec.empty:
            row = avg_dec.loc[avg_dec["pct_cost"].idxmin()]  # most negative
            comp = row[component_col]
            trough = dd_comp[dd_comp[component_col] == comp].loc[
                dd_comp[dd_comp[component_col] == comp]["pct_cost"].idxmin()
            ]
            detail_parts.append(
                _t(
                    f"the largest average decrease comes from **{comp}** ({_fmt_signed(row['pct_cost'])} on average), "
                    f"with the biggest drop in grade {trough['AB_Grade_ID']} ({_fmt_signed(trough['pct_cost'])})",
                    f"die größte durchschnittliche Abnahme stammt von **{comp}** (im Mittel {_fmt_signed(row['pct_cost'])}), "
                    f"mit dem stärksten Rückgang in Sorte {trough['AB_Grade_ID']} ({_fmt_signed(trough['pct_cost'])})"
                )
            )

        if detail_parts:
            lines.append(
                _t(
                    "A more detailed breakdown by component shows that "
                    + " and ".join(detail_parts)
                    + ". These components stand out and may require further investigation.",
                    "Eine detailliertere Aufschlüsselung nach Komponenten zeigt, dass "
                    + " und ".join(detail_parts)
                    + ". Diese Komponenten stechen hervor und sollten ggf. weiter untersucht werden."
                )
            )

        return "\n\n".join(lines + extra_lines)

    # ----------------------------
    # Normal mode (no breakdown): row-level descriptions + 'Looking across...'
    # ----------------------------
    descriptions = drilldown.apply(describe_drilldown_row, axis=1).astype(str).tolist()

    # If overprocessing, adjust wording + units in row-level descriptions
    if is_overprocessing:
        descriptions = [
            re.sub(r"\bcost\b", _t("overprocessing", "Überverarbeitung"), d, flags=re.IGNORECASE).replace("€/t", "%")
            for d in descriptions
        ]

    # Translate row-level descriptions if German is requested.
    if lang == "de":
        descriptions = [_translate_row_text_de(d) for d in descriptions]

    # Multi-grade summary "Looking across..." (exclude ALL)
    dd_grades = drilldown.copy()
    if has_grade:
        dd_grades = dd_grades[dd_grades["AB_Grade_ID"].astype(str).str.upper() != "ALL"]

    if len(dd_grades) > 1 and {"pct_cost", "AB_Grade_ID"}.issubset(dd_grades.columns):
        dd_valid = dd_grades.dropna(subset=["pct_cost"])

        attention_parts = []
        if not dd_valid.empty:
            inc = dd_valid[dd_valid["pct_cost"] > 0]
            if not inc.empty:
                inc_row = inc.loc[inc["pct_cost"].idxmax()]
                attention_parts.append(
                    _t(
                        f"the strongest {metric_noun_en} increase was observed for grade {inc_row['AB_Grade_ID']} "
                        f"({_fmt_signed(inc_row['pct_cost'])})",
                        f"der stärkste {metric_noun_de}-Anstieg wurde für Sorte {inc_row['AB_Grade_ID']} beobachtet "
                        f"({_fmt_signed(inc_row['pct_cost'])})"
                    )
                )

            dec = dd_valid[dd_valid["pct_cost"] < 0]
            if not dec.empty:
                dec_row = dec.loc[dec["pct_cost"].idxmin()]
                attention_parts.append(
                    _t(
                        f"the strongest {metric_noun_en} decrease was observed for grade {dec_row['AB_Grade_ID']} "
                        f"({_fmt_signed(dec_row['pct_cost'])})",
                        f"der stärkste {metric_noun_de}-Rückgang wurde für Sorte {dec_row['AB_Grade_ID']} beobachtet "
                        f"({_fmt_signed(dec_row['pct_cost'])})"
                    )
                )

        if attention_parts:
            extra_lines.append(
                _t(
                    "Looking across individual grades, "
                    + " and ".join(attention_parts)
                    + ". These movements stand out and may require further investigation.",
                    "Über alle einzelnen Sorten betrachtet "
                    + " und ".join(attention_parts)
                    + ". Diese Bewegungen stechen hervor und sollten ggf. weiter untersucht werden."
                )
            )

    return "\n\n".join(descriptions + extra_lines)

def build_shapley_text(shapley_contrib, top_frac=0.20, lang="en"):
    """
    Build a human-readable, Markdown-ready summary of the main SHAP (Shapley) contributors.

    Parameters
    ----------
    shapley_contrib : pd.DataFrame
        Expected columns:
          - AB_Grade_ID
          - variable
          - contribution  (signed impact on predicted cost change, e.g. €/t)
          - value_change  (signed change in the feature value)
    top_frac : float
        Fraction of variables (per grade) to keep as "top contributors" by absolute contribution.
        Example: 0.20 keeps the top 20%.
    lang : {"en","de"}
        Output language. "en" keeps English. "de" outputs German.

    Returns
    -------
    str : multi-paragraph Markdown text (LaTeX-safe).
    """
    import pandas as pd
    import numpy as np
    import re

    lang = (lang or "en").lower()

    def _t(en, de):
        return en if lang == "en" else de

    df = shapley_contrib.copy()
    df = df[~df["variable"].astype(str).str.lower().eq("unknown")]

    required = {"AB_Grade_ID", "variable", "contribution", "value_change"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"shapley_contrib is missing required columns: {sorted(missing)}")

    # Clean + types
    df["variable"] = df["variable"].astype(str)
    df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")
    df["value_change"] = pd.to_numeric(df["value_change"], errors="coerce")
    df = df.dropna(subset=["contribution", "value_change"])

    if df.empty:
        return _t(
            "No SHAP contribution data available to summarise.",
            "Keine SHAP-Beitragsdaten zur Zusammenfassung verfügbar."
        )

    df["abs_contribution"] = df["contribution"].abs()

    def _pretty_var(name: str) -> str:
        # Make variable names more readable, without adding units.
        # Examples: "Starch_uptake__g/m2_" -> "Starch uptake"
        s = name
        s = s.replace("__", " ")
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if name.lower() == name or "_" in name:
            s = s[:1].upper() + s[1:]
        return s

    def _fmt_signed(x, decimals=2):
        x = float(x)
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.{decimals}f}"

    def _fmt_abs(x, decimals=2):
        return f"{abs(float(x)):.{decimals}f}"

    def _direction_word(x):
        x = float(x)
        if x > 0:
            return _t("increased", "gestiegen")
        if x < 0:
            return _t("decreased", "gesunken")
        return _t("was unchanged", "unverändert geblieben")

    # Build text per grade
    out_lines = []
    for grade, g in df.groupby("AB_Grade_ID", sort=False):
        g = g.sort_values("abs_contribution", ascending=False).copy()
        n = len(g)
        k = max(1, int(np.ceil(n * float(top_frac))))
        top = g.head(k).copy()

        inc = top[top["contribution"] > 0].copy()
        dec = top[top["contribution"] < 0].copy()

        header = _t(
            f"For grade {grade}, the cost change is mainly explained by the following factors (top {int(top_frac*100)}% by contribution):",
            f"Für Sorte {grade} wird die Kostenänderung hauptsächlich durch die folgenden Faktoren erklärt (Top {int(top_frac*100)}% nach Beitrag):"
        )
        out_lines.append(header)

        if not dec.empty:
            out_lines.append(_t("Cost-reducing drivers:", "Kostensenkende Treiber:"))
            for _, r in dec.iterrows():
                var = _pretty_var(r["variable"])
                out_lines.append(
                    f"- {var} {_direction_word(r['value_change'])} ({_fmt_signed(r['value_change'])})"
                    # f", which contributed to a reduction in cost of {_fmt_abs(r['contribution'])}."
                )

        if not inc.empty:
            out_lines.append(_t("Cost-increasing drivers:", "Kostentreibende Faktoren:"))
            for _, r in inc.iterrows():
                var = _pretty_var(r["variable"])
                out_lines.append(
                    f"- {var} {_direction_word(r['value_change'])} ({_fmt_signed(r['value_change'])})"
                    # f", which contributed to an increase in cost of {_fmt_abs(r['contribution'])}."
                )

        if inc.empty and dec.empty:
            out_lines.append(
                _t(
                    "No meaningful drivers were identified (all contributions were near zero).",
                    "Es wurden keine aussagekräftigen Treiber identifiziert (alle Beiträge lagen nahe bei null)."
                )
            )

        out_lines.append("")

    return "\n".join(out_lines).strip()
    




class GermanyTempService:
    def __init__(self, latitude=51.1657, longitude=10.4515, timezone="Europe/Berlin"):
        self.lat = latitude
        self.lon = longitude
        self.tz = timezone
        self.cache = {}   # {date -> pd.Series(hourly temps)}
 
    def _fetch_day(self, day):
        from datetime import timedelta
        import requests

        today = pd.Timestamp.now(tz=self.tz).date()
        use_archive = day < (today - timedelta(days=14))
 
        base_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            if use_archive
            else "https://api.open-meteo.com/v1/forecast"
        )
 
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": self.tz,
        }
 
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
 
        idx = pd.to_datetime(data["hourly"]["time"])
        temps = pd.Series(data["hourly"]["temperature_2m"], index=idx)
        return temps
 
    def get(self, dt):
        ts = pd.Timestamp(dt)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(self.tz).tz_localize(None)
 
        day = ts.date()
 
        if day not in self.cache:
            self.cache[day] = self._fetch_day(day)
 
        series = self.cache[day]
        pos = series.index.get_indexer([ts], method="nearest")[0]
        return float(series.iloc[pos])
    
def add_germany_temperature_column(
    df: pd.DataFrame,
    datetime_col: str,
    out_col: str = "ambient_temp_C",
    inplace: bool = False,
) -> pd.DataFrame | None:
    """
    Add a Germany ambient temperature column based on timestamps.
 
    If inplace=False (default): returns a modified copy.
    If inplace=True: modifies df in place and returns None.
    """
    service = GermanyTempService()
 
    if not inplace:
        df = df.copy()
 
    ts = pd.to_datetime(df[datetime_col])
 
    # Only query unique timestamps (huge speed-up)
    unique_ts = ts.unique()
    lookup = {t: service.get(t) for t in unique_ts}
 
    df[out_col] = ts.map(lookup)
 
    if not inplace:
        return df
    return None

def _lab_data(local, bucket, fs):
    if local:
        return pd.read_parquet(f"./data/optvsn_df.parquet",engine="fastparquet")
    else:
        return pd.read_parquet(f"s3://{bucket}/turnup/optvsn_df.parquet")

# class FeatureCreator(BaseEstimator, TransformerMixin):
#         """
#         Create predefined engineered features, selectively enabled via `features_to_create`.
    
#         Parameters
#         ----------
#         features_to_create : list[str] | None
#             Names of engineered features to create. If None, creates all known features.
#         errors : {"raise", "ignore"}
#             What to do if a requested feature cannot be created because required input columns
#             are missing (or division by zero etc.). "ignore" will skip that feature.
#         """
    
#         def __init__(self, features_to_create=None, features_to_keep=None, errors="raise"):
#             self.features_to_create = features_to_create
#             self.features_to_keep = features_to_keep
#             self.errors = errors
    
#         # ----- public sklearn API -----
    
#         def fit(self, X, y=None):
#             self.input_features_ = list(X.columns) if hasattr(X, "columns") else None
#             self.feature_names_in_ = self.input_features_
#             self._registry_ = self._build_registry()
#             self._deps_ = self._build_dependencies()
#             return self
    
#         def transform(self, X):
#             X_df = self._to_dataframe(X)
    
#             # Determine which engineered features to create
#             requested = (
#                 list(self._registry_.keys())
#                 if self.features_to_create is None
#                 else list(self.features_to_create)
#             )
    
#             # Expand prerequisites automatically
#             to_make = self._expand_with_deps(requested)
    
#             # Create them in dependency order
#             created = []
#             for name in to_make:
#                 if name in X_df.columns:
#                     # Don't overwrite if already exists
#                     continue
    
#                 fn = self._registry_.get(name)
#                 if fn is None:
#                     msg = f"Unknown feature requested: {name}. Available: {sorted(self._registry_.keys())}"
#                     if self.errors == "raise":
#                         raise ValueError(msg)
#                     else:
#                         continue
    
#                 try:
#                     X_df = fn(X_df)
#                     created.append(name)
#                 except Exception as e:
#                     if self.errors == "raise":
#                         raise
#                     # ignore: skip feature
#                     continue
    
#             self.created_features_ = created            
#             return X_df[self.features_to_keep]
    
#         def get_feature_names_out(self, input_features=None):
            
#             out = self.features_to_keep
            
#             return np.array(out, dtype=object)
    
#         # ----- internals -----
    
#         def _to_dataframe(self, X):
#             if isinstance(X, pd.DataFrame):
#                 return X.copy()
#             if self.input_features_ is None:
#                 raise ValueError(
#                     "Input is not a DataFrame and feature names are unknown. "
#                     "Pass a DataFrame into the pipeline or provide columns another way."
#                 )
#             return pd.DataFrame(X, columns=self.input_features_)
    
#         def _expand_with_deps(self, requested):
#             # simple DFS expansion
#             seen = set()
#             order = []
    
#             def visit(f):
#                 if f in seen:
#                     return
#                 seen.add(f)
#                 for d in self._deps_.get(f, []):
#                     visit(d)
#                 order.append(f)
    
#             for f in requested:
#                 visit(f)
#             return order

#         def features_required(self):    
#             dd = []
#             for v in self.features_to_create:
#                 if v == "Fibre__g/m2_":
#                     dd.append("Current_basis_weight")
#                     dd.append("Current_reel_moisture_average(reel)")
#                     dd.append("Starch_uptake__g/m2_")
#                 elif v == "concentration_starch":
#                     dd.append("Flow_starch_main_line_to_working_tank_1~^0")
#                     dd.append("Flow_starch_main_line_to_working_tank_2~^0")
#                     dd.append("Dilution_water_working_tank_1")
#                     dd.append("Dilution_water_working_tank_2")
#                 elif v == "Water_flow_Predryer":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Moisture_out_of_PreDryer")
#                 elif v == "Water_flow_Predryer":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Moisture_out_of_PreDryer")
#                 elif v == "Water_flow_Afterdryer":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Moisture_after_SpeedSizer")
#                     dd.append("Actual_moisture")
#                 elif v == "Water_flow":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Moisture_out_of_PreDryer")
#                     dd.append("Moisture_after_SpeedSizer")
#                     dd.append("Actual_moisture")
#                 elif v == "flow_diluted_starch":
#                     dd.append("Flow_starch_main_line_to_working_tank_2~^0")
#                     dd.append("concentration_starch_working_tank_2")
#                     dd.append("Flow_starch_main_line_to_working_tank_1~^0")
#                     dd.append("concentration_starch_working_tank_1")
#                 elif v =="Water_flow_Afterdryer_input":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Flow_starch_main_line_to_working_tank_2~^0")
#                     dd.append("concentration_starch_working_tank_2")
#                     dd.append("Flow_starch_main_line_to_working_tank_1~^0")
#                     dd.append("concentration_starch_working_tank_1")
#                 elif v =="Water_flow_Afterdryer_output":
#                     dd.append("Current_basis_weight")
#                     dd.append("Speed_PD1")
#                     dd.append("Current_reel_width")
#                     dd.append("Actual_moisture")
#             return list(set(dd))
                    

    
#         def _build_dependencies(self):
#             """
#             Declare which engineered features depend on which other engineered features.
#             Raw input-column requirements are enforced by the feature function itself.
#             """
#             return {
#                 "Water_flow": ["Water_flow_Predryer", "Water_flow_Afterdryer"],
#                 "Water_flow_Afterdryer_input": ["flow_diluted_starch"],
#                 # flow_diluted_starch depends on concentrations, which you already have as columns:
#                 # "flow_diluted_starch": ["concentration_starch_working_tank_1", "concentration_starch_working_tank_2"]
#                 # But those look like raw columns, not engineered features, so we don't list them here.
#             }
    
#         def _build_registry(self):
#             """
#             Map engineered feature name -> function that takes df and returns df with the feature added.
#             Keep functions small and single-responsibility.
#             """
    
#             def add_concentration_starch(df):
#                 # your formula (note: you referenced concentration_starch_working_tank_2 elsewhere;
#                 # I'm keeping your shown name "concentration_starch" here)
#                 df = df.copy()
#                 num = df["Flow_starch_main_line_to_working_tank_1~^0"] + df["Flow_starch_main_line_to_working_tank_2~^0"]
#                 den = (
#                     df["Dilution_water_working_tank_2"]
#                     + df["Dilution_water_working_tank_1"]
#                     + df["Flow_starch_main_line_to_working_tank_1~^0"]
#                     + df["Flow_starch_main_line_to_working_tank_2~^0"]
#                 )
#                 df["concentration_starch"] = num / den
#                 return df
    
#             def add_fibre(df):
#                 df = df.copy()
#                 df["Fibre__g/m2_"] = (
#                     df["Current_basis_weight"] * (1 - df["Current_reel_moisture_average(reel)"] / 100)
#                     - df["Starch_uptake__g/m2_"]
#                 )
#                 return df
    
#             def add_water_flow_predryer(df):
#                 df = df.copy()
#                 df["Water_flow_Predryer"] = (
#                     df["Current_basis_weight"]
#                     * df["Speed_PD1"]
#                     * df["Current_reel_width"]
#                     * (100 - 35 - df["Moisture_out_of_PreDryer"])
#                     * 60
#                     / 1e10
#                 )
#                 return df
    
#             def add_water_flow_afterdryer(df):
#                 df = df.copy()
#                 df["Water_flow_Afterdryer"] = (
#                     df["Current_basis_weight"]
#                     * df["Speed_PD1"]
#                     * df["Current_reel_width"]
#                     * (df["Moisture_after_SpeedSizer"] - df["Actual_moisture"])
#                     * 60
#                     / 1e10
#                 )
#                 return df
    
#             def add_water_flow(df):
#                 df = df.copy()
#                 df["Water_flow"] = df["Water_flow_Predryer"] + df["Water_flow_Afterdryer"]
#                 return df
    
#             def add_flow_diluted_starch(df):
#                 df = df.copy()
#                 df["flow_diluted_starch"] = (
#                     df["Flow_starch_main_line_to_working_tank_2~^0"] / df["concentration_starch_working_tank_2"]
#                     + df["Flow_starch_main_line_to_working_tank_1~^0"] / df["concentration_starch_working_tank_1"]
#                 )
#                 return df
    
#             def add_water_flow_afterdryer_input(df):
#                 df = df.copy()
#                 df["Water_flow_Afterdryer_input"] = (
#                     df["Current_basis_weight"]
#                     * df["Speed_PD1"]
#                     * df["Current_reel_width"]
#                     * df["flow_diluted_starch"]
#                     * 60
#                     / 1e10
#                 )
#                 return df
    
#             def add_water_flow_afterdryer_output(df):
#                 df = df.copy()
#                 df["Water_flow_Afterdryer_output"] = (
#                     df["Current_basis_weight"]
#                     * df["Speed_PD1"]
#                     * df["Current_reel_width"]
#                     * df["Actual_moisture"]
#                     * 60
#                     / 1e10
#                 )
#                 return df
    
#             return {
#                 "concentration_starch": add_concentration_starch,
#                 "Fibre__g/m2_": add_fibre,
#                 "Water_flow_Predryer": add_water_flow_predryer,
#                 "Water_flow_Afterdryer": add_water_flow_afterdryer,
#                 "Water_flow": add_water_flow,
#                 "flow_diluted_starch": add_flow_diluted_starch,
#                 "Water_flow_Afterdryer_input": add_water_flow_afterdryer_input,
#                 "Water_flow_Afterdryer_output": add_water_flow_afterdryer_output,
#             }
class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that creates predefined engineered features.
    Only the features listed in `features_to_create` are created (plus any prerequisites).
 
    Parameters
    ----------
    features_to_create : list[str] | None    
        Names of engineered features to create. If None, creates all known engineered features.
    features_to_keep : list[str] | None
        Names of features to keep.
    errors : {"raise", "ignore"}
        If "raise", raise if a feature cannot be computed (missing columns, division by zero, etc.).
        If "ignore", silently skip that feature.
    copy : bool
        If True, works on a copy of the input DataFrame.
    """

    def __init__(self, features_to_create=None, features_to_keep=None, errors="raise", copy=True):
        self.features_to_create = features_to_create
        self.features_to_keep = features_to_keep
        self.errors = errors
        self.copy=copy

    def features_required(self):    
            dd = []
            for v in self.features_to_create:
                if v == "Fibre__g/m2_":
                    dd.append("Current_basis_weight")
                    dd.append("Current_reel_moisture_average(reel)")
                    dd.append("Starch_uptake__g/m2_")
                # elif v == "concentration_starch":
                #     dd.append("Flow_starch_main_line_to_working_tank_1~^0")
                #     dd.append("Flow_starch_main_line_to_working_tank_2~^0")
                #     dd.append("Dilution_water_working_tank_1")
                #     dd.append("Dilution_water_working_tank_2")
                elif v == "Water_flow_Predryer":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Moisture_out_of_PreDryer")
                elif v == "Water_flow_Predryer":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Moisture_out_of_PreDryer")
                elif v == "Water_flow_Afterdryer":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Moisture_after_SpeedSizer")
                    dd.append("Actual_moisture")
                elif v == "Water_flow":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Moisture_out_of_PreDryer")
                    dd.append("Moisture_after_SpeedSizer")
                    dd.append("Actual_moisture")
                elif v == "flow_diluted_starch":
                    #dd.append("Flow_starch_main_line_to_working_tank_2~^0")
                    dd.append("concentration_starch_working_tank_2")
                    #dd.append("Flow_starch_main_line_to_working_tank_1~^0")
                    dd.append("concentration_starch_working_tank_1")
                    dd.append("Starch_uptake_by_paper_Bottom_Roll__g/m2_")
                    dd.append("Starch_uptake_by_paper_Top_Roll__g/m2_")

                elif v == "flow_diluted_starch_index":
                    dd.append("Flow_starch_main_line_to_working_tank_2~^0")
                    dd.append("concentration_starch_working_tank_2")
                    dd.append("Flow_starch_main_line_to_working_tank_1~^0")
                    dd.append("concentration_starch_working_tank_1")
                    dd.append("Production_Rate__T/h_")
                elif v =="Water_flow_Afterdryer_input":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Flow_starch_main_line_to_working_tank_2~^0")
                    dd.append("concentration_starch_working_tank_2")
                    dd.append("Flow_starch_main_line_to_working_tank_1~^0")
                    dd.append("concentration_starch_working_tank_1")
                elif v =="Water_flow_Afterdryer_output":
                    dd.append("Current_basis_weight")
                    dd.append("Speed_PD1")
                    dd.append("Current_reel_width")
                    dd.append("Actual_moisture")
            return list(set(dd))
 
    # -------------------------
    # sklearn API
    # -------------------------
    def fit(self, X, y=None):
        import numpy as np

        # Keep track of input feature names (sklearn convention)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        else:
            # If X is numpy and you want names, pass a DataFrame to the pipeline,
            # or wrap numpy into a DataFrame before fitting.
            self.feature_names_in_ = None
        return self
 
    def transform(self, X):
        # Allow transform to be called even if someone forgot fit()
        if not hasattr(self, "feature_names_in_"):
            self.fit(X)
 
        X_df = self._to_dataframe(X)
        if self.copy:
            X_df = X_df.copy()
 
        # Which engineered features were requested?
        registry = self._registry()
        requested = list(registry.keys()) if self.features_to_create is None else list(self.features_to_create)
 
        # Expand dependencies
        to_make = self._expand_with_deps(requested)
 
        created = []
        for name in to_make:
            if name in X_df.columns:
                continue  # do not overwrite
 
            fn = registry.get(name)
            if fn is None:
                msg = f"Unknown engineered feature '{name}'. Available: {sorted(registry.keys())}"
                if self.errors == "raise":
                    raise ValueError(msg)
                else:
                    continue
 
            try:
                X_df = fn(X_df)
                created.append(name)
            except Exception as e:
                if self.errors == "raise":
                    raise
                # ignore
                continue
 
        self.created_features_ = created
        return X_df[self.features_to_keep]
 
    def get_feature_names_out(self, input_features=None):
        import numpy as np
        # If input_features not given, use what we saw at fit time
        # if input_features is None:
        #     if getattr(self, "feature_names_in_", None) is not None:
        #         input_features = list(self.feature_names_in_)
        #     else:
        #         input_features = []
 
        # registry = self._registry()
        # requested = list(registry.keys()) if self.features_to_create is None else list(self.features_to_create)
        # requested = self._expand_with_deps(requested)
 
        # out = list(input_features)
        # for f in requested:
        #     if f not in out:
        #         out.append(f)
        # return np.array(out, dtype=object)
        return np.array(self.features_to_keep, dtype=object)
 
    # -------------------------
    # Internal helpers
    # -------------------------
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if getattr(self, "feature_names_in_", None) is None:
            raise ValueError(
                "FeatureCreator received a numpy array without known column names. "
                "Pass a pandas DataFrame into the pipeline (recommended) or ensure "
                "feature_names_in_ is set by fitting with a DataFrame first."
            )
        return pd.DataFrame(X, columns=list(self.feature_names_in_))
 
    @classmethod
    def _deps(cls):
        # Dependencies between engineered features (engineered -> engineered)
        return {
            "Water_flow": ["Water_flow_Predryer", "Water_flow_Afterdryer"],
            "Water_flow_Afterdryer_input": ["flow_diluted_starch"],
        }
 
    def _expand_with_deps(self, requested):
        deps = self._deps()
        seen = set()
        order = []
 
        def visit(f):
            if f in seen:
                return
            seen.add(f)
            for d in deps.get(f, []):
                visit(d)
            order.append(f)
 
        for f in requested:
            visit(f)
        return order
 
    # -------------------------
    # Engineered feature functions
    # (STATIC / CLASS METHODS ONLY -> picklable)
    # -------------------------
    # @staticmethod
    # def _add_concentration_starch(df: pd.DataFrame) -> pd.DataFrame:
    #     df = df.copy()
    #     num = df["Flow_starch_main_line_to_working_tank_1~^0"] + df["Flow_starch_main_line_to_working_tank_2~^0"]
    #     den = (
    #         df["Dilution_water_working_tank_2"]
    #         + df["Dilution_water_working_tank_1"]
    #         + df["Flow_starch_main_line_to_working_tank_1~^0"]
    #         + df["Flow_starch_main_line_to_working_tank_2~^0"]
    #     )
    #     df["concentration_starch"] = num / den
    #     return df
 
    @staticmethod
    def _add_fibre(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Fibre__g/m2_"] = (
            df["Current_basis_weight"] * (1 - df["Current_reel_moisture_average(reel)"] / 100)
            - df["Starch_uptake__g/m2_"]
        )
        return df
 
    @staticmethod
    def _add_water_flow_predryer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_flow_Predryer"] = (
            df["Current_basis_weight"]
            * df["Speed_PD1"]
            * df["Current_reel_width"]
            * (100 - 35 - df["Moisture_out_of_PreDryer"])
            * 60
            / 1e10
        )
        return df
 
    @staticmethod
    def _add_water_flow_afterdryer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_flow_Afterdryer"] = (
            df["Current_basis_weight"]
            * df["Speed_PD1"]
            * df["Current_reel_width"]
            * (df["Moisture_after_SpeedSizer"] - df["Actual_moisture"])
            * 60
            / 1e10
        )
        return df
 
    @staticmethod
    def _add_water_flow(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_flow"] = df["Water_flow_Predryer"] + df["Water_flow_Afterdryer"]
        return df
 
    @staticmethod
    def _add_flow_diluted_starch(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # df["flow_diluted_starch"] = (
        #     df["Flow_starch_main_line_to_working_tank_2~^0"] / df["concentration_starch_working_tank_2"]
        #     + df["Flow_starch_main_line_to_working_tank_1~^0"] / df["concentration_starch_working_tank_1"]
        # )
        df["flow_diluted_starch"] = (
            df["Starch_uptake_by_paper_Top_Roll__g/m2_"] / df["concentration_starch_working_tank_2"]
            + df["Starch_uptake_by_paper_Bottom_Roll__g/m2_"] / df["concentration_starch_working_tank_1"]
        )
        return df
    
    @staticmethod
    def _add_flow_diluted_starch_index(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["flow_diluted_starch_index"] = (
            (df["Flow_starch_main_line_to_working_tank_2~^0"] / df["concentration_starch_working_tank_2"]
            + df["Flow_starch_main_line_to_working_tank_1~^0"] / df["concentration_starch_working_tank_1"])/df['Production_Rate__T/h_']
        )
        return df
 
    @staticmethod
    def _add_water_flow_afterdryer_input(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_flow_Afterdryer_input"] = (
            df["Current_basis_weight"]
            * df["Speed_PD1"]
            * df["Current_reel_width"]
            * df["flow_diluted_starch"]
            * 60
            / 1e10
        )
        return df
 
    @staticmethod
    def _add_water_flow_afterdryer_output(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_flow_Afterdryer_output"] = (
            df["Current_basis_weight"]
            * df["Speed_PD1"]
            * df["Current_reel_width"]
            * df["Actual_moisture"]
            * 60
            / 1e10
        )
        return df
 
    @classmethod
    def _registry(cls):
        # IMPORTANT: references are to class methods (picklable), not local functions
        return {
            #"concentration_starch": cls._add_concentration_starch,
            "Fibre__g/m2_": cls._add_fibre,
            "Water_flow_Predryer": cls._add_water_flow_predryer,
            "Water_flow_Afterdryer": cls._add_water_flow_afterdryer,
            "Water_flow": cls._add_water_flow,
            "flow_diluted_starch": cls._add_flow_diluted_starch,
            "flow_diluted_starch_index": cls._add_flow_diluted_starch_index,
            "Water_flow_Afterdryer_input": cls._add_water_flow_afterdryer_input,
            "Water_flow_Afterdryer_output": cls._add_water_flow_afterdryer_output,
        }