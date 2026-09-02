"""
Configuration module for the paper mill cost optimization training pipeline.

Contains target-to-feature mappings (control_vars) and variable group definitions.
"""

# =============================================================================
# Target -> predictor feature mappings
# =============================================================================

CONTROL_VARS = {
    "Steam__kWh/T_": [
        "Production_Rate__T/h_",
        "Water_Predryer",
        "dewatering",
        "Draw_PS-PD1",
        "Moisture_out_of_PreDryer",
        "Draw_WS-PS",
        "PickUp_Tension",
        "Vacuum_presszone_of_suction-press_roll",
        "Vacuum_uhle-box_Pick-Up",
        "Vacuum_uhle-box_bottom_felt",
        "Linepressure_1st_press_FS__bar_",
        "Linepressure_2nd_press_FS__bar_",
        "Linepressure_1st_press_DS__bar_",
        "Linepressure_2nd_press_DS__bar_",
        "Linepressure_shoe_press__bar_",
        "Vacuum_Zone_1_PickUp",
        "ambient_temp_C",
        "Dewatering_top_wire_suction_box_zone_2",
        "Vacuum_suction_box_9",
        "Vacuum_wet_suction_box",
        "Vacuum_sheet_seperator_box",
        "Vacuum_suction_box_10",
        "Vacuum_suction_box_11",
        "Vacuum_wire_suction_box_1",
        "Vacuum_wire_suction_box_2",
        "White_water_temperature",
        "Conductivity_white_water_B46",
        "Top_wire_tenstion",
        "pH_measurement_white_water_B41",
        "Jet/wire_ratio",
        "Lip_settings",
        "Top_Felt_Tension",
        "Bottom_wire_tension",
        "Bottom_Felt_Tension",
        "PD4_fabric_tension",
        "PD1_Fabric_tension",
        "PD2_Fabric_tension",
        "PD5_fabric_tension_top",
        "PD5_fabric_tension_bottom",
        "pH-Messung_Verdünnungswasser__2..12_pH_",
        "Dissolved_gas_after_stock_deculator_measurement_1",
        "Dissolved_gas_before_dilution_water_deculator",
        "Air_pressure_of_rod_clamping_hose_Top_Roll",
        "Free_gas_before_dilution_water_deculator_measurement_2",
        "Free_gas_before_dilution_water_deculator_measurement_1~^0",
        "Free_gas_after_stock_deculator~^0",
        "Vacuum_top_wire_suction_box_zone_2",
        "Vacuum_formning_roll",
        "Vacuum_top_wire_suction_box_zone_1",
        "Airturn_pillow_pressure",
        "Dewatering_Suction_Press_Roll",
        "Dewatering_First_Press_Roll",
        "Steam_temperature_for_PM",
        "Steam_pressure_for_PM",
        "Dewatering_Shoe_press",
        "Total_Dewatering_Press",
        "Uhle_box_1_flow___l/min_",
        "Stock_deculator_temperature",
        "Stock_deculator_pressure",
        "Inlet_Air_2_Temperature",
        "Inlet_Air_1_Temperature",
        "DG1_temperature_Inlet_Air",
        "DG2_temperature_Inlet_Air",
        "DG3_temperature_Inlet_Air",
        "DG1_Moisture_content_Outlet_Air",
        "DG2_Moisture_content_Outlet_Air",
        "DG3_Moisture_content_Outlet_Air",
        "DG1_Ventilator_Revolution_Output",
        "DG2_Ventilator_Revolution_Output",
        "DG3_Ventilator_Revolution_Output",
        "fibre_short/long",
        "retention",
        #"diluted_starch",
        "Water_Afterdryer_output",
        "Starch_uptake__g/m2_",
        "Web_tension_AD6",
        "Moisture_out_of_PreDryer",
        "DG4_Temperature_Inlet_Air",
        "DG5_Temperature_Inlet_Air",
        "DG4_Moisture_content_Outlet_Air",
        "DG5_Moisture_content_Outlet_Air",
        "AD7_fabric_tension_bottom",
        "AD6_fabric_tension",
        "CO2_mass_flow__g/T_",
        "Act_Deaerator_mass_flow__g/T_",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_",
        "Natriumhydroxide_mass_flow__g/T_",
        "Fixative_2_mass_flow__g/T_",
    ],
    "Electricity__kWh/T_": [
        "retention",
        "dewatering",
        "Current_basis_weight",
        "Speed",
        "Current_reel_moisture_average(reel)",
        "SpeedSizer_Linepressure_DS",
        "SpeedSizer_Linepressure_FS",
        "Consistency_starch_main_line",
        "Moisture_out_of_PreDryer",
        "PickUp_Tension",
        "Vacuum_presszone_of_suction-press_roll",
        "Vacuum_uhle-box_Pick-Up",
        "Vacuum_uhle-box_bottom_felt",
        "Linepressure_1st_press_FS__bar_",
        "Linepressure_2nd_press_FS__bar_",
        "Linepressure_1st_press_DS__bar_",
        "Linepressure_2nd_press_DS__bar_",
        "Linepressure_shoe_press__bar_",
        "Dewatering_top_wire_suction_box_zone_2",
        "Vacuum_suction_box_9",
        "Vacuum_wet_suction_box",
        "Vacuum_sheet_seperator_box",
        "Vacuum_suction_box_10",
        "Vacuum_suction_box_11",
        "Vacuum_wire_suction_box_1",
        "Vacuum_wire_suction_box_2",
        "White_water_temperature",
        "Conductivity_white_water_B46",
        "Top_wire_tenstion",
        "pH_measurement_white_water_B41",
        "Jet/wire_ratio",
        "Lip_settings",
        "CO2_mass_flow__g/T_",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_",
        "Thick_Stock_Consistency__%_",
        "Act_Deaerator_mass_flow__g/T_",
        "Natriumhydroxide_mass_flow__g/T_",
    ],
    "Electrical_power_MW": [
        "retention",
        "dewatering",
        "Current_basis_weight",
        "Speed",
        "Current_reel_moisture_average(reel)",
        "SpeedSizer_Linepressure_DS",
        "SpeedSizer_Linepressure_FS",
        "Consistency_starch_main_line",
        "Moisture_out_of_PreDryer",
        "PickUp_Tension",
        "Vacuum_presszone_of_suction-press_roll",
        "Vacuum_uhle-box_Pick-Up",
        "Vacuum_uhle-box_bottom_felt",
        "Linepressure_1st_press_FS__bar_",
        "Linepressure_2nd_press_FS__bar_",
        "Linepressure_1st_press_DS__bar_",
        "Linepressure_2nd_press_DS__bar_",
        "Linepressure_shoe_press__bar_",
        "Dewatering_top_wire_suction_box_zone_2",
        "Vacuum_suction_box_9",
        "Vacuum_wet_suction_box",
        "Vacuum_sheet_seperator_box",
        "Vacuum_suction_box_10",
        "Vacuum_suction_box_11",
        "Vacuum_wire_suction_box_1",
        "Vacuum_wire_suction_box_2",
        "White_water_temperature",
        "Conductivity_white_water_B46",
        "Top_wire_tenstion",
        "pH_measurement_white_water_B41",
        "Jet/wire_ratio",
        "Lip_settings",
        "CO2_mass_flow__g/T_",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_",
        "Thick_Stock_Consistency__%_",
        "Act_Deaerator_mass_flow__g/T_",
        "Natriumhydroxide_mass_flow__g/T_",
    ],
    "Starch_uptake__g/m2_": [
        "Rod_pressure_Top_Roll",
        "Rod_Pressure_Bottom_Roll",
        "Speed_Size_Press",
        "Current_basis_weight",
        "concentration_starch_working_tank_1",
        "Temperature_starch_working_tank_1",
        "concentration_starch_working_tank_2",
        "Temperature_starch_working_tank_2",
        "Current_reel_moisture_average(SpeedSizer)",
        "Moisture_after_SpeedSizer",
        "retention",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_",
        "Current_reel_moisture_average(reel)",
        "Moisture_out_of_PreDryer",
        "Jet/wire_ratio",
        "Lip_settings",
        "White_water_temperature",
        "Conductivity_white_water_B46",
        "Top_wire_tenstion",
        "pH_measurement_white_water_B41",
        "SpeedSizer_Linepressure_DS",
        "SpeedSizer_Linepressure_FS",
    ],
    'Starch_uptake_by_paper_Bottom_Roll__g/m2_' :[
        'Viscosity_for_working_tank_bottom_roll',
        'CO2_mass_flow__g/T_',
        "grammage",
        "delta_basis_weight",
        "inv_Rod_Pressure_Bottom_Roll",
        #"Rod_Pressure_Bottom_Roll",
        "Speed_Size_Press",
        #"Current_basis_weight",
        "concentration_starch_working_tank_1",   
        "Temperature_starch_working_tank_1",       
        'Moisture_after_SpeedSizer',
        "retention",
        "Retention_Aid_mass_flow__g/T_",
        # "Bentonite_1_mass_flow__g/T_", 
        # "Bentonite_2_mass_flow__g/T_", 
        # "Current_reel_moisture_average(reel)",
        "Moisture_out_of_PreDryer",
        "Jet/wire_ratio", 
        "Lip_settings", 
        "White_water_temperature", # Forming Wire
        "Conductivity_white_water_B46", # Forming Wire
        "Top_wire_tenstion", # Forming Wire
        "pH_measurement_white_water_B41", # Forming Wire
        "SpeedSizer_Linepressure_DS", # Size Press
        "SpeedSizer_Linepressure_FS", # Size Press
        'Linepressure_1st_press_FS__bar_', 
        'Linepressure_2nd_press_FS__bar_', 
        'Linepressure_1st_press_DS__bar_', 
        'Linepressure_2nd_press_DS__bar_', 
        'Linepressure_shoe_press__bar_',
        'CO2_mass_flow__g/T_',
        "Retention_Aid_mass_flow__g/T_", # Approach Flow
        "Bentonite_1_mass_flow__g/T_", # Approach Flow
        "Bentonite_2_mass_flow__g/T_", # Approach Flow
        "Thick_Stock_Consistency__%_", # Approach Flow
        "Act_Deaerator_mass_flow__g/T_",
        "Natriumhydroxide_mass_flow__g/T_",
    ],
    'Starch_uptake_by_paper_Top_Roll__g/m2_':[
        'Viscosity_for_working_tank_bottom_roll',
        'CO2_mass_flow__g/T_',
        "grammage",
        "delta_basis_weight",
        "inv_Rod_pressure_Top_Roll",
        #"Rod_pressure_Top_Roll",
        #"square_Rod_pressure_Top_Roll",
        "Speed_Size_Press",
        #"Current_basis_weight",
        "concentration_starch_working_tank_2",   
        "Temperature_starch_working_tank_2",   
        'Moisture_after_SpeedSizer',    
        "retention",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_", 
        "Bentonite_2_mass_flow__g/T_",     
        "Moisture_out_of_PreDryer",
        "Jet/wire_ratio", 
        "Lip_settings", 
        "White_water_temperature", # Forming Wire
        "Conductivity_white_water_B46", # Forming Wire
        "Top_wire_tenstion", # Forming Wire
        "pH_measurement_white_water_B41", # Forming Wire
        "SpeedSizer_Linepressure_DS", # Size Press
        "SpeedSizer_Linepressure_FS", # Size Press
        'Linepressure_1st_press_FS__bar_', 'Linepressure_2nd_press_FS__bar_', 'Linepressure_1st_press_DS__bar_', 'Linepressure_2nd_press_DS__bar_', 'Linepressure_shoe_press__bar_',
        'CO2_mass_flow__g/T_',
        "Retention_Aid_mass_flow__g/T_", # Approach Flow
        "Bentonite_1_mass_flow__g/T_", # Approach Flow
        "Bentonite_2_mass_flow__g/T_", # Approach Flow
        "Thick_Stock_Consistency__%_", # Approach Flow
        "Act_Deaerator_mass_flow__g/T_",
        "Natriumhydroxide_mass_flow__g/T_",
    ],

    "MBS_SCT_CD": [
        "retention",
        "grammage",
        "delta_basis_weight",
        "Speed",
        "Moisture_after_SpeedSizer",
        "Current_reel_moisture_average(reel)",
        "SpeedSizer_Linepressure_DS",
        "SpeedSizer_Linepressure_FS",
        "Viscosity_for_working_tank_bottom_roll",
        "Starch_uptake__g/m2_",
        "Consistency_starch_main_line",
        "Draw_PD5-SS", "Draw_AD7-PR", "Draw_AD6-AD7",
        "Draw_PD4-PD5", "Draw_WS-PS", "Draw_PD1-PD2",
        "Draw_PD3-PD4", "Draw_SS-AD6", "Draw_PS-PD1", "Draw_PD2-PD3",
        "Linepressure_1st_press_FS__bar_",
        "Linepressure_2nd_press_FS__bar_",
        "Linepressure_1st_press_DS__bar_",
        "Linepressure_2nd_press_DS__bar_",
        "Linepressure_shoe_press__bar_",
        "pH_measurement_white_water_B41",
        "Headbox_consistency",
        "Lip_settings",
        "Jet/wire_ratio",
        "Retention_Aid_mass_flow__g/T_",
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_",
        "CO2_mass_flow__g/T_",
        "fibre_short/long",
        "White_water_temperature",
        "Conductivity_white_water_B46",
        "pH_measurement_white_water_B41",
        "Current_reel_width",
    ],
    "Steam_power" : [ 
        "Water_flow_Predryer",
        "Water_flow_Afterdryer_input",
        "Water_flow_Afterdryer_output",
          

        "dewatering",

        "Draw_PS-PD1", #Pre-dryer
        # "Draw_PD2-PD3", #Pre-dryer
        # "Draw_PD4-PD5", #Pre-dryer 
        # "Draw_PD3-PD4", #Pre-dryer 
        # "Draw_PD1-PD2", #Pre-dryer
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
        "Vacuum_Zone_1_PickUp",# Press Section
        
        'Cylinder_1_differential_pressure',
        'Cylinder_1_steam_pressure',
        'Cylinder_2_steam_pressure',
        'Cylinder_3_differential_pressure',
        'Cylinder_2_differential_pressure',
        'Cylinder_3_steam_pressure',
        'Cylinder_4_differential_pressure',
        'Cylinder_5_steam_pressure',
        'Cylinder_4_steam_pressure',
        'Cylinder_5_differential_pressure',
        'Cylinder_1-5_steam_pressure',
        'Cylinder_1-5_fresh_steam',
        'Cylinder_6-15_differential_pressure',
        'Cylinder_6-15_steam_pressure',
        'Cylinder_1-5_steam_temperature',
        'Cylinder_14_differential_pressure',
        'Cylinder_16-24_steam_pressure',
        'Cylinder_25-35_steam_pressure',
        'Cylinder_6-35_differential_pressure',
        
        "ambient_temp_C",
        
        #"Dewatering_top_wire_suction_box_zone_1", # Forming Wire (BAD)
        #"Dewatering_Jet_channel", # Forming Wire (BAD)
        "Dewatering_top_wire_suction_box_zone_2", # Forming Wire
        "Vacuum_suction_box_9", 
        "Vacuum_wet_suction_box", # Forming Wire
        "Vacuum_sheet_seperator_box", # Forming Wire
        "Vacuum_suction_box_10", # Forming Wire
        "Vacuum_suction_box_11", # Forming Wire
        "Vacuum_wire_suction_box_1", # Forming Wire
        "Vacuum_wire_suction_box_2", # Forming Wire
        #"Consistency_white_water", # Forming Wire
        "White_water_temperature", # Forming Wire
        "Conductivity_white_water_B46", # Forming Wire
        "Top_wire_tenstion", # Forming Wire
        "pH_measurement_white_water_B41", # Forming Wire
        
        "Jet/wire_ratio", # Headbox
        "Lip_settings", # Headbox
        
        #"Thick_Stock_Consistency__%_", # Approach Flow
        
        #"Headbox_consistency",
        
        
        
        "Top_Felt_Tension",
        "Bottom_wire_tension",
        "Bottom_Felt_Tension",
        
        "PD4_fabric_tension",
        "PD1_Fabric_tension",
        "PD2_Fabric_tension",
        "PD5_fabric_tension_top",
        "PD5_fabric_tension_bottom",
       
        "pH-Messung_Verdünnungswasser__2..12_pH_",
        "Dissolved_gas_after_stock_deculator_measurement_1",
        "Dissolved_gas_before_dilution_water_deculator",        
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
             
        "Steam_temperature_for_PM",
        "Steam_pressure_for_PM",
                
        'Dewatering_Shoe_press',        
        'Total_Dewatering_Press',        
        'Uhle_box_1_flow___l/min_',
        
        "Stock_deculator_temperature",
        'Stock_deculator_pressure',
        
        'Inlet_Air_2_Temperature',
        'Inlet_Air_1_Temperature',
    
        'DG1_temperature_Inlet_Air',
        'DG2_temperature_Inlet_Air',
        'DG3_temperature_Inlet_Air',
        
        'DG1_Moisture_content_Outlet_Air',
        'DG2_Moisture_content_Outlet_Air',
        'DG3_Moisture_content_Outlet_Air',
       
        'DG1_Ventilator_Revolution_Output',
        'DG2_Ventilator_Revolution_Output',
        'DG3_Ventilator_Revolution_Output',
        
        "fibre_short/long",
        "retention",
             
        "Web_tension_AD6",
        "Moisture_out_of_PreDryer",
        'Cylinder_36_steam_pressure',
        'Cylinder_36_differential_pressure',
        'Cylinder_37_steam_pressure',
        'Cylinder_38_differential_pressure',
        'Cylinder_37_differential_pressure',
        'Cylinder_38_steam_pressure',
        'Cylinder_39_steam_pressure',
        'Cylinder_36-39_steam_pressure',
        'Cylinder_39_differential_pressure',
        'Cylinder_40-53_differential_pressure',
        'Cylinder_40-53_steam_pressure',
        'DG4_Temperature_Inlet_Air',
        'DG5_Temperature_Inlet_Air',
        'DG4_Moisture_content_Outlet_Air',
        'DG5_Moisture_content_Outlet_Air',
        "AD7_fabric_tension_bottom",
        "AD6_fabric_tension",


        'CO2_mass_flow__g/T_',
        "Act_Deaerator_mass_flow__g/T_",
        "Retention_Aid_mass_flow__g/T_", # Approach Flow
        "Bentonite_1_mass_flow__g/T_", # Approach Flow
        "Bentonite_2_mass_flow__g/T_", # Approach Flow
        "Natriumhydroxide_mass_flow__g/T_",
        'Fixative_2_mass_flow__g/T_',
    ],
}

CONTROL_VARS["MBS_SCT_MD"] = CONTROL_VARS["MBS_SCT_CD"]
CONTROL_VARS["MBS_Burst"]= CONTROL_VARS["MBS_SCT_CD"]
CONTROL_VARS["MBS_CMT30"]= CONTROL_VARS["MBS_SCT_CD"]
CONTROL_VARS["Steam_power_corrected"]= CONTROL_VARS["Steam_power"]

# =============================================================================
# Variable group definitions (for PLS dimensionality reduction)
# =============================================================================

SPEED_VARS = [
    "AD6_speed", "AD7_speed_bottom", "Speed", "Forming_Wire_Speed",
    "Speed_Size_Press", "Speed_PD4_bottom", "Speed_PD5_bottom",
    "AD7_speed_top", "Speed_PD1", "Speed_PD3", "Speed_PD2",
    "Speed_PD4_top", "Speed_PD5_top", "Speed_press_section",
]

DRAW_VARS = [
    "Draw_PD5-SS", "Draw_AD7-PR", "Draw_AD6-AD7", "Draw_PD4-PD5",
    "Draw_WS-PS", "Draw_PD1-PD2", "Draw_PD3-PD4", "Draw_SS-AD6",
    "Draw_PS-PD1", "Draw_PD2-PD3",
]

SPEEDSIZER_LINEPRESSURE_VARS = [
    "SpeedSizer_Linepressure_DS",
    "SpeedSizer_Linepressure_FS",
]

LINEPRESSURE_VARS = [
    "Linepressure_1st_press_FS__bar_",
    "Linepressure_2nd_press_FS__bar_",
    "Linepressure_1st_press_DS__bar_",
    "Linepressure_2nd_press_DS__bar_",
    "Linepressure_shoe_press__bar_",
]

CONC_STARCH_VARS = [
    "Starch_uptake__g/m2_",
    "concentration_starch_working_tank_2",
    "concentration_starch_working_tank_1",
    "Viscosity_for_working_tank_bottom_roll",
]

INLET_TEMP_VARS = [
    "DG1_temperature_Inlet_Air", "DG2_temperature_Inlet_Air",
    "DG3_temperature_Inlet_Air", "DG4_Temperature_Inlet_Air",
    "DG5_Temperature_Inlet_Air",
]

VACUUM_VARS = [
    "Vacuum_wire_suction_box_1", "Vacuum_wire_suction_box_2",
    "Vacuum_Zone_1_PickUp", "Vacuum_presszone_of_suction-press_roll",
    "Vacuum_holding/pre_positions_of_suction-press_roll",
    "Vacuum_uhle-box_Pick-Up", "Vacuum_uhle-box_bottom_felt",
]

EXHAUST_MOISTURE_VARS = [
    "DG1_Moisture_content_Outlet_Air", "DG2_Moisture_content_Outlet_Air",
    "DG4_Moisture_content_Outlet_Air", "DG5_Moisture_content_Outlet_Air",
]

GAS_DECULATOR_VARS = [
    "Free_gas_before_dilution_water_deculator_measurement_2",
    "Free_gas_before_dilution_water_deculator_measurement_1~^0",
    "Dissolved_gas_after_stock_deculator_measurement_2",
    "Dissolved_gas_after_stock_deculator_measurement_1",
    "Dissolved_gas_before_dilution_water_deculator",
    "Free_gas_after_stock_deculator~^0",
]

FABRIC_TENSION_VARS = [
    "AD6_fabric_tension", "PD1_Fabric_tension", "PD2_Fabric_tension",
    "PD4_fabric_tension", "PD5_fabric_tension_top", "PD5_fabric_tension_bottom",
]

STARCH_TOP_VARS = [
    "inv_Rod_pressure_Top_Roll", "concentration_starch_working_tank_2",
    "Viscosity_for_working_tank_bottom_roll", "MBS_SCT_CD",
]

STARCH_BOTTOM_VARS = [
    "inv_Rod_Pressure_Bottom_Roll", "concentration_starch_working_tank_1",
    "Viscosity_for_working_tank_bottom_roll", "MBS_SCT_CD",
]

STEAM_PRESSURE_VARS = [
    "Cylinder_36_steam_pressure", "Cylinder_37_steam_pressure",
    "Cylinder_38_steam_pressure", "Cylinder_39_steam_pressure",
    "Cylinder_36-39_steam_pressure", "Cylinder_40-53_steam_pressure",
    "Cylinder_1_steam_pressure", "Cylinder_2_steam_pressure",
    "Cylinder_3_steam_pressure", "Cylinder_5_steam_pressure",
    "Cylinder_4_steam_pressure", "Cylinder_1-5_steam_pressure",
    "Cylinder_6-15_steam_pressure", "Cylinder_16-24_steam_pressure",
    "Cylinder_25-35_steam_pressure",
]

STEAM_DIFF_PRESSURE_VARS = [
    "Cylinder_36_differential_pressure", "Cylinder_38_differential_pressure",
    "Cylinder_37_differential_pressure", "Cylinder_39_differential_pressure",
    "Cylinder_40-53_differential_pressure", "Cylinder_1_differential_pressure",
    "Cylinder_3_differential_pressure", "Cylinder_2_differential_pressure",
    "Cylinder_4_differential_pressure", "Cylinder_5_differential_pressure",
    "Cylinder_6-15_differential_pressure", "Cylinder_14_differential_pressure",
    "Cylinder_6-35_differential_pressure",
]
