"""Fixed-variable subset selection, v1. No baseline comparison.
Dependencies: numpy, pandas, scipy, scikit-learn, pyarrow and your project
utility.py, preprocessing.py and their imported project modules.
Use run_analysis(...) from the companion notebook; importing does not run it.
"""
PIPELINE_PREFIXES = {
    "Steam_power": ["steam_pressure", "exha_mois", "inlet_temp", "vacuum", "linepressure", "fabric_tension", "gas_decu"],
    "Steam__kWh/T_": ["exha_mois", "inlet_temp", "linepressure", "fabric_tension", "gas_decu"],
    "Electrical_power_MW": ["speedsizer_linepressure", "linepressure"],
    "Electricity__kWh/T_": ["speedsizer_linepressure", "linepressure"],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": ["speedsizer_linepressure", "linepressure", "starch2"],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": ["speedsizer_linepressure", "linepressure", "starch1"],
    #**{v: ["speedsizer_linepressure", "conc_starch"] for v in ["MBS_SCT_CD", "MBS_SCT_MD", "MBS_Burst", "MBS_CMT30"]},
    **{v: ["speedsizer_linepressure"] for v in ["MBS_SCT_CD", "MBS_SCT_MD", "MBS_Burst", "MBS_CMT30"]},
}
PIPELINE_PREFIXES["Steam_power_corrected"] = list(PIPELINE_PREFIXES["Steam_power"])
AVAILABLE_VARS = ['Stock_pump_power', 'Speed', 'Current_reel_weight', 'Current_reel_width', 'Current_reel_length', 'Production_Rate__T/h_', 'Current_reel_moisture_average(reel)', 'Actual_moisture', 'Contact_pressure_reel_holders', 'Reel_discharge_pressure', 'Draw_meaurement_pope_reel_FS', 'Draw_measurement_pope_reel_DS', 'Total_draw_measurement_pope_reel', 'Contact_pressure_secondary_arm_OS', 'Contact_pressure_secondary_arm_DS', 'Steam_energy_from_power_plant_to_paper_plant', 'Condensate_energy_from_paper_plant_to_power_plant', 'Speed_Size_Press', 'DG4_Temperature_Inlet_Air', 'Cylinder_36_steam_pressure', 'Cylinder_36_differential_pressure', 'DG5_Temperature_Inlet_Air', 'Cylinder_37_steam_pressure', 'Cylinder_38_differential_pressure', 'Cylinder_37_differential_pressure', 'Cylinder_38_steam_pressure', 'Cylinder_39_steam_pressure', 'Cylinder_36-39_steam_pressure', 'Cylinder_39_differential_pressure', 'Cylinder_40-53_differential_pressure', 'Cylinder_40-53_steam_pressure', 'Web_tension_AD6', 'Steam_flow_to_AfterDryers', 'DG4_Moisture_content_Outlet_Air', 'DG5_Moisture_content_Outlet_Air', 'DG_4-5_zero_level_(Pa)', 'DG5_Ventilator_Revolution_Output', 'DG4_Ventilator_Revolution_Output', 'AD7_fabric_tension_bottom', 'AD6_fabric_tension', 'SpeedSizer_Linepressure_DS', 'SpeedSizer_Linepressure_FS', 'Pressure_of_Starch_flow_Speedsizer_Bottom_Roll~^0', 'Starch_consumption_Bottom___m³/h_', 'Starch_consumption_Top__m³/h_', 'Consistency_starch_main_line', 'Pressure_of_Starch_flow_Speedsizer_Top_Roll', 'Dilution_water_working_tank_2', 'Flow_starch_main_line_to_working_tank_2~^0', 'Dilution_water_working_tank_1', 'Flow_starch_main_line_to_working_tank_1~^0', 'Starch_Top_Roll__ml/m²_', 'Starch_Bottom_Roll__ml/m²_', 'Moisture_after_SpeedSizer', 'Current_reel_moisture_average(SpeedSizer)', 'AirTurn_Temperature', 'Airturn_pillow_pressure', 'Rod_Pressure_Bottom_Roll', 'Rod_clamping_pressure_Bottom_Roll', 'Air_pressure_of_rod_clamping_hose_Bottom_Roll', 'Rod_clamping_pressure_Top_Roll', 'Rod_pressure_Top_Roll', 'Air_pressure_of_rod_clamping_hose_Top_Roll', 'Steam_flow_to_starch_kitchen', 'Enzyme_flow_Slurry_1', 'Slurry_1_dosing_screw_rotation', 'Slurry_1_level', 'Dilution_water_flow_slurry_1', 'Flow_slurry_1_to_reactor', 'Enzyme_flow_Slurry_2', 'Slurry_2_dosing_screw_rotation', 'Slurry_2_level', 'Dilution_water_flow_slurry_2', 'Flow_slurry_2_to_reactor', 'Enzyme_flow_Slurry_3', 'Slurry_3_dosing_screw_rotation', 'Slurry_3_level', 'Dilution_water_flow_slurry_3', 'Flow_slurry_3_to_reactor', 'Pressure_slurry_pipe_3', 'Pressure_slurry_main_pipe', 'Reactor_steam_pressure', 'Reactor_Temperature', 'Reactor_level', 'Starch_flow_to_inactivation', 'Pressure_to_inactivation', 'Temperature_inactivation', 'Pressure_after_inactivation', 'Storage_tank_level', 'Storage_tank_temperature', 'Pressure_starch_main_line', 'Temperature_starch_working_tank_1', 'Level_storage_tank_2~^0', 'Differential_pressure_filter_storage_tank_1', 'Temperature_starch_working_tank_2', 'Level_storage_tank_2', 'Differential_pressure_filter_storage_tank_2', 'Steam_Pressure_Inactivation', 'Dilution_water_flow_slurry_2_control_setting', 'Pressure_cold_water_for_slurry_mixing', 'Dilution_water_flow_slurry_1_control_setting', 'Revolution_of_Slurry_2_to_reactor_(control_setting)', 'Revolution_of_Slurry_3_to_reactor_(control_setting)', 'Revolution_of_Slurry_1_to_reactor_(control_setting)', 'Pressure_in_cyclon_storage_tank', 'Fresh_water_main_pipe_pressure', 'Speed_press_section', 'Cylinder_1_differential_pressure', 'Cylinder_1_steam_pressure', 'Cylinder_2_steam_pressure', 'Cylinder_3_differential_pressure', 'Cylinder_2_differential_pressure', 'Cylinder_3_steam_pressure', 'Cylinder_4_differential_pressure', 'Cylinder_5_steam_pressure', 'Cylinder_4_steam_pressure', 'Cylinder_5_differential_pressure', 'Cylinder_1-5_steam_pressure', 'Cylinder_1-5_fresh_steam', 'Cylinder_6-15_steam_pressure', 'Cylinder_1-5_steam_temperature', 'Cylinder_14_differential_pressure', 'Cylinder_16-24_steam_pressure', 'Cylinder_25-35_steam_pressure', 'Moisture_out_of_PreDryer', 'Cylinder_6-35_differential_pressure', 'Inlet_Air_2_Temperature', 'Inlet_Air_1_Temperature', 'DG_1-3_zero_point', 'DG1_Ventilator_Revolution_Output', 'DG1_Moisture_content_Outlet_Air', 'DG1_temperature_Inlet_Air', 'DG2_Moisture_content_Outlet_Air', 'DG2_temperature_Inlet_Air', 'DG2_Ventilator_Revolution_Output', 'DG3_temperature_Inlet_Air', 'DG3_Ventilator_Revolution_Output', 'PD1_Fabric_tension', 'PD2_Fabric_tension', 'PD5_fabric_tension_bottom', 'PD4_fabric_tension', 'PD4_fabric_tension_bottom', 'PD5_fabric_tension_top', 'ProRelease_1_LoVac', 'ProRelease_2_LoVac', 'ProRelease_3_LoVac', 'ProRelease_4_LoVac', 'ProRelease_5_LoVac', 'ProRelease_6_LoVac', 'ProRelease_7_LoVac', 'Compressor_1_outgoing_pressure', 'Compressor_2_outgoing_pressure', 'Forming_Wire_Speed', 'PickUp_Tension', 'Vacuum_Zone_1_PickUp', 'Vacuum_presszone_of_suction-press_roll', 'Vacuum_holding/pre_positions_of_suction-press_roll', 'Vacuum_uhle-box_Pick-Up', 'Vacuum_uhle-box_bottom_felt', 'Power_Vacuum_pump_press_section_305', 'Power_Vacuum_pump_308_(reserve)', 'Power_Vacuum_pump_press_section_306', 'Dewatering_First_Press_Roll', 'Dewatering_Suction_Press_Roll', 'Dewatering_Shoe_press', 'Dewatering_Pick-Up', 'Linepressure_1st_press_FS__bar_', 'Linepressure_2nd_press_FS__bar_', 'Linepressure_1st_press_DS__bar_', 'Linepressure_2nd_press_DS__bar_', 'Linepressure_shoe_press__bar_', 'Uhle_box_1_flow___l/min_', 'Uhle_box_2_flow___l/min_', 'Top_Felt_Tension', 'Bottom_Felt_Tension', 'Total_Dewatering_Press', 'Bottom_wire_tension', 'Vacuum_top_wire_suction_box_zone_1', 'Vacuum_top_wire_suction_box_zone_2', 'Vacuum_formning_roll', 'Dewatering_top_wire_suction_box_zone_1', 'Dewatering_Jet_channel', 'Dewatering_top_wire_suction_box_zone_2', 'Vacuum_suction_box_9', 'Vacuum_wet_suction_box', 'Vacuum_sheet_seperator_box', 'Vacuum_suction_box_10', 'Vacuum_suction_box_11', 'Vacuum_wire_suction_box_1', 'Vacuum_wire_suction_box_2', 'Consistency_white_water', 'White_water_temperature', 'Conductivity_white_water_B46', 'Top_wire_tenstion', 'pH_measurement_white_water_B41', 'Differential_pressure_retention_aid_filter', 'Inlet_pressure_TrumpJet_station_A1-A4', 'Differential_pressure_A1-A4_between_stock_and_chemical', 'Inlet_pressure_TrumpJet_station_C1-C4', 'Differential_pressure_C1-C4_between_stock_and_chemical', 'Pre_pressure_TrumpJet', 'Stock_to_TrumpJet', 'Mixing_water_to_TrumpJet', 'Pressure_of_pressure_amplifying_pump_for_retention_aid_injection', 'Flow_of_pressure_amplyfing_pump', 'Bentonite_filter_differential_pressure', 'Pressure_Bentonite_TrumpJet', 'Differential_pressure_B1-B4_between_stock_and_chemical', 'Backflow_cross-flow_distributor_dilution_water~^0', 'Backflow_cross-flow_distributor_stock~^0', 'Headbox_pressure', 'Headbox_pressure_DS', 'Headbox_pressure_FS', 'Headbox_consistency', 'Jet/wire_ratio', 'Thick_Stock_Consistency__%_', 'Thick_Stock_Flow__l/min_', 'Machine_chest_consistency', 'Dilution_water_deculator_pressure', 'Sorter_stock_power', 'Dilution_water_deculator_temperature', 'Stock_deculator_pressure', 'Stock_deculator_temperature', 'Dilutionwater_pump_power', 'Short_fibre_flow', 'Long_fibre_flow', 'Short_fibre_B06_consistency', 'Long_fibre_consistency_B07', 'Dry_broke_flow', 'pH-Messung_Verdünnungswasser__2..12_pH_', 'Wet_broke_flow', 'Wet_broke_consistency', 'Sludge_addition_to_stock', 'Free_gas_before_dilution_water_deculator_measurement_1~^0', 'Free_gas_before_dilution_water_deculator_measurement_2', 'Dissolved_gas_before_dilution_water_deculator', 'Free_gas_after_stock_deculator~^0', 'Dissolved_gas_after_stock_deculator_measurement_1', 'Dissolved_gas_after_stock_deculator_measurement_2', 'Stock_Valve_Opening_From_Machine_Chest', 'Dry_Broke_Consistancy__%_', 'Pulper_consistency', 'ATS1_power', 'ATS1_differential_pressure', 'ATS1_light_reject_flow', 'ATS2_differential_pressure', 'ATS2_light_reject_flow', 'Combisorter_1_power', 'Combisorter_2_power', 'Combisorter_3_power', 'Contaminex_1_power', 'Contaminex_2_power', 'Contaminex_3_power', 'Multifractor_1_long_fibre_flow', 'Multifractor_1_short_fibre_flow', 'Multifractor_1_consistency', 'Multifractor_2_long_fibre_flow', 'Multifractor_2_short_fibre_flow', 'Multifractor_2_consistency', 'Multifractor_3_long_fibre_flow', 'Multifractor_3_short_fibre_flow', 'Multifractor_3_consistency', 'LF_screen_1_power', 'LF_screen_1_accept_flow', 'LF_screen_2_inlet_consistency', 'LF_screen_2_power', 'LF_screen_3_inlet_consistency', 'LF_screen_3_power', 'LF_screen_1_reject_flow', 'LF_screen_2_accept_flow', 'LF_screen_2_reject_flow', 'LF_screen_3_reject_flow', 'LF_screen_3_accept_flow', 'Steam_flow_to_heat_exchangers', 'Steam_flow_to_hall_heating', 'Steam_flow_to_white_water_heating', 'Steam_flow_to_steam_box', 'Steam_flow_to_PM', 'Pressure_main_steam_line', 'Steam_temperature_for_PM', 'Steam_pressure_for_PM', 'Condensate_conductivity', 'Total_condensate_flow', 'Freshwater_warm__l/min_', 'Freshwater_retention__l/min_', 'Freshwater_to_Machine_Tank__l/min_', 'Freshwater_pressure_from_the_city__bar_', 'Waste_steam_flow', 'Temperature_working_tank_to_bottom_roll', 'Concentration_of_starch_to_bottom_roll', 'Viscosity_for_working_tank_bottom_roll', 'Viscosity_for_starch_ring_main_line', 'Draw_AD7-PR', 'Draw_SS-AD6', 'Draw_AD6-AD7', 'Draw_PD5-SS', 'Draw_PS-PD1', 'Draw_PD2-PD3', 'Draw_PD4-PD5', 'Draw_PD3-PD4', 'Draw_PD1-PD2', 'Draw_WS-PS', 'Produzierte_Quadratmeter', 'Auftrag_Stärkeslurry_FW', 'Auftrag_Stärkeslurry_BW', 'Starch_application_FW_in_ml', 'Starch_application_BW_in_ml', 'Slurry_1_pumping_to_reactor', 'Slurry_2_pumping_to_reactor', 'Slurry_3_pumping_to_reactor', 'Total_Dewatering_Pick-Up', 'Total_Dewatering_Bottom_Felt', 'Multifractor_1_Long_fibre_fraction', 'Multifractor_2_long_fibre_fraction', 'Multifractor_3_long_fibre_fraction', 'Steam_flow_from_power_plant_to_PM', 'Steam_flow_to_PreDryers', 'Retention_Aid_mass_flow__g/T_', 'Bentonite_1_mass_flow__g/T_', 'Bentonite_2_mass_flow__g/T_', 'Fixiermittelmenge_check', 'Fixative_2_mass_flow__g/T_', 'Act_Deaerator_mass_flow__g/T_', 'Paper_break_status_check', 'Starch_mass_flow__kg/T_', 'Defoamer_mass_flow__g/T_', 'Leimungsmenge_gesamt_pro_Tonne', 'Leimungsmenge_gesamt_pro_Tonne_check', 'Sizing_Agent__g/T_', 'Dry_Strength_Agent_mass_flow__kg/T_', 'Natriumhydroxide_mass_flow__g/T_', 'CO2_mass_flow__g/T_', 'Summe_Energieverbrauch_APA_in_kW', 'Electricity__kWh/T_', 'Steam__kWh/T_', 'MC_SF_LF_Demand', 'Whitewater_solids_flow', 'Approach_flow_returns', 'Fibre_usage__T/T_', 'Consistency_main_pipe_%', 'Concentration_of_starch_to_top_roll', 'concentration_starch_working_tank_1', 'concentration_starch_working_tank_2', 'ambient_temp_C', 'mismatch_run_pos', 'Steam_power', 'Steam_power_corrected', 'Electrical_power_MW', 'Steam_Predryers__kWh/T_', 'Steam_Afterdryers__kWh/T_', 'Steam_Starchkitchen__kWh/T_', 'Steam_Heatexchangers__kWh/T_', 'Steam_Whitewater__kWh/T_', 'Dry_Strength_Agent_mass_flow__g/T_', 'Retention_Aid__g/T_', 'Bentonite_1__g/T_', 'Bentonite_2__g/T_', 'Fixative_2__g/T_', 'Act_Deaerator__g/T_', 'Defoamer__g/T_', 'Dry_Strength_Agent__g/T_', 'Natriumhydroxide__g/T_', 'Current_basis_weight']
# Explicit selected candidates; no exclusion rules are applied.
DEFAULT_CONSIDERED_VARS = ['Stock_pump_power', 'Speed', 'Current_reel_weight', 'Current_reel_width', 'Current_reel_length', 'Production_Rate__T/h_', 'Current_reel_moisture_average(reel)', 'Actual_moisture', 'Contact_pressure_reel_holders', 'Reel_discharge_pressure', 'Draw_meaurement_pope_reel_FS', 'Draw_measurement_pope_reel_DS', 'Total_draw_measurement_pope_reel', 'Contact_pressure_secondary_arm_OS', 'Contact_pressure_secondary_arm_DS', 'Steam_energy_from_power_plant_to_paper_plant', 'Condensate_energy_from_paper_plant_to_power_plant', 'Speed_Size_Press', 'DG4_Temperature_Inlet_Air', 'Cylinder_36_steam_pressure', 'Cylinder_36_differential_pressure', 'DG5_Temperature_Inlet_Air', 'Cylinder_37_steam_pressure', 'Cylinder_38_differential_pressure', 'Cylinder_37_differential_pressure', 'Cylinder_38_steam_pressure', 'Cylinder_39_steam_pressure', 'Cylinder_36-39_steam_pressure', 'Cylinder_39_differential_pressure', 'Cylinder_40-53_differential_pressure', 'Cylinder_40-53_steam_pressure', 'Web_tension_AD6', 'Steam_flow_to_AfterDryers', 'DG4_Moisture_content_Outlet_Air', 'DG5_Moisture_content_Outlet_Air', 'DG_4-5_zero_level_(Pa)', 'DG5_Ventilator_Revolution_Output', 'DG4_Ventilator_Revolution_Output', 'AD7_fabric_tension_bottom', 'AD6_fabric_tension', 'SpeedSizer_Linepressure_DS', 'SpeedSizer_Linepressure_FS', 'Pressure_of_Starch_flow_Speedsizer_Bottom_Roll~^0', 'Starch_consumption_Bottom___m³/h_', 'Starch_consumption_Top__m³/h_', 'Consistency_starch_main_line', 'Pressure_of_Starch_flow_Speedsizer_Top_Roll', 'Dilution_water_working_tank_2', 'Flow_starch_main_line_to_working_tank_2~^0', 'Dilution_water_working_tank_1', 'Flow_starch_main_line_to_working_tank_1~^0', 'Starch_Top_Roll__ml/m²_', 'Starch_Bottom_Roll__ml/m²_', 'Moisture_after_SpeedSizer', 'Current_reel_moisture_average(SpeedSizer)', 'AirTurn_Temperature', 'Airturn_pillow_pressure', 'Rod_Pressure_Bottom_Roll', 'Rod_clamping_pressure_Bottom_Roll', 'Air_pressure_of_rod_clamping_hose_Bottom_Roll', 'Rod_clamping_pressure_Top_Roll', 'Rod_pressure_Top_Roll', 'Air_pressure_of_rod_clamping_hose_Top_Roll', 'Steam_flow_to_starch_kitchen', 'Enzyme_flow_Slurry_1', 'Slurry_1_dosing_screw_rotation', 'Slurry_1_level', 'Dilution_water_flow_slurry_1', 'Flow_slurry_1_to_reactor', 'Enzyme_flow_Slurry_2', 'Slurry_2_dosing_screw_rotation', 'Slurry_2_level', 'Dilution_water_flow_slurry_2', 'Flow_slurry_2_to_reactor', 'Enzyme_flow_Slurry_3', 'Slurry_3_dosing_screw_rotation', 'Slurry_3_level', 'Dilution_water_flow_slurry_3', 'Flow_slurry_3_to_reactor', 'Pressure_slurry_pipe_3', 'Pressure_slurry_main_pipe', 'Reactor_steam_pressure', 'Reactor_Temperature', 'Reactor_level', 'Starch_flow_to_inactivation', 'Pressure_to_inactivation', 'Temperature_inactivation', 'Pressure_after_inactivation', 'Storage_tank_level', 'Storage_tank_temperature', 'Pressure_starch_main_line', 'Temperature_starch_working_tank_1', 'Level_storage_tank_2~^0', 'Differential_pressure_filter_storage_tank_1', 'Temperature_starch_working_tank_2', 'Level_storage_tank_2', 'Differential_pressure_filter_storage_tank_2', 'Steam_Pressure_Inactivation', 'Dilution_water_flow_slurry_2_control_setting', 'Pressure_cold_water_for_slurry_mixing', 'Dilution_water_flow_slurry_1_control_setting', 'Revolution_of_Slurry_2_to_reactor_(control_setting)', 'Revolution_of_Slurry_3_to_reactor_(control_setting)', 'Revolution_of_Slurry_1_to_reactor_(control_setting)', 'Pressure_in_cyclon_storage_tank', 'Fresh_water_main_pipe_pressure', 'Speed_press_section', 'Cylinder_1_differential_pressure', 'Cylinder_1_steam_pressure', 'Cylinder_2_steam_pressure', 'Cylinder_3_differential_pressure', 'Cylinder_2_differential_pressure', 'Cylinder_3_steam_pressure', 'Cylinder_4_differential_pressure', 'Cylinder_5_steam_pressure', 'Cylinder_4_steam_pressure', 'Cylinder_5_differential_pressure', 'Cylinder_1-5_steam_pressure', 'Cylinder_1-5_fresh_steam', 'Cylinder_6-15_steam_pressure', 'Cylinder_1-5_steam_temperature', 'Cylinder_14_differential_pressure', 'Cylinder_16-24_steam_pressure', 'Cylinder_25-35_steam_pressure', 'Moisture_out_of_PreDryer', 'Cylinder_6-35_differential_pressure', 'Inlet_Air_2_Temperature', 'Inlet_Air_1_Temperature', 'DG_1-3_zero_point', 'DG1_Ventilator_Revolution_Output', 'DG1_Moisture_content_Outlet_Air', 'DG1_temperature_Inlet_Air', 'DG2_Moisture_content_Outlet_Air', 'DG2_temperature_Inlet_Air', 'DG2_Ventilator_Revolution_Output', 'DG3_temperature_Inlet_Air', 'DG3_Ventilator_Revolution_Output', 'PD1_Fabric_tension', 'PD2_Fabric_tension', 'PD5_fabric_tension_bottom', 'PD4_fabric_tension', 'PD4_fabric_tension_bottom', 'PD5_fabric_tension_top', 'ProRelease_1_LoVac', 'ProRelease_2_LoVac', 'ProRelease_3_LoVac', 'ProRelease_4_LoVac', 'ProRelease_5_LoVac', 'ProRelease_6_LoVac', 'ProRelease_7_LoVac', 'Compressor_1_outgoing_pressure', 'Compressor_2_outgoing_pressure', 'Forming_Wire_Speed', 'PickUp_Tension', 'Vacuum_Zone_1_PickUp', 'Vacuum_presszone_of_suction-press_roll', 'Vacuum_holding/pre_positions_of_suction-press_roll', 'Vacuum_uhle-box_Pick-Up', 'Vacuum_uhle-box_bottom_felt', 'Power_Vacuum_pump_press_section_305', 'Power_Vacuum_pump_308_(reserve)', 'Power_Vacuum_pump_press_section_306', 'Dewatering_First_Press_Roll', 'Dewatering_Suction_Press_Roll', 'Dewatering_Shoe_press', 'Dewatering_Pick-Up', 'Linepressure_1st_press_FS__bar_', 'Linepressure_2nd_press_FS__bar_', 'Linepressure_1st_press_DS__bar_', 'Linepressure_2nd_press_DS__bar_', 'Linepressure_shoe_press__bar_', 'Uhle_box_1_flow___l/min_', 'Uhle_box_2_flow___l/min_', 'Top_Felt_Tension', 'Bottom_Felt_Tension', 'Total_Dewatering_Press', 'Bottom_wire_tension', 'Vacuum_top_wire_suction_box_zone_1', 'Vacuum_top_wire_suction_box_zone_2', 'Vacuum_formning_roll', 'Dewatering_top_wire_suction_box_zone_1', 'Dewatering_Jet_channel', 'Dewatering_top_wire_suction_box_zone_2', 'Vacuum_suction_box_9', 'Vacuum_wet_suction_box', 'Vacuum_sheet_seperator_box', 'Vacuum_suction_box_10', 'Vacuum_suction_box_11', 'Vacuum_wire_suction_box_1', 'Vacuum_wire_suction_box_2', 'Consistency_white_water', 'White_water_temperature', 'Conductivity_white_water_B46', 'Top_wire_tenstion', 'pH_measurement_white_water_B41', 'Differential_pressure_retention_aid_filter', 'Inlet_pressure_TrumpJet_station_A1-A4', 'Differential_pressure_A1-A4_between_stock_and_chemical', 'Inlet_pressure_TrumpJet_station_C1-C4', 'Differential_pressure_C1-C4_between_stock_and_chemical', 'Pre_pressure_TrumpJet', 'Stock_to_TrumpJet', 'Mixing_water_to_TrumpJet', 'Pressure_of_pressure_amplifying_pump_for_retention_aid_injection', 'Flow_of_pressure_amplyfing_pump', 'Bentonite_filter_differential_pressure', 'Pressure_Bentonite_TrumpJet', 'Differential_pressure_B1-B4_between_stock_and_chemical', 'Backflow_cross-flow_distributor_dilution_water~^0', 'Backflow_cross-flow_distributor_stock~^0', 'Headbox_pressure', 'Headbox_pressure_DS', 'Headbox_pressure_FS', 'Headbox_consistency', 'Jet/wire_ratio', 'Thick_Stock_Consistency__%_', 'Thick_Stock_Flow__l/min_', 'Machine_chest_consistency', 'Dilution_water_deculator_pressure', 'Sorter_stock_power', 'Dilution_water_deculator_temperature', 'Stock_deculator_pressure', 'Stock_deculator_temperature', 'Dilutionwater_pump_power', 'Short_fibre_flow', 'Long_fibre_flow', 'Short_fibre_B06_consistency', 'Long_fibre_consistency_B07', 'Dry_broke_flow', 'pH-Messung_Verdünnungswasser__2..12_pH_', 'Wet_broke_flow', 'Wet_broke_consistency', 'Sludge_addition_to_stock', 'Free_gas_before_dilution_water_deculator_measurement_1~^0', 'Free_gas_before_dilution_water_deculator_measurement_2', 'Dissolved_gas_before_dilution_water_deculator', 'Free_gas_after_stock_deculator~^0', 'Dissolved_gas_after_stock_deculator_measurement_1', 'Dissolved_gas_after_stock_deculator_measurement_2', 'Stock_Valve_Opening_From_Machine_Chest', 'Dry_Broke_Consistancy__%_', 'Pulper_consistency', 'ATS1_power', 'ATS1_differential_pressure', 'ATS1_light_reject_flow', 'ATS2_differential_pressure', 'ATS2_light_reject_flow', 'Combisorter_1_power', 'Combisorter_2_power', 'Combisorter_3_power', 'Contaminex_1_power', 'Contaminex_2_power', 'Contaminex_3_power', 'Multifractor_1_long_fibre_flow', 'Multifractor_1_short_fibre_flow', 'Multifractor_1_consistency', 'Multifractor_2_long_fibre_flow', 'Multifractor_2_short_fibre_flow', 'Multifractor_2_consistency', 'Multifractor_3_long_fibre_flow', 'Multifractor_3_short_fibre_flow', 'Multifractor_3_consistency', 'LF_screen_1_power', 'LF_screen_1_accept_flow', 'LF_screen_2_inlet_consistency', 'LF_screen_2_power', 'LF_screen_3_inlet_consistency', 'LF_screen_3_power', 'LF_screen_1_reject_flow', 'LF_screen_2_accept_flow', 'LF_screen_2_reject_flow', 'LF_screen_3_reject_flow', 'LF_screen_3_accept_flow', 'Steam_flow_to_heat_exchangers', 'Steam_flow_to_hall_heating', 'Steam_flow_to_white_water_heating', 'Steam_flow_to_steam_box', 'Steam_flow_to_PM', 'Pressure_main_steam_line', 'Steam_temperature_for_PM', 'Steam_pressure_for_PM', 'Condensate_conductivity', 'Total_condensate_flow', 'Freshwater_warm__l/min_', 'Freshwater_retention__l/min_', 'Freshwater_to_Machine_Tank__l/min_', 'Freshwater_pressure_from_the_city__bar_', 'Waste_steam_flow', 'Temperature_working_tank_to_bottom_roll', 'Concentration_of_starch_to_bottom_roll', 'Viscosity_for_working_tank_bottom_roll', 'Viscosity_for_starch_ring_main_line', 'Draw_AD7-PR', 'Draw_SS-AD6', 'Draw_AD6-AD7', 'Draw_PD5-SS', 'Draw_PS-PD1', 'Draw_PD2-PD3', 'Draw_PD4-PD5', 'Draw_PD3-PD4', 'Draw_PD1-PD2', 'Draw_WS-PS', 'Produzierte_Quadratmeter', 'Auftrag_Stärkeslurry_FW', 'Auftrag_Stärkeslurry_BW', 'Starch_application_FW_in_ml', 'Starch_application_BW_in_ml', 'Slurry_1_pumping_to_reactor', 'Slurry_2_pumping_to_reactor', 'Slurry_3_pumping_to_reactor', 'Total_Dewatering_Pick-Up', 'Total_Dewatering_Bottom_Felt', 'Multifractor_1_Long_fibre_fraction', 'Multifractor_2_long_fibre_fraction', 'Multifractor_3_long_fibre_fraction', 'Steam_flow_from_power_plant_to_PM', 'Steam_flow_to_PreDryers', 'Retention_Aid_mass_flow__g/T_', 'Bentonite_1_mass_flow__g/T_', 'Bentonite_2_mass_flow__g/T_', 'Fixiermittelmenge_check', 'Fixative_2_mass_flow__g/T_', 'Act_Deaerator_mass_flow__g/T_', 'Paper_break_status_check', 'Starch_mass_flow__kg/T_', 'Defoamer_mass_flow__g/T_', 'Leimungsmenge_gesamt_pro_Tonne', 'Leimungsmenge_gesamt_pro_Tonne_check', 'Sizing_Agent__g/T_', 'Dry_Strength_Agent_mass_flow__kg/T_', 'Natriumhydroxide_mass_flow__g/T_', 'CO2_mass_flow__g/T_', 'Summe_Energieverbrauch_APA_in_kW', 'Electricity__kWh/T_', 'Steam__kWh/T_', 'MC_SF_LF_Demand', 'Whitewater_solids_flow', 'Approach_flow_returns', 'Fibre_usage__T/T_', 'Consistency_main_pipe_%', 'Concentration_of_starch_to_top_roll', 'concentration_starch_working_tank_1', 'concentration_starch_working_tank_2', 'ambient_temp_C', 'mismatch_run_pos', 'Steam_power', 'Steam_power_corrected', 'Electrical_power_MW', 'Steam_Predryers__kWh/T_', 'Steam_Afterdryers__kWh/T_', 'Steam_Starchkitchen__kWh/T_', 'Steam_Heatexchangers__kWh/T_', 'Steam_Whitewater__kWh/T_', 'Dry_Strength_Agent_mass_flow__g/T_', 'Retention_Aid__g/T_', 'Bentonite_1__g/T_', 'Bentonite_2__g/T_', 'Fixative_2__g/T_', 'Act_Deaerator__g/T_', 'Defoamer__g/T_', 'Dry_Strength_Agent__g/T_', 'Natriumhydroxide__g/T_', 'delta_basis_weight', 'Starch_uptake__g/m2_', 'Fibre__g/m2_', 'fibre_short/long', 'temperature_starch_working_tank']
DEFAULT_FIXED_VARS = ["Jet/wire_ratio", "Starch_uptake__g/m2_"]

from pathlib import Path
from itertools import combinations
from functools import lru_cache
import sys, re, json, html, warnings
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
def lag_pairs(t_hours, lag_hours, tolerance=0.35):
    """Your nearest-time pairing, with input validation and an empty-input guard."""
    t_hours = np.asarray(t_hours, dtype=float)
    if lag_hours <= 0 or not np.isfinite(lag_hours) or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Lag must be positive and tolerance finite/nonnegative.")
    if not np.isfinite(t_hours).all() or np.any(np.diff(t_hours) < 0):
        raise ValueError("Pairing requires sorted, finite times.")
    if not len(t_hours):
        return np.array([], dtype=int), np.array([], dtype=int)
    target = t_hours - lag_hours
    j = np.clip(np.searchsorted(t_hours, target), 0, len(t_hours) - 1)
    j_alt = np.clip(j - 1, 0, len(t_hours) - 1)
    better = np.abs(t_hours[j_alt] - target) < np.abs(t_hours[j] - target)
    j = np.where(better, j_alt, j)
    realised = t_hours - t_hours[j]
    ok = (j < np.arange(len(t_hours))) & (np.abs(realised - lag_hours) <= tolerance * lag_hours)
    later = np.flatnonzero(ok)
    return later, j[later]


def to_hours(index):
    idx = pd.DatetimeIndex(index)
    if idx.hasnans:
        raise ValueError("Time index contains NaT.")
    if not len(idx):
        return np.array([], dtype=float)
    # Subtract integers before float conversion to retain timestamp precision.
    return (idx.asi8 - idx.asi8[0]).astype(float) / 3.6e12


def ewm_filter(s, halflife="6h"):
    out = s.ewm(halflife=halflife, times=s.index, adjust=True).mean()
    if len(s) >= 2:
        out.iloc[0] = (s.iloc[0] + s.iloc[1]) / 2.0
    return out


def filter_training_period(data, target, start_date=None, end_date="2026-07-01"):
    starts = {"Steam_power": "2026-01-01", "Steam_power_corrected": "2026-01-01", "Steam__kWh/T_": "2026-01-01", "Electrical_power_MW": "2025-11-15", "Starch_uptake_by_paper_Top_Roll__g/m2_": "2026-03-01", "Starch_uptake_by_paper_Bottom_Roll__g/m2_": "2026-02-01", "MBS_SCT_CD": "2025-11-15", "MBS_SCT_MD": "2026-02-01", "MBS_Burst": "2026-02-01", "MBS_CMT30": "2026-02-01"}
    lower = starts.get(target) if start_date is None else start_date
    if lower is not None:
        data = data.loc[data.index > pd.Timestamp(lower)]
    if end_date is not None:
        data = data.loc[data.index < pd.Timestamp(end_date)]
    steam = target in {"Steam_power", "Steam_power_corrected", "Steam__kWh/T_"}
    filtered_targets = set(starts)
    windows = [("2026-01-24 07:00", "2026-01-26 10:00")]
    if steam:
        windows += [("2026-01-11 12:00", "2026-01-12 11:00"), ("2026-01-17 12:00", "2026-01-19 11:00")]
    if target in filtered_targets:
        for a, b in windows:
            data = data.loc[~((data.index > pd.Timestamp(a)) & (data.index < pd.Timestamp(b)))]
    vacuum_filter = steam or target in {"Electrical_power_MW", "Starch_uptake_by_paper_Top_Roll__g/m2_", "Starch_uptake_by_paper_Bottom_Roll__g/m2_"}
    if vacuum_filter:
        data = data.loc[pd.to_numeric(data["Vacuum_Zone_1_PickUp"], errors="coerce") < -0.5]
    if steam:
        data = data.loc[pd.to_numeric(data["DG4_Temperature_Inlet_Air"], errors="coerce") > 100]
    if target == "MBS_CMT30":
        data = data.loc[data["AB_Grade_ID"].astype(str).isin(["6010120", "6010100"])]
    return data.loc[pd.to_numeric(data["grammage"], errors="coerce").isin([115, 120, 100, 90, 85, 110])].copy()


def prepare_differencing_data(engineered_data, target,
                              lag_days=0.25, halflife="6h", tolerance=0.35,
                              max_gap="12h", same_segment_only=True,
                              start_date=None, end_date="2026-07-01", considered_vars=(), fixed_vars=(), creator_class=None, prep_pip=None,
                              pre_estimator_builder=None):
    data = engineered_data.copy()
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names must be resolved first.")
    if "Wedge_Time" in data:
        data["Wedge_Time"] = pd.to_datetime(data["Wedge_Time"], errors="raise")
        data = data.set_index("Wedge_Time")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError("A valid datetime index or Wedge_Time column is required.")
    if data.index.has_duplicates:
        raise ValueError("Duplicate timestamps: resolve them before pairing.")
    data = filter_training_period(data.sort_index(), target, start_date, end_date)
    if {"Current_basis_weight", "grammage"}.issubset(data.columns):
        data["delta_basis_weight"] = (pd.to_numeric(data["Current_basis_weight"], errors="coerce")
                                    - pd.to_numeric(data["grammage"], errors="coerce"))
    if target not in data:
        raise KeyError(f"Target column is missing: {target}")
    fixed_vars = unique_in_order(fixed_vars)
    explicit_candidates = unique_in_order(list(considered_vars) + fixed_vars)
    selected = unique_in_order([target] + explicit_candidates)
    pls_groups = []
    pls_sources = []
    pls_scores = []
    pre_estimator = None
    if prep_pip is not None and prep_pip.steps:
        if pre_estimator_builder is None or creator_class is None:
            raise ValueError("PLS requires build_pre_estimator and FeatureCreator.")
        for _, step in prep_pip.steps:
            sources = list(step.pls_columns)
            scores = [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
            if target in sources:
                raise ValueError(f"Target {target} cannot be a PLS input.")
            if set(scores) & set(data.columns):
                raise ValueError(f"PLS score names collide with input columns: {scores}")
            pls_groups.append({"prefix": step.score_prefix, "sources": sources, "scores": scores})
            pls_sources.extend(sources)
            pls_scores.extend(scores)
        if len(pls_scores) != len(set(pls_scores)):
            raise ValueError("PLS score names must be unique.")
        # Remove ALL sources after adding ALL scores: neither controls nor passthrough can reintroduce them.
        selected = [target] + [v for v in unique_in_order(selected[1:] + pls_scores)
                               if v not in set(pls_sources)]
    to_create = [v for v in selected if v not in data and v not in pls_scores]
    creator = None
    required = []
    if to_create:
        if creator_class is None:
            raise KeyError(f"Missing requested variables (FeatureCreator required): {to_create}")
        unknown = [v for v in to_create if v not in creator_class._registry()]
        if unknown:
            raise KeyError(f"requested variables missing from data and FeatureCreator registry: {unknown}")
        expanded = creator_class()._expand_with_deps(to_create)
        creator = creator_class(features_to_create=expanded, features_to_keep=selected, errors="raise", copy=True)
        required = creator.features_required()
        missing = [v for v in required if v not in data and v not in expanded]
        if missing:
            raise KeyError(f"Missing raw dependencies for requested variables: {missing}")
    if pls_groups:
        # Registered engineered sources must be created too, even if excluded as direct predictors.
        missing_features = [v for v in unique_in_order(selected + pls_sources)
                            if v not in data and v not in pls_scores]
        unknown = [v for v in missing_features if v not in creator_class._registry()]
        if unknown:
            raise KeyError(f"Missing PLS/pipeline variables: {unknown}")
        created_vars = creator_class()._expand_with_deps(missing_features)
        pre_estimator, feat_list = pre_estimator_builder(
            exog_vars_reduced=unique_in_order(["grammage"] + selected[1:]),
            prep_pip=prep_pip, created_vars=created_vars, apply_ewm=False)
        # Drop the model-only final scaling/grammage encoding; correlation does not need it.
        pre_estimator = pre_estimator[:-1]
        missing = [v for v in feat_list if v not in data]
        if missing:
            raise KeyError(f"Missing raw pipeline inputs: {missing}")
        required = feat_list
    raw_columns = unique_in_order([v for v in selected if v in data] + [v for v in required if v in data])
    numeric = data[raw_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    unusable = numeric.columns[numeric.isna().all()].tolist()
    if set(required) & set(unusable):
        raise ValueError(f"All-missing dependencies for requested variables: {sorted(set(required) & set(unusable))}")
    if target in unusable:
        raise ValueError(f"No numeric observations for {target}.")
    numeric = numeric.drop(columns=unusable)
    print("Dropping nonnumeric/all-missing columns:", unusable)
    raw = numeric.dropna().copy()
    if len(raw) < 5:
        raise ValueError("Fewer than five valid raw rows remain.")
    grammage = pd.to_numeric(data.loc[raw.index, "grammage"], errors="coerce")
    grade_change = grammage.ne(grammage.shift())
    time_gap = raw.index.to_series().diff().gt(pd.Timedelta(max_gap))
    segments = (grade_change | time_gap).cumsum().to_numpy()
    smoothed = raw.copy()
    for col in smoothed:
        smoothed[col] = smoothed[col].groupby(segments).transform(lambda s: ewm_filter(s, halflife)).to_numpy()
    selected = [v for v in selected if v not in unusable]
    if pre_estimator is not None:
        pre_estimator.named_steps["filter"].columns = selected[1:]
        # Preserve the original grammage for pipeline bookkeeping (not an analysis predictor).
        smoothed["grammage"] = grammage
        physical = pre_estimator.fit_transform(smoothed.copy(), smoothed[target])
        absent = [v for v in selected[1:] if v not in physical]
        if absent:
            raise ValueError(f"PLS did not produce all requested outputs: {absent}")
        physical = physical[selected[1:]].apply(pd.to_numeric, errors="coerce")
        physical.insert(0, target, smoothed[target])
        leaked = set(pls_sources) & set(physical.columns)
        if leaked:
            raise AssertionError(f"PLS inputs leaked into discovery: {sorted(leaked)}")
        if not physical.index.equals(smoothed.index):
            raise ValueError("Pre-estimator changed row alignment.")
        print("PLS replacements:")
        for group in pls_groups:
            print(f"  {group['prefix']}: {len(group['sources'])} sources -> {group['scores']}")
        print("CHECK PASSED: no PLS source columns in discovery.")
    elif creator is None:
        physical = smoothed[selected].copy()
    else:
        creator.features_to_keep = selected
        physical = creator.fit_transform(smoothed.copy()).apply(pd.to_numeric, errors="coerce")
        if not physical.index.equals(smoothed.index):
            raise ValueError("FeatureCreator changed row alignment.")
    t_hours = to_hours(physical.index)
    later, earlier = lag_pairs(t_hours, lag_days * 24, tolerance)
    crosses = segments[later] != segments[earlier]
    crossing_count = int(crosses.sum())
    if same_segment_only:
        later, earlier = later[~crosses], earlier[~crosses]
    pair_info = pd.DataFrame({"later_time": physical.index[later], "earlier_time": physical.index[earlier],
                              "realised_lag_hours": t_hours[later] - t_hours[earlier],
                              "same_ewm_segment": segments[later] == segments[earlier],
                              "later_grammage": grammage.iloc[later].to_numpy(),
                              "earlier_grammage": grammage.iloc[earlier].to_numpy()}, index=physical.index[later])
    arr = physical.to_numpy(dtype=float)
    diff = pd.DataFrame(arr[later] - arr[earlier], index=physical.index[later], columns=physical.columns)
    valid = np.isfinite(diff.to_numpy()).all(axis=1)
    diff, pair_info = diff.loc[valid].copy(), pair_info.loc[valid].copy()
    std = diff.std(ddof=0)
    constant = std.index[(~np.isfinite(std)) | (std <= 1e-12)].tolist()
    if constant:
        print("Dropping constant differenced columns:", constant)
        diff = diff.drop(columns=constant)
    if target not in diff or len(diff) < 5:
        raise ValueError("Insufficient differenced target variation or observations.")
    lost_fixed = set(fixed_vars) - set(diff.columns)
    if lost_fixed:
        raise ValueError(f"Fixed variables cannot be used (missing/non-numeric/constant): {sorted(lost_fixed)}")
    selected = physical.columns.tolist()
    diff.attrs["preparation"] = {"dataset": "lagged_differences", "target": target, "lag_hours": lag_days * 24,
        "ewm_halflife": halflife, "pair_tolerance": tolerance, "same_segment_only": same_segment_only,
        "cross_segment_pairs_before_filter": crossing_count, "cross_segment_pairs_retained": int((~pair_info.same_ewm_segment).sum()),
        "raw_rows": len(raw), "pairs": len(diff), "physical_variables_before_constant_removal": selected,
        "constant_columns_removed": constant, "all_missing_columns_removed": unusable,
        "variable_selection": "explicit inclusion list plus fixed vars; PLS sources replaced",
        "analysis_columns": diff.columns.tolist(), "considered_vars_requested": explicit_candidates,
         "engineered_columns_created": to_create,
        "fixed_vars": fixed_vars, "supervised_pls": bool(pls_groups), "pls_groups": pls_groups,
        "pls_fit_target": target if pls_groups else None, "within_grade_centering": False}
    print(f"Target {target}: {len(raw)} raw rows -> {len(diff)} difference pairs; cross-segment pairs: {int((~pair_info.same_ewm_segment).sum())}")
    control_status = pd.DataFrame([
        {"variable": v, "status": "target" if v == target else
         "replaced_by_pls" if v in pls_sources else
         "target_derived_excluded" if v not in selected and v not in unusable else
         "no_numeric_values" if v in unusable else
         "constant_difference" if v in constant else "included"}
        for v in explicit_candidates], columns=["variable", "status"])
    diff.attrs["preparation"]["inclusion_audit"] = control_status.to_dict("records")
    print("INCLUSION LIST audit:")
    print(control_status.to_string(index=False))
    display_columns = ["realised_lag_hours", "same_ewm_segment"]
    print(pair_info[display_columns].describe().to_string())
    return diff, pair_info, raw, smoothed, physical
def unique_in_order(values):
    return list(dict.fromkeys(values))


def run_analysis(data_path, target="MBS_SCT_CD", considered_vars=None,
                 fixed_vars=None, module_dirs=(".",), use_pls=True,
                 pls_prefixes=None, start_date=None, end_date="2026-07-01",
                 lag_days=.25, ewm_halflife="6h", pair_tolerance=.35,
                 max_gap="12h", alpha_conditional=.05, alpha_marginal=.70,
                 max_k=3, max_states=5000, max_tests=500_000,
                 output_dir=None, verbose=True):
    """Prepare data and select predictors; no baseline is fitted.

    Missing requested columns must be creatable by FeatureCreator, otherwise
    preparation fails explicitly. Project imports occur only when called.
    PLS operates on included nonfixed sources only. Fixed features stay explicit.
    """
    for directory in reversed(module_dirs):
        directory = str(Path(directory).resolve())
        if directory not in sys.path:
            sys.path.insert(0, directory)
    from utility import _feature_engineering, setpoint_df, FeatureCreator
    from preprocessing import make_prep_pip, build_pre_estimator
    from copy import deepcopy

    considered = unique_in_order(DEFAULT_CONSIDERED_VARS if considered_vars is None else considered_vars)
    fixed = unique_in_order(DEFAULT_FIXED_VARS if fixed_vars is None else fixed_vars)
    if target in fixed:
        raise ValueError("The target cannot be a fixed predictor.")
    considered = unique_in_order([v for v in considered if v != target] + fixed)
    engineered = _feature_engineering(pd.read_parquet(data_path), setpoint_df,
                                     steam_null=False, clip=True)
    prefixes = PIPELINE_PREFIXES.get(target, []) if pls_prefixes is None else pls_prefixes
    prep_pip, _ = make_prep_pip(prefixes=prefixes if use_pls else [])
    retained_steps = []
    pls_audit = []
    for name, original in prep_pip.steps:
        step = deepcopy(original)
        sources = [v for v in step.pls_columns if v in considered and v not in fixed]
        if target in sources:
            raise ValueError("The target cannot be a PLS source.")
        if len(sources) < 2:
            pls_audit.append(dict(group=name, status="kept_explicit_fewer_than_two_sources", sources=sources))
            continue
        step.pls_columns = sources
        step.n_components = min(step.n_components, len(sources))
        retained_steps.append((name, step))
        pls_audit.append(dict(group=name, status="replaced_by_pls", sources=sources,
                              protected_fixed=[v for v in original.pls_columns if v in fixed]))
    prep_pip.steps = retained_steps
    print("RUNNING FIXED SUBSET SELECTION v1 | 6h defaults | inclusion list")
    print("Fixed variables:", fixed)
    diff, pairs, raw, smoothed, physical = prepare_differencing_data(
        engineered, target, lag_days=lag_days, halflife=ewm_halflife,
        tolerance=pair_tolerance, max_gap=max_gap, same_segment_only=True,
        start_date=start_date, end_date=end_date, considered_vars=considered,
        fixed_vars=fixed, creator_class=FeatureCreator, prep_pip=prep_pip,
        pre_estimator_builder=build_pre_estimator)
    selection = select_subset_survivors(
        diff, target, alpha=alpha_conditional, alpha_marginal=alpha_marginal,
        max_k=max_k, max_states=max_states, max_tests=max_tests,
        verbose=verbose, fixed_vars=fixed)
    table = selection["selected_df"].rename(columns={"r":"weakest_subset_r", "p":"p_at_weakest_subset"}).copy()
    table["marginal_r"] = table.variable.map(lambda v: diff[v].corr(diff[target]))
    table["attenuation_pct"] = 100*(1-table.weakest_subset_r.abs()/table.marginal_r.abs().replace(0,np.nan))
    # For blocked fixed rows, r/p refer to a recorded separator, not an exhaustive minimum.
    table["evidence_type"] = np.where(table.get("status",pd.Series(index=table.index,dtype=object)).eq("blocked_but_fixed"),
                                      "recorded_separator_fixed", "weakest_tested_subset")
    result = dict(analysis_df_diff=diff, pair_info=pairs, turnup_ts_raw=raw,
                  turnup_ts=smoothed, physical_ts=physical, results=selection,
                  selected_table=table, pls_audit=pls_audit,
                  preparation=diff.attrs["preparation"])
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        diff.to_parquet(folder/"analysis_df_diff.parquet")
        pairs.to_parquet(folder/"pair_info.parquet")
        physical.to_parquet(folder/"physical_ts.parquet")
        table.to_csv(folder/"selected_features.csv", index=False)
        for key in ["separators_df","fixed_audit_df","unresolved_df","aliases_df","history_df"]:
            selection[key].to_json(folder/(key+".json"), orient="records", indent=2)
        (folder/"metadata.json").write_text(json.dumps(dict(selection=selection["metadata"],
            preparation=result["preparation"],pls_audit=pls_audit),indent=2,default=str))
        print("Saved results:", folder.resolve())
    return result


"""Subset-survival selection trial. Input: existing prepared analysis_df_diff."""
import heapq
from itertools import combinations
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def select_subset_survivors(data, target, alpha=.05, max_k=3,
                           max_states=500, max_tests=500_000, verbose=True, alpha_marginal=None, fixed_vars=()):
    """Search alternative forward/backward choices; fixed variables are retained and ordinary blocked features are removable.

    Completed means both conditions hold up to max_k on the supplied CI tests.
    It does not mean a globally optimal set or causal identification.
    Numerically identical standardized columns are represented once and audited.
    """
    if alpha_marginal is None: alpha_marginal = alpha
    if not 0 < alpha_marginal < 1:
        raise ValueError('alpha_marginal must be between zero and one.')
    if target not in data or not data.columns.is_unique:
        raise ValueError('Require a target and unique column names.')
    if not 0 < alpha < 1 or max_k < 0 or max_states < 1 or max_tests < 1:
        raise ValueError('Invalid search parameters.')
    fixed_vars = list(dict.fromkeys(fixed_vars))
    if target in fixed_vars or set(fixed_vars)-set(data.columns):
        raise ValueError("Fixed variables must be predictor columns in the data.")
    data = data[[target] + fixed_vars + [c for c in data if c != target and c not in fixed_vars]]
    a=data.to_numpy(float)
    if not np.isfinite(a).all() or np.any(a.std(0)<=1e-12):
        raise ValueError('Data must be finite and nonconstant.')
    if len(a)<=max_k+3: raise ValueError('Insufficient observations.')
    a=(a-a.mean(0))/a.std(0)
    cols=list(data.columns); ti=cols.index(target)
    # Resolve equivalent representations before CI testing, retaining input order.
    representatives=[]; aliases=[]
    for j,v in enumerate(cols):
        if j==ti: continue
        equivalent=next((k for k in representatives
            if min(np.max(np.abs(a[:,j]-a[:,k])),np.max(np.abs(a[:,j]+a[:,k])))<1e-7),None)
        if equivalent is None: representatives.append(j)
        elif v in fixed_vars:
            raise ValueError(f'Fixed variables are numerically equivalent: {v}, {cols[equivalent]}')
        else: aliases.append({'variable':v,'representative':cols[equivalent],
                              'reason':'numerically identical standardized representation'})
    names=[cols[j] for j in representatives]; m=len(names); n=len(a)
    corr=np.corrcoef(a[:,[ti]+representatives],rowvar=False)
    return _search_subset_states(names, lambda x,z: _partial_test(corr,n,x,z),
        alpha,max_k,max_states,max_tests,verbose,aliases,target,alpha_marginal,fixed_vars)


def _partial_test(corr,n,x,z):
    ids=[0,x+1]+[v+1 for v in z]
    c=corr[np.ix_(ids,ids)]
    if z:
        zz=c[2:,2:]
        eigen=np.linalg.eigvalsh(zz)
        if eigen[0]<=1e-10*max(1.,eigen[-1]): return np.nan,np.nan
        residual=c[:2,:2]-c[:2,2:] @ np.linalg.solve(zz,c[2:,:2])
    else: residual=c[:2,:2]
    if min(np.diag(residual))<=1e-10:return np.nan,np.nan
    r=float(np.clip(residual[0,1]/np.sqrt(residual[0,0]*residual[1,1]),-1,1))
    df=n-len(z)-2
    p=float(2*student_t.sf(abs(r)*np.sqrt(df/max(1-r*r,1e-15)),df))
    return r,p


def _search_subset_states(names,test,alpha,max_k,max_states,max_tests,verbose,aliases,target,alpha_marginal=None,fixed_vars=()):
    fixed = {names.index(v) for v in fixed_vars}
    if alpha_marginal is None: alpha_marginal = alpha
    calls=0
    class Limit(Exception): pass
    @lru_cache(maxsize=500_000)
    def ci(x,z):
        nonlocal calls
        if calls>=max_tests: raise Limit()
        calls+=1
        return test(x,z)
    # Multiple known separators are reused whenever their members remain selected.
    known={x:[] for x in range(len(names))}
    def inspect(x,others):
        for z,r,p in known[x]:
            if set(z).issubset(others):return (z,r,p),None,False
        weakest=None;invalid=False
        for k in range(min(max_k,len(others))+1):
            for z in combinations(sorted(others),k):
                r,p=ci(x,z)
                if not np.isfinite(r) or not np.isfinite(p):invalid=True;continue
                if weakest is None or abs(r)<abs(weakest[1]):weakest=(z,r,p)
                if p>(alpha_marginal if not z else alpha):
                    known[x].append((z,r,p));return (z,r,p),weakest,invalid
        return None,weakest,invalid
    def record(x,e):
        z,r,p=e
        return dict(variable=names[x],conditioning_set=tuple(names[j] for j in z),r=r,p=p)
    queue=[];queued=set();visited=set();parent={};serial=0
    def push(s,prev=None,action=None,var=None):
        nonlocal serial
        s=frozenset(s)
        if s in queued:return
        queued.add(s);serial+=1;parent[s]=(prev,action,var)
        # Prefer larger admissible candidates; deterministic insertion order breaks ties.
        heapq.heappush(queue,(-len(s),serial,s))
    push(fixed);best=None;solution=None;reason='state_limit'
    allvars=set(range(len(names)))
    try:
        while queue and len(visited)<max_states:
            _,_,s=heapq.heappop(queue);visited.add(s)
            blocked=[];strengths=[];unknown=False
            for x in sorted(s-fixed):
                sep,weak,invalid=inspect(x,s-{x});unknown|=invalid
                if sep is not None:blocked.append((x,sep))
                elif weak is not None:strengths.append(record(x,weak))
            if blocked:
                for x,sep in blocked:push(s-{x},s,'remove',x)
                continue
            # This state passes all evaluable survivor tests. Revisit all omissions.
            rejected=[];eligible=[]
            for x in sorted(allvars-s):
                sep,weak,invalid=inspect(x,s)
                if sep is not None:rejected.append(record(x,sep))
                else:
                    unknown|=invalid
                    eligible.append((x,weak))
            state=dict(selected=s,strengths=strengths,separators=rejected,
                       eligible=eligible,unknown=unknown)
            if best is None or len(eligible)<len(best['eligible']):best=state
            if verbose:print(f'States {len(visited)} | selected {len(s)} | unblocked omitted {len(eligible)} | CI tests {calls:,}',flush=True)
            if not eligible and not unknown:solution=state;reason='complete';break
            eligible.sort(key=lambda item: -(abs(item[1][1]) if item[1] else 0))
            for x,_ in eligible:push(s|{x},s,'add',x)
        else:
            reason='state_limit' if queue else 'no_verified_solution_in_reachable_states'
    except Limit:reason='test_limit'
    state=solution or best
    if state is None:state=dict(selected=frozenset(fixed),strengths=[],separators=[],eligible=[(x,None) for x in range(len(names))],unknown=True)
    s=state['selected'];path=[];cur=s
    while cur in parent and parent[cur][0] is not None:
        prev,action,x=parent[cur];path.append(dict(action=action,variable=names[x],selected_count=len(cur)));cur=prev
    fixed_audit=[]
    for x in sorted(fixed):
        try:
            sep,weak,invalid=inspect(x,s-{x})
            evidence=sep or weak
            row=record(x,evidence) if evidence else dict(variable=names[x],conditioning_set=None,r=np.nan,p=np.nan)
            row['status']='blocked_but_fixed' if sep else ('invalid_tests' if invalid else 'passes_tested_subsets')
        except Limit:
            row=dict(variable=names[x],conditioning_set=None,r=np.nan,p=np.nan,status='audit_budget_exhausted')
        fixed_audit.append(row)
    selected_df=pd.DataFrame(state['strengths'],columns=['variable','conditioning_set','r','p'])
    selected_df['role']='selected'
    fixed_df=pd.DataFrame(fixed_audit)
    if len(fixed_df):
        fixed_rows = fixed_df.assign(role='fixed')
        selected_df = (fixed_rows.copy() if selected_df.empty else
                       pd.concat([fixed_rows,selected_df],ignore_index=True))

    omitted_df=pd.DataFrame(state['separators'],columns=['variable','conditioning_set','r','p'])
    metadata=dict(target=target,complete=solution is not None,stop_reason=reason,
        selected_count=len(s),fixed_vars=list(fixed_vars),fixed_count=len(fixed),states_visited=len(visited),ci_tests=calls,alpha=alpha,alpha_conditional=alpha,alpha_marginal=alpha_marginal,max_k=max_k,
        unblocked_omitted_count=len(state['eligible']),invalid_survivor_or_candidate_tests=state['unknown'],
        note='Any incomplete output is provisional. Completion is bounded subset-test consistency, not global optimality. Fixed variables are exempt from survival; see fixed_audit_df.')
    print(metadata)
    return dict(selected_vars=[names[x] for x in sorted(s)],selected_df=selected_df,
        fixed_audit_df=fixed_df,separators_df=omitted_df,unresolved_df=pd.DataFrame([{'variable':names[x]} for x,_ in state['eligible']]),
        aliases_df=pd.DataFrame(aliases),history_df=pd.DataFrame(path[::-1]),metadata=metadata)


def independence_diagnostics(data, target, selected_vars, fixed_vars=(),
                             max_k=3, alpha=0.05, alpha_marginal=None):
    """Per-variable conditional-independence metrics for an ALREADY-selected subset.

    For every variable in ``selected_vars`` this quantifies *how close it came to
    being conditionally independent of the target* given the other selected
    variables -- i.e. how robustly it survived. It does not change the selection;
    it only reports diagnostics (intended for results.json).

    Parameters
    ----------
    data : DataFrame
        The same CI frame handed to ``select_subset_survivors`` (column 0 is the
        target; values finite and non-constant).
    target : str
    selected_vars : sequence of str
        The variables whose survival strength we report (the final CI subset).
    fixed_vars : sequence of str
        Variables forced to stay; reported with ``role='fixed'`` so a surviving
        forced variable is not mistaken for one the test endorsed.
    max_k : int
        Max conditioning-set size searched (same meaning as in the selector).
    alpha, alpha_marginal : float
        Thresholds used to flag separation. ``alpha`` applies to non-empty
        conditioning sets, ``alpha_marginal`` to the empty (marginal) set. These
        must match the selection run for the flags to be comparable.

    Returns
    -------
    dict
        ``{"target", "alpha", "alpha_marginal", "max_k", "variables": [ ... ]}``
        where each entry in ``variables`` is a dict with:
          - ``variable``, ``role`` ('selected' | 'fixed')
          - ``marginal_r``, ``marginal_p``: unconditioned correlation with target
          - ``min_abs_partial_r``: smallest |partial r| over all tested
            conditioning subsets (0 = looked conditionally independent)
          - ``max_p``: largest p-value reached (closest it came to the threshold)
          - ``weakest_conditioning_set``: the subset achieving ``min_abs_partial_r``
          - ``survives``: True if no tested subset separated it (p never exceeded
            the applicable threshold), i.e. the relationship survived conditioning
          - ``separating_set``: the first subset that DID separate it, or None
          - ``margin_to_alpha``: alpha - max_p for a survivor (how close to being
            dropped; small positive = borderline). None if it did not survive.
          - ``borderline``: True if it survived but ``margin_to_alpha`` is small
          - ``n_invalid_tests``: count of non-evaluable (nan) tests encountered
    """
    if alpha_marginal is None:
        alpha_marginal = alpha
    selected_vars = unique_in_order(selected_vars)
    fixed_set = set(fixed_vars)
    missing = [v for v in selected_vars if v not in data.columns] + (
        [] if target in data.columns else [target])
    if missing:
        raise ValueError(f"Variables absent from the CI frame: {sorted(set(missing))}")

    cols = [target] + [v for v in selected_vars if v != target]
    a = data[cols].to_numpy(float)
    if not np.isfinite(a).all() or np.any(a.std(0) <= 1e-12):
        raise ValueError("CI frame must be finite and non-constant.")
    a = (a - a.mean(0)) / a.std(0)
    n = len(a)
    corr = np.corrcoef(a, rowvar=False)
    others = cols[1:]
    idx = {v: i for i, v in enumerate(others)}  # 0-based position among predictors

    def _borderline_margin(threshold):
        # a survivor is "borderline" if it came within 20% of alpha of the cutoff
        return 0.2 * threshold

    results = []
    for v in others:
        x = idx[v]
        pool = [idx[o] for o in others if o != v]
        marginal_r, marginal_p = _partial_test(corr, n, x, ())
        min_abs_r = None
        max_p = None
        weakest_z = None
        separating_z = None
        n_invalid = 0
        for k in range(min(max_k, len(pool)) + 1):
            for z in combinations(sorted(pool), k):
                r, p = _partial_test(corr, n, x, z)
                if not np.isfinite(r) or not np.isfinite(p):
                    n_invalid += 1
                    continue
                if min_abs_r is None or abs(r) < min_abs_r:
                    min_abs_r = abs(r)
                    weakest_z = z
                if max_p is None or p > max_p:
                    max_p = p
                threshold = alpha_marginal if not z else alpha
                if p > threshold and separating_z is None:
                    separating_z = z
        survives = separating_z is None
        # The margin uses the conditional alpha; for a pure-marginal survivor this
        # is conservative, which is the safe direction for a "should I drop it?" read.
        margin = (alpha - max_p) if (survives and max_p is not None) else None
        borderline = bool(survives and margin is not None
                          and margin <= _borderline_margin(alpha))
        results.append(dict(
            variable=v,
            role="fixed" if v in fixed_set else "selected",
            marginal_r=float(marginal_r) if np.isfinite(marginal_r) else None,
            marginal_p=float(marginal_p) if np.isfinite(marginal_p) else None,
            min_abs_partial_r=float(min_abs_r) if min_abs_r is not None else None,
            max_p=float(max_p) if max_p is not None else None,
            weakest_conditioning_set=[others[j] for j in weakest_z] if weakest_z is not None else None,
            survives=bool(survives),
            separating_set=[others[j] for j in separating_z] if separating_z is not None else None,
            margin_to_alpha=float(margin) if margin is not None else None,
            borderline=borderline,
            n_invalid_tests=int(n_invalid),
        ))
    # Strongest survivors first (smallest max_p = most dependent); None sorts last.
    results.sort(key=lambda d: (d["max_p"] is None, d["max_p"] if d["max_p"] is not None else 1.0))
    return dict(target=target, alpha=alpha, alpha_marginal=alpha_marginal,
                max_k=max_k, variables=results)
