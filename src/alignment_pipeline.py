import pandas as pd
import numpy as np

def data_alignment(df, quality_list=['MBS_SCT_MD', 'MBS_SCT_CD', 'MBS_Burst', 'MBS_CMT30']):
    """
    Aligns data by adjusting the MBS_Current_reel_ID column, extracting relevant rows, 
    and merging process data with selected targets.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing the data to be aligned.

    Returns:
        pd.DataFrame: The aligned dataframe.
    """
    df_2 = df.copy()
    print(f"Number of records initially: {df_2.shape}")

    # Create quality dataset with MBS_Current_reel_ID
    df_quality = df_2[quality_list + ['MBS_Current_reel_ID']].copy()

    # Drop quality metrics from the extracted data
    df_process = df_2.drop(columns=quality_list)

    # Subtract one from reel ID quality to align it with the PHD
    df_quality.loc[:, 'MBS_Current_reel_ID'] = df_quality['MBS_Current_reel_ID'] - 1

    # Extract one row per MBS_Current_reel_ID from the quality dataset
    df_quality = df_quality.groupby('MBS_Current_reel_ID').tail(2).groupby('MBS_Current_reel_ID').head(1)

    # Merge quality dataset with the process data
    df_2 = pd.merge(df_quality, df_process, how='right', on='MBS_Current_reel_ID')
    print(f"Number of records after merging quality with process: {df_2.shape}")

    # Extract Wedge_Time from overall dataset
    time_column = df_2.pop('Wedge_Time')

    # Put back Wedge_Time into the dataset
    data_aligned = pd.concat([time_column, df_2], axis=1)

    return data_aligned

def assign_missing_values_to_grades_without_cmt30(df, grade_list):
    # Assign nan for predefined grades
    df.loc[df['AB_Grade_ID'].isin(grade_list), 'MBS_CMT30'] = np.nan

    return df

def drop_paperbreaks(df):
    print(f"Number of records before dropping paperbreaks: {df.shape}")

    # drop paperbreaks
    df_without_paperbreaks = df[df['Sheet_Off']==0].reset_index(drop=True)

    print(f"Number of records after dropping paperbreaks: {df_without_paperbreaks.shape}")

    return df_without_paperbreaks


def drop_grades_equal_to_zero(df):
    print(f"Number of records before removing rows with Grade = 0: {df.shape}")
    # drop zero grades
    df_new = df[df['AB_Grade_ID']!=0].reset_index(drop=True)
    print(f"Number of records after removing rows with Grade = 0: {df_new.shape}")

    return df_new

def create_turnup_and_continuous_data(df, vars_average=[], option="last"):

    if option=="median":
        df_final_turnup = df.groupby('MBS_Current_reel_ID').median().reset_index()
    else:
        # Select 10min to turnup
        df_final_turnup = df.groupby('MBS_Current_reel_ID').tail(2).groupby('MBS_Current_reel_ID').head(1)
        #df_final_turnup = df.groupby('MBS_Current_reel_ID').tail(10).groupby('MBS_Current_reel_ID').mean().reset_index()
    
    for v in vars_average:
        #df_final_turnup[v] = df_final_turnup["MBS_Current_reel_ID"].map(df.groupby('MBS_Current_reel_ID', dropna=False)[v].mean())
        df_final_turnup[v] = df["MBS_Current_reel_ID"].map(df.groupby('MBS_Current_reel_ID', dropna=False)[v].mean())
    
    print(f"(TURNUP) Number of records after selecting 10min to turnup: {df_final_turnup.shape}\n")
    # Drop duplicates
    # df_final_turnup_2 = df_final_turnup.drop_duplicates(subset=['MBS_SCT_MD', 'MBS_SCT_CD', 'MBS_Burst', 'AB_Grade_ID'], keep='first')
    df_final_turnup_2 = df_final_turnup.fillna('placeholder').drop_duplicates(subset=['MBS_SCT_MD', 'MBS_SCT_CD', 'MBS_Burst', 'AB_Grade_ID'], keep='first')
    # Replace the placeholder with NaN
    df_final_turnup_2 = df_final_turnup_2.replace('placeholder', np.nan)
    print(f"(TURNUP) Number of records after dropping duplicates: {df_final_turnup_2.shape}\n")

    # Create dropped duplicates dataframe
    df_dropped_reels = df_final_turnup[~df_final_turnup['MBS_Current_reel_ID'].isin(df_final_turnup_2['MBS_Current_reel_ID'])]
    # Drop duplicates from continuous data
    df_final_continuous = df[df['MBS_Current_reel_ID'].isin(df_final_turnup_2['MBS_Current_reel_ID'])]
    print(f"(CONTINUOUS) Number of records after dropping duplicates: {df_final_continuous.shape}\n")

    return df_final_turnup_2, df_final_continuous


