import pandas as pd
import numpy as np
import re
 
 
def convert_to_dataframe(data, tag_list: list) -> pd.DataFrame:
    """
    Build the dataframe in one shot instead of creating many small DataFrames.
    """
    cols = {}
    first = True
 
    for d in data:
        tag_name = d["TagName"]
        if first:
            cols["TimeStamp"] = d["TimeStamp"]
            first = False
        cols[tag_name] = d["Value"]
 
    result_df = pd.DataFrame(cols)
    ordered_cols = ["TimeStamp"] + [c for c in tag_list if c in result_df.columns]
    return result_df.loc[:, ordered_cols]
 
 
def meta_data(data) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Units": [d["Units"] for d in data],
            "tag_name": [d["TagName"] for d in data],
        }
    )
 
 
def replace_tags_with_name(df: pd.DataFrame, df_tags: pd.DataFrame) -> pd.DataFrame:
    extra_row = pd.DataFrame(
        [{
            "Name": "Wedge Time",
            "Tags": df.columns[0],
            "TagNumber": 0
        }]
    )
 
    df_tags2 = pd.concat([extra_row, df_tags], ignore_index=True)
    df_tags2 = df_tags2[df_tags2["TagNumber"].notna()]
 
    tags = [t for t in df_tags2["Tags"].tolist() if t in df.columns]
    rename_map = dict(zip(df_tags2["Tags"], df_tags2["Name"]))
 
    df = df.loc[:, tags]
    df = df.rename(columns=rename_map, copy=False)
 
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df.loc[:, num_cols] = df.loc[:, num_cols].replace([np.inf, -np.inf], np.nan)
 
    return df
 
 
def paper_break(df: pd.DataFrame) -> pd.DataFrame:
    break_col = 'O=Alles OK sonst Abrissort li:PM1-OPC-DATA.F:break_location'
 
    break_mask = df[break_col] > 0
    shut_mask = break_mask & (df['Stock pump power'] < 50)
 
    df['Break'] = break_mask.astype('int8')
    df['Shut indicator'] = shut_mask.astype('int8')
    df['Sheet Off'] = (df['Break'] | df['Shut indicator']).astype('int8')
 
    return df.drop(columns=[break_col, 'Break', 'Shut indicator'])
 
 
def draws(df: pd.DataFrame) -> pd.DataFrame:
    df['Draw AD7-PR'] = ((df['Speed'] / df['AD7 speed top']) - 1) * 100
    df['Draw SS-AD6'] = ((df['AD6 speed'] / df['Speed Size Press']) - 1) * 100
    df['Draw AD6-AD7'] = ((df['AD7 speed bottom'] / df['AD6 speed']) - 1) * 100
    df['Draw PD5-SS'] = ((df['Speed Size Press'] / df['Speed PD5 top']) - 1) * 100
    df['Draw PS-PD1'] = ((df['Speed PD1'] / df['Speed press section']) - 1) * 100
    df['Draw PD2-PD3'] = ((df['Speed PD3'] / df['Speed PD2']) - 1) * 100
    df['Draw PD4-PD5'] = ((df['Speed PD5 top'] / df['Speed PD4 top']) - 1) * 100
    df['Draw PD3-PD4'] = ((df['Speed PD4 top'] / df['Speed PD3']) - 1) * 100
    df['Draw PD1-PD2'] = ((df['Speed PD2'] / df['Speed PD1']) - 1) * 100
    df['Draw WS-PS'] = ((df['Speed press section'] / df['Forming Wire Speed']) - 1) * 100
    return df
 
 
def starch_application(df: pd.DataFrame) -> pd.DataFrame:
    prod_sqm = df['Aktuelle Arbeitsbreite in cm~^0'] / 100 * df['AD7 speed bottom'] * 60
    slurry_fw = df['Dilution water working tank 1'] + df['Flow starch main line to working tank 1~^0']
    slurry_bw = df['Dilution water working tank 2'] + df['Flow starch main line to working tank 2~^0']
 
    df['Produzierte Quadratmeter'] = prod_sqm
    df['Auftrag Stärkeslurry FW'] = slurry_fw
    df['Auftrag Stärkeslurry BW'] = slurry_bw
    df['Starch application FW in ml'] = slurry_fw * 1_000_000 / prod_sqm
    df['Starch application BW in ml'] = slurry_bw * 1_000_000 / prod_sqm
 
    return df
 
 
def slurry(df: pd.DataFrame) -> pd.DataFrame:
    df['Slurry 1 pumping to reactor'] = (df['Flow slurry 1 to reactor'] > 50).astype('int8')
    df['Slurry 2 pumping to reactor'] = (df['Flow slurry 2 to reactor'] > 50).astype('int8')
    df['Slurry 3 pumping to reactor'] = (df['Flow slurry 3 to reactor'] > 50).astype('int8')
    return df
 
 
def dewatering(df: pd.DataFrame) -> pd.DataFrame:
    df['Total Dewatering Pick-Up'] = df['Dewatering Pick-Up'] + df['Dewatering Suction Press Roll']
    df['Total Dewatering Bottom Felt'] = df['Dewatering First Press Roll'] + df['Uhle box 1 flow [ l/min]']
    return df
 
 
def multifractionator(df: pd.DataFrame) -> pd.DataFrame:
    df['Multifractor 1 Long fibre fraction'] = (
        df['Multifractor 1 long fibre flow'] / df['Multifractor 1 short fibre flow']
    ) * 100
    df['Multifractor 2 long fibre fraction'] = (
        df['Multifractor 2 long fibre flow'] / df['Multifractor 2 short fibre flow']
    ) * 100
    df['Multifractor 3 long fibre fraction'] = (
        df['Multifractor 3 long fibre flow'] / df['Multifractor 3 short fibre flow']
    ) * 100
    return df
 
 
def steam(df: pd.DataFrame) -> pd.DataFrame:
    df['Steam flow from power plant to PM'] = (
        df['Steam flow to PM']
        - df['Steam flow to white water heating']
        - df['Steam flow to steam box']
        - df['Steam flow to hall heating']
        - df['Steam flow to heat exchangers']
    )
 
    df['Steam flow to PreDryers'] = (
        df['Steam flow from power plant to PM']
        - df['Steam flow to AfterDryers']
        + df['Waste steam flow']
    )
    return df
 
 
def flows(df: pd.DataFrame) -> pd.DataFrame:
    pr = df['Production Rate [T/h]']
 
    df['Retention Aid mass flow [g/T]'] = (df['Retentionsmittel Menge l/h [0..9000 l/h] '] * 0.004 / pr) * 1000
    df['Bentonite 1 mass flow [g/T]'] = (df['Bentonitmenge z. Mischrohr [-10..9000 l/h]'] * 0.025 / pr) * 1000
    df['Bentonite 2 mass flow [g/T]'] = (df['Bentonitmenge z. Siebwasserturm [-10..9000 l/h]'] * 0.025 / pr) * 1000
 
    fix2_raw = df['Fixiermittelmenge Trockenausschuss [0..100 l/h] ']
    df['Fixiermittelmenge check'] = (fix2_raw > 5).astype('int8')
    df['Fixative 2 mass flow [g/T]'] = fix2_raw / pr * 1000 * df['Fixiermittelmenge check']
 
    df['Act Deaerator mass flow [g/T]'] = (
        df['Mengenregelung Entlüfter : MESSWERT      [0..40 l/h]'] / pr * 1000
    )
 
    pb = df['Paper break status li:PM1-OPC-DATA.F:rtup_pls (0 = false)']
    df['Paper break status check'] = (pb >= 0).astype('int8')
    df['Act Production Rate Gross'] = pr * df['Paper break status check']
 
    df['Starch mass flow [kg/T]'] = (
        (
            ((df['Flow starch main line to working tank 2~^0'] * 0.215) +
             (df['Flow starch main line to working tank 1~^0'] * 0.215)) * 1.085
            + (271 / (24 * 30))
        ) / pr
    ) * 1000
 
    df['Defoamer mass flow [g/T]'] = (
        (df['Entschäumermenge BW [-2..60 l/h]'] + df['Entschäumermenge FW [-1..60 l/h]']) / pr
    ) * 1000
 
    leim = (
        df['Mengenistwert Leimungsmittel BW : MESSWERT     [0..500 l/h]'] +
        df['Menge Leimungsmittel : MESSWERT      [0..500 l/h]']
    ) / pr * 1000
    df['Leimungsmenge gesamt pro Tonne'] = leim
    df['Leimungsmenge gesamt pro Tonne check'] = (leim >= 2000).astype('int8')
    df['Sizing Agent [g/T]'] = leim * df['Leimungsmenge gesamt pro Tonne check']
 
    trock = (
        df['Trockenverfestigermenge bewegliche Walze [0..2500 l/h]'] +
        df['Trockenverfestigermenge feste Walze [0..2500 l/h]']
    )
    df['Trockenverfestigermenge gesamt'] = trock
    df['Trockenverfestigermenge gesamt check'] = (trock >= 200).astype('int8')
    df['Dry Strength Agent mass flow [kg/T]'] = trock / pr * df['Trockenverfestigermenge gesamt check'] * 1.14
 
    df['Natriumhydroxide mass flow [g/T]'] = (
        (
            df['SP1: 1-000-FC028_01    NaOH-Dosierung Pulper :  MESSWERT'] +
            df['SP1: 1-000-FC028_02    NaOH-Dosierung Kurzfaser-SF :  MESSWE'] +
            df['SP1: 1-000-FC028_03    NaOH-Dosierung Langfaser-SF :  MESSWE']
        ) / pr
    ) * 1000
 
    df['CO2 mass flow [g/T]'] = (
        (
            df['CO2-Dosierung Deculator 1 [0..60 kg/h]'] +
            df['CO2-Dosierung Deculator 2 [0..60 kg/h]'] +
            df['CO2-Dosierung Bio-Rückwasser [0..60 kg/h]']
        ) / pr
    ) * 1000
 
    df['Summe Energieverbrauch APA in kW'] = (
        df['APA 6KV [-1..10 MW]'] +
        df['APA 660V [-1..10 MW]'] +
        df['APA 400V [-1..10 MW]'] +
        df['APA 660V [-1..10 MW]~^0']
    ) * 1000
 
    df['Spezifischer Energieverbrauch APA '] = df['Summe Energieverbrauch APA in kW'] / pr
 
    df['Electrical Consumtion Wire Section'] = df['Mehrmotorenantrieb Former [-1..10 MW]'] * 1000 / pr
    df['Press and Steambox'] = df['Mehrmotorenantrieb Presse [-1..10 MW]'] * 1000 / pr
 
    df['PM Freq Converters 1 to 3'] = (
        df['VariSTEP div. Hauben+Lueftung Ventilatoren'] +
        df['Varisprint 660V [-10..10 MW]'] +
        df['Varisprint 400V [-10..10 MW]']
    ) * 1000 / pr
 
    df['Other PM 1 to 10'] = (
        df['Mehrmotorenantrieb Trockenpartie [-1..10 MW]'] +
        df['PM 400V [-1..10 MW]'] +
        df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^1'] +
        df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^0'] +
        df['PM 660V Festantriebe Verteiler2 [-1..10 MW]'] +
        df['PM 660V Festantriebe Verteiler2 [-1..10 MW]~^1'] +
        df['PM 660V Festantriebe Verteiler3 [-1..10 MW]~^0'] +
        df['PM 660V Festantriebe Verteiler3 [-1..10 MW]'] +
        df['MCC Einzel-FU 660V [-10..10 MW]~^0'] +
        df['MCC Einzel-FU 660V [-10..10 MW]~^1']
    ) * 1000 / pr
 
    df['Predryers sensor 10 to 25'] = (
        df['Last Antrieb 10 1.TG [0..100 kW]'] +
        df['Last Antrieb 11 1.TG [0..100 kW]'] +
        df['Last Antrieb 12 2.TG [0..100 kW]'] +
        df['Last Antrieb 13 2.TG [0..100 kW]'] +
        df['Last Antrieb 14 2.TG [0..100 kW]'] +
        df['Last Antrieb 15 3.TG [0..100 kW]'] +
        df['Last Antrieb 16 3.TG [0..100 kW]'] +
        df['Last Antrieb 17 3.TG [0..100 kW]'] +
        df['Last Antrieb 18 4.TG [0..100 kW]'] +
        df['Last Antrieb 19 4.TG [0..100 kW]'] +
        df['Last Antrieb 20 4.TG [0..100 kW]'] +
        df['Last Antrieb 21 4.TG [0..100 kW]'] +
        df['Last Antrieb 22 5.TG [0..100 kW]'] +
        df['Last Antrieb 23 5.TG [0..100 kW]'] +
        df['Last Antrieb 24 5.TG [0..100 kW]'] +
        df['Last Antrieb 25 5.TG [0..100 kW]']
    ) / pr
 
    df['Afterdryers sensor 1 to 9'] = (
        df['Last Antrieb 31 6.TG [0..100 kW]'] +
        df['Last Antrieb 32 6.TG [0..100 kW]'] +
        df['Last Antrieb 33 6.TG [0..100 kW]'] +
        df['Last Antrieb 34 6.TG [0..100 kW]'] +
        df['Last Antrieb 35 6.TG [0..100 kW]'] +
        df['Last Antrieb 36 7.TG [0..200 kW]'] +
        df['Last Antrieb 37 7.TG [0..200 kW]'] +
        df['Last Antrieb 38 7.TG [0..200 kW]'] +
        df['Last Antrieb 39 7.TG [0..200 kW]']
    ) / pr
 
    df['Electricity [kWh/T]'] = (
        df['Electrical Consumtion Wire Section'] +
        df['Press and Steambox'] +
        df['PM Freq Converters 1 to 3'] +
        df['Other PM 1 to 10'] +
        df['Predryers sensor 10 to 25'] +
        df['Afterdryers sensor 1 to 9'] +
        df['Spezifischer Energieverbrauch APA ']
    )
 
    cond_col = df['Condensate energy from paper plant to power plant']
    cond_med = cond_col[(cond_col >= 0) & (cond_col <= 10)].median()
    cond = cond_col.where((cond_col >= 0) & (cond_col <= 10), cond_med)
 
    pr_safe = pr.where(pr > 45, 45)
 
    df['Steam [kWh/T]'] = (
        ((((df['Steam flow to PM'] + df['Waste steam flow']) * 0.788) - cond) * 1.02 - (0.5938 / 24))
        / pr_safe
    ) * 1000
 
    df['MC_SF_LF_Demand'] = (
        (df['Short fibre flow'] * (df['Short fibre B06 consistency'] / 100)) +
        (df['Long fibre flow'] * (df['Long fibre consistency B07'] / 100))
    ) * (60 / 1000)
 
    df['Whitewater_solids_flow'] = (
        (df['Short fibre flow'] + df['Long fibre flow']) *
        (df['Consistency white water'] / 1000)
    ) * (60 / 1000)
 
    df['Approach_flow_returns'] = df['Inlet Thickerner 2 [m3/h]'] * (1.74 / 100)
 
    df['Fibre usage [T/T]'] = (
        ((((df['MC_SF_LF_Demand'] - df['Whitewater_solids_flow'] - df['Approach_flow_returns']) * 1.5688) - 23.1) * 1.1)
        / pr
    )
 
    return df
 
 
def SQM_calc(df: pd.DataFrame) -> pd.Series:
    denom = 1 - df['AB Grade ID'].astype(str).str.replace(".0", "", regex=False).str[-3:].astype(int)
    return (
        df['Production Rate [T/h]']
        * df['Combined cost [€/T]']
        * df['Current basis weight']
        / denom
    )
 
 
def cost(df: pd.DataFrame) -> pd.DataFrame:
    df['Retention Aid [€/T]'] = df['Retention Aid mass flow [g/T]'] * 4.08 / 1000
    df['Bentonite 1 [€/T]'] = df['Bentonite 1 mass flow [g/T]'] * 0.28 / 1000
    df['Bentonite 2 [€/T]'] = df['Bentonite 2 mass flow [g/T]'] * 0.28 / 1000
    df['Fixative 2 [€/T]'] = df['Fixative 2 mass flow [g/T]'] * 3.58 / 1000
    df['Deaerator [€/T]'] = df['Act Deaerator mass flow [g/T]'] * 1.53 / 1000
    df['Starch [€/T]'] = df['Starch mass flow [kg/T]'] * 434.22 / 1000
    df['Defoamer [€/T]'] = df['Defoamer mass flow [g/T]'] * 4.22 / 1000
    df['Sizing Agent [€/T]'] = df['Sizing Agent [g/T]'] * 0.84 / 1000
    df['Dry Strength Agent [€/T]'] = df['Dry Strength Agent mass flow [kg/T]'] * 0.30
    df['Natriumhydroxide [€/T]'] = df['Natriumhydroxide mass flow [g/T]'] * 0.30 / 1000
    df['Electricity [€/T]'] = df['Electricity [kWh/T]'] * 113.66 / 1000
    df['Steam [€/T]'] = df['Steam [kWh/T]'] * 89.03 / 1000
    df['Fibre cost [€/T]'] = df['Fibre usage [T/T]'] * 146.46
 
    df['Combined cost [€/T]'] = (
        df['Retention Aid [€/T]'] +
        df['Bentonite 1 [€/T]'] +
        df['Bentonite 2 [€/T]'] +
        df['Fixative 2 [€/T]'] +
        df['Deaerator [€/T]'] +
        df['Starch [€/T]'] +
        df['Defoamer [€/T]'] +
        df['Sizing Agent [€/T]'] +
        df['Dry Strength Agent [€/T]'] +
        df['Natriumhydroxide [€/T]'] +
        df['Electricity [€/T]'] +
        df['Steam [€/T]'] +
        df['Fibre cost [€/T]']
    )
 
    df['Total SQM cost [€/hm2]'] = SQM_calc(df)
    return df
 
 
def summarize_dataframe(df: pd.DataFrame, name: str = "Dataframe"):
    numeric_df = df.select_dtypes(include=[np.number])
 
    if numeric_df.empty:
        inf_pct = 0
        extreme_pct = 0
    else:
        vals = numeric_df.to_numpy(copy=False)
        inf_pct = np.isinf(vals).mean() * 100
        extreme_pct = ((vals == 999999) | (vals == -999999)).mean() * 100
 
    summary = {
        "Name": name,
        "Shape": df.shape,
        "Missing Values (%)": df.isna().to_numpy().mean() * 100,
        "Infinite Values (%)": inf_pct,
        "Substituted infinite values": extreme_pct,
        "Duplicate Rows": df.duplicated().sum()
    }
    return summary
 
 
def compare_dataframes(before: pd.DataFrame, after: pd.DataFrame, name: str):
    before_summary = summarize_dataframe(before, f"Before {name}")
    after_summary = summarize_dataframe(after, f"After {name}")
    comparison_df = pd.DataFrame([before_summary, after_summary])
    print(comparison_df)
 
 
def combined_calculations(df: pd.DataFrame, compare: bool = False) -> pd.DataFrame:
    df_initial = df.copy(deep=False) if compare else None
 
    df = paper_break(df)
    df = draws(df)
    df = starch_application(df)
    df = slurry(df)
    df = dewatering(df)
    df = multifractionator(df)
    df = steam(df)
    df = flows(df)
    df = cost(df)
 
    if compare:
        compare_dataframes(df_initial, df, "Combined calculations")
 
    df = df.replace([np.inf, -np.inf], [999999, -999999])
    return df
 
 
def alligning_dataframe(df: pd.DataFrame, df_gloss: pd.DataFrame) -> pd.DataFrame:
    drop_list = [
        'Contact pressure support drum OS', 'Contact pressure support drum DS',
        'Fixative 1 [€/T]', 'Biozid - ClO2 [€/T]', 'CO2 [€/T]',
        'Fixative 1 mass flow [g/T]', 'Biozid - ClO2 mass flow [g/T]',
        'Rodsize BottomRoll', 'Rod Size TopRoll', 'Power Vacuum pump press section 305',
        'Power Vacuum pump 308 (reserve)', 'Power Vacuum pump press section 306',
        'Ratio of Dewatering of Pick UP of Total Dewatering Press',
        'Total Dewatering Top Felt', 'Ratio of Dewatering of Bottom Felt of Total Dewatering',
        'Ratio of Double Felt Press on Total Dewatering',
        'Total Felt Dewatering Press Two', 'Total Dewatering Press',
        'Condensate cloudidity'
    ]
 
    features = df_gloss.loc[~df_gloss['Feature'].isin(drop_list), 'Feature']
    features = [c for c in features if c in df.columns]
 
    return df.loc[:, features]
 
 
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns=lambda col: re.sub(r"[\[\]<>]", "_", str(col)).replace(" ", "_"),
        copy=False
    )
    return df
 
 
def ingestion_pipeline_API_alternative(
    df: pd.DataFrame,
    df_tags: pd.DataFrame,
    df_gloss: pd.DataFrame,
    compare: bool = False
) -> pd.DataFrame:
    df["R2NAE20CF905"] = df["R2NAE20CF905"].fillna(0)
 
    df_initial = df.copy(deep=False) if compare else None
 
    print("Step 1: Replace tags with corresponding names in PHD\n")
    df = replace_tags_with_name(df, df_tags)
 
    print("Step 2: Perform calculations")
    df = combined_calculations(df, compare=False)
 
    print("Step 3: Align dataframe with features from data glossary\n")
    df = alligning_dataframe(df, df_gloss)
 
    print("Step 4: Clean name of the columns\n")
    df = clean_column_names(df)
 
    if compare:
        print("Before and after of ingestion pipeline:\n")
        compare_dataframes(df_initial, df, "Ingestion Pipeline")
 
    return df
 
 
def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
 
    if len(float_cols) > 0:
        df.loc[:, float_cols] = df.loc[:, float_cols].apply(pd.to_numeric, downcast="float")
    if len(int_cols) > 0:
        df.loc[:, int_cols] = df.loc[:, int_cols].apply(pd.to_numeric, downcast="integer")
 
    return df