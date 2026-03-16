import pandas as pd
import numpy as np
import re


def convert_to_dataframe(data: pd.DataFrame, tag_list:list) -> pd.DataFrame:
    dfs = []
    i=0
    for d in data:
        tag_name = d['TagName']
        timestamps = d['TimeStamp']
        values = d['Value']
                   
        if i==0:
            df_data = {
                'TimeStamp': timestamps,
                f'{tag_name}': values
            }
        else:
            df_data = {
                f'{tag_name}': values
            }

        df = pd.DataFrame(df_data)
        dfs.append(df,)
        i=i+1

    result_df = pd.concat(dfs, axis=1)
    result_df = result_df[['TimeStamp'] + tag_list]

    return result_df



def meta_data(data:pd.DataFrame)->pd.DataFrame:
    dfs = []
    for d in data:
        tag_name = d['TagName']
        units = d['Units']

        df_meta = {
            'Units': units,
            'tag_name': tag_name
            }
        
        df = pd.DataFrame(df_meta, index=[0])
        dfs.append(df,)

    meta_df = pd.concat(dfs, axis=0)
    return meta_df


def replace_tags_with_name(df, df_tags) -> pd.DataFrame:
    # substituing the time column to Wedge Time - we should change that
    row = {
        'Name':'Wedge Time',
        'Tags':df.columns[0],
        'TagNumber':00
        }

    df_tags = pd.concat([pd.DataFrame([row]), df_tags], ignore_index=True)
    df_tags = df_tags[df_tags['TagNumber'].notna()]
    tags = df_tags["Tags"].tolist()
    rename_map = dict(zip(df_tags["Tags"], df_tags["Name"]))

    df = df.loc[:, tags]
    df.columns = df.columns.map(rename_map)

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].where(
        np.isfinite(df[num_cols]),
        np.nan
    )
    return df

def paper_break(df:pd.DataFrame) -> pd.DataFrame:
    #drop Break
    df.loc[(df['O=Alles OK sonst Abrissort li:PM1-OPC-DATA.F:break_location']>0),  'Break'] = 1
    df.loc[(df['Break']!=1),  'Break'] = 0

    #drop Shut indicator
    df.loc[(df['O=Alles OK sonst Abrissort li:PM1-OPC-DATA.F:break_location']>0) & (df['Stock pump power']<50),  'Shut indicator'] = 1
    df.loc[(df['Shut indicator']!=1),  'Shut indicator'] = 0

    #Sheet off
    df['Sheet Off'] = df['Shut indicator'] + df['Break']

    #dropping what is needed for calculation
    df = df.drop(columns=['O=Alles OK sonst Abrissort li:PM1-OPC-DATA.F:break_location', 'Break', 'Shut indicator'])
    return df


def draws(df:pd.DataFrame) -> pd.DataFrame:
    #Draw AD7-PR
    df['Draw AD7-PR'] = ((df['Speed']/df['AD7 speed top'])-1)*100

    #Draw SS-AD6
    df['Draw SS-AD6'] = ((df['AD6 speed']/df['Speed Size Press'])-1)*100

    #Draw AD6-AD7
    df['Draw AD6-AD7'] = ((df['AD7 speed bottom']/df['AD6 speed'])-1)*100

    #Draw PD5-SS
    df['Draw PD5-SS'] = ((df['Speed Size Press']/df['Speed PD5 top'])-1)*100

    #Draw PS-PD1
    df['Draw PS-PD1'] = ((df['Speed PD1']/df['Speed press section'])-1)*100

    #Draw PS-PD1
    df['Draw PD2-PD3'] = ((df['Speed PD3']/df['Speed PD2'])-1)*100

    #Draw PD4-PD5
    df['Draw PD4-PD5'] = ((df['Speed PD5 top']/df['Speed PD4 top'])-1)*100

    #Draw PD3-PD4
    df['Draw PD3-PD4'] = ((df['Speed PD4 top']/df['Speed PD3'])-1)*100

    #Draw PD1-PD2
    df['Draw PD1-PD2'] = ((df['Speed PD2']/df['Speed PD1'])-1)*100

    #Draw WS-PS
    df['Draw WS-PS'] = ((df['Speed press section']/df['Forming Wire Speed'])-1)*100

    return df


def starch_application(df:pd.DataFrame)->pd.DataFrame:
    #Drop Produzierte Quadratmeter
    df['Produzierte Quadratmeter'] = df['Aktuelle Arbeitsbreite in cm~^0'] / 100 * df['AD7 speed bottom'] * 60

    #Drop Auftrag Stärkeslurry FW
    df['Auftrag Stärkeslurry FW'] = df['Dilution water working tank 1'] + df['Flow starch main line to working tank 1~^0']

    #Drop Auftrag Stärkeslurry BW
    df['Auftrag Stärkeslurry BW'] = df['Dilution water working tank 2'] + df['Flow starch main line to working tank 2~^0']

    #Starch application FW in ml
    df['Starch application FW in ml'] = (df['Auftrag Stärkeslurry FW']*1000*1000)/df['Produzierte Quadratmeter']

    #Starch application BW in ml
    df['Starch application BW in ml'] = (df['Auftrag Stärkeslurry BW']*1000*1000)/df['Produzierte Quadratmeter']

    return df


def slurry(df:pd.DataFrame)->pd.DataFrame:
    #Slurry 1 to reactor
    df.loc[(df['Flow slurry 1 to reactor']>50),  'Slurry 1 pumping to reactor'] = 1
    df.loc[(df['Slurry 1 pumping to reactor']!=1),  'Slurry 1 pumping to reactor'] = 0

    #Slurry 2 to reactor
    df.loc[(df['Flow slurry 2 to reactor']>50),  'Slurry 2 pumping to reactor'] = 1
    df.loc[(df['Slurry 2 pumping to reactor']!=1),  'Slurry 2 pumping to reactor'] = 0

    #Slurry 3 to reactor
    df.loc[(df['Flow slurry 3 to reactor']>50),  'Slurry 3 pumping to reactor'] = 1
    df.loc[(df['Slurry 3 pumping to reactor']!=1),  'Slurry 3 pumping to reactor'] = 0

    return df


def dewatering(df:pd.DataFrame)->pd.DataFrame:
    #Total Dewatering Pick-Up
    df['Total Dewatering Pick-Up'] = df['Dewatering Pick-Up'] + df['Dewatering Suction Press Roll']

    #Total Dewatering Bottom Felt
    df['Total Dewatering Bottom Felt'] = df['Dewatering First Press Roll'] + df['Uhle box 1 flow [ l/min]']

    #Ratio of Dewatering of Pick UP of Total Dewatering Press
    # df['Ratio of Dewatering of Pick UP of Total Dewatering Press'] = df['Total Dewatering Pick-Up'] / df['Total Dewatering Press'] * 100 

    #Total Dewatering Top Felt
    # df['Total Dewatering Top Felt'] = df['Dewatering Shoe press'] / df['Total Dewatering Press']*100

    #Ratio of Dewatering of Bottom Felt of Total Dewatering
    # df['Ratio of Dewatering of Bottom Felt of Total Dewatering'] = df['Total Dewatering Bottom Felt'] / df['Total Dewatering Press']*100

    #Ratio of Double Felt Press on Total Dewatering
    # df['Ratio of Double Felt Press on Total Dewatering'] = df['Ratio of Dewatering of Bottom Felt of Total Dewatering'] + df['Ratio of Dewatering of Pick UP of Total Dewatering Press']

    #Total Felt Dewatering Press Two
    # df['Total Felt Dewatering Press Two'] = df['Total Dewatering Press']/df['Production Rate [T/h]']

    return df


def multifractionator(df:pd.DataFrame)->pd.DataFrame:
    #Multifractor 1 Long fibre fraction
    df['Multifractor 1 Long fibre fraction'] = (df['Multifractor 1 long fibre flow']/df['Multifractor 1 short fibre flow'])*100

    #Multifractor 2 Long fibre fraction
    df['Multifractor 2 long fibre fraction'] = (df['Multifractor 2 long fibre flow']/df['Multifractor 2 short fibre flow'])*100

    #Multifractor 3 Long fibre fraction
    df['Multifractor 3 long fibre fraction'] = (df['Multifractor 3 long fibre flow']/df['Multifractor 3 short fibre flow'])*100

    return df


def steam(df:pd.DataFrame)->pd.DataFrame:
    df['Steam flow from power plant to PM'] = df['Steam flow to PM'] - df['Steam flow to white water heating'] - df['Steam flow to steam box'] - df['Steam flow to hall heating'] - df['Steam flow to heat exchangers']

    #Steam flow to PreDryers
    df['Steam flow to PreDryers'] = df['Steam flow from power plant to PM'] - df['Steam flow to AfterDryers'] + df["Waste steam flow"]

    return df


def flows(df:pd.DataFrame)->pd.DataFrame:
    #Retention Aid [g/T]
    df['Retention Aid mass flow [g/T]'] = (df['Retentionsmittel Menge l/h [0..9000 l/h] ']*0.004)/df['Production Rate [T/h]']*1000

    #Bentonite 1 [g/T]
    df['Bentonite 1 mass flow [g/T]'] = (df['Bentonitmenge z. Mischrohr [-10..9000 l/h]']*0.025)/df['Production Rate [T/h]']*1000

    #Bentonite 2 [g/T]
    df['Bentonite 2 mass flow [g/T]'] = (df['Bentonitmenge z. Siebwasserturm [-10..9000 l/h]']*0.025)/df['Production Rate [T/h]']*1000

    #Fixative 1 [g/T]
    # df['Fixative 1 mass flow [g/T]'] = df['Sollwert Fixiermittelmenge [-1..50 l/h]'] / df['Production Rate [T/h]']*1000

    #Delete Fixiermittelmenge check
    df.loc[df['Fixiermittelmenge Trockenausschuss [0..100 l/h] ']>5, 'Fixiermittelmenge check'] = 1
    df.loc[df['Fixiermittelmenge check']!=1, 'Fixiermittelmenge check'] = 0

    #Fixative 2 [g/T]
    df['Fixative 2 mass flow [g/T]'] = df['Fixiermittelmenge Trockenausschuss [0..100 l/h] '] / df['Production Rate [T/h]']*1000 * df['Fixiermittelmenge check']

    #Deaerator [g/T]
    df['Act Deaerator mass flow [g/T]'] = df['Mengenregelung Entlüfter : MESSWERT      [0..40 l/h]'] / df['Production Rate [T/h]']*1000

    #Paper break status check
    df.loc[df['Paper break status li:PM1-OPC-DATA.F:rtup_pls (0 = false)'] >= 0 , 'Paper break status check'] = 1
    df.loc[df['Paper break status check']!=1 , 'Paper break status check'] = 0

    #Act Production Rate Gross
    df['Act Production Rate Gross'] = df['Production Rate [T/h]'] * df['Paper break status check']

    # #Starch [kg/T] (Previous Costimier Formula)
    # df['Starch mass flow [kg/T]'] = (df['Starch uptake by paper Top Roll [g/m2]']+df['Starch uptake by paper Bottom Roll [g/m2]'])/0.88*7.55*df['Speed']*60/1000/df['Act Production Rate Gross']
    
    #Starch [kg/T] (Updated OT Platform)
    df['Starch mass flow [kg/T]'] = (((((df['Flow starch main line to working tank 2~^0']*0.215)+(df['Flow starch main line to working tank 1~^0']*0.215)) * 1.085) + (271/(24*30)))/df['Production Rate [T/h]'])*1000
   
    #Defoamer [g/T]
    df['Defoamer mass flow [g/T]'] = (df['Entschäumermenge BW [-2..60 l/h]']+df['Entschäumermenge FW [-1..60 l/h]'])/df['Production Rate [T/h]']*1000

    #Leimungsmenge gesamt pro Tonne
    df['Leimungsmenge gesamt pro Tonne'] = (df['Mengenistwert Leimungsmittel BW : MESSWERT     [0..500 l/h]']+df['Menge Leimungsmittel : MESSWERT      [0..500 l/h]'])/df['Production Rate [T/h]']*1000

    #Leimungsmenge gesamt pro Tonne check
    df.loc[df['Leimungsmenge gesamt pro Tonne'] >= 2000 , 'Leimungsmenge gesamt pro Tonne check'] = 1
    df.loc[df['Leimungsmenge gesamt pro Tonne check']!=1 , 'Leimungsmenge gesamt pro Tonne check'] = 0

    #Sizing Agent [g/T]
    df['Sizing Agent [g/T]'] = df['Leimungsmenge gesamt pro Tonne'] * df['Leimungsmenge gesamt pro Tonne check']

    #Trockenverfestigermenge gesamt
    df['Trockenverfestigermenge gesamt'] = df['Trockenverfestigermenge bewegliche Walze [0..2500 l/h]'] + df['Trockenverfestigermenge feste Walze [0..2500 l/h]']

    #Trockenverfestigermenge gesamt check
    df.loc[df['Trockenverfestigermenge gesamt'] >= 200 , 'Trockenverfestigermenge gesamt check'] = 1
    df.loc[df['Trockenverfestigermenge gesamt check']!=1 , 'Trockenverfestigermenge gesamt check'] = 0

    #Dry Strength Agent [kg/T]
    df['Dry Strength Agent mass flow [kg/T]'] = df['Trockenverfestigermenge gesamt']/df['Production Rate [T/h]']*df['Trockenverfestigermenge gesamt check']*1.14

    #Biozid - ClO2 [kg/T]

    # df['Biozid - ClO2 mass flow [g/T]'] = df['Purate Laufrückmeldung : MESSWERT      [0..1 ]'] * 1000

    #Natriumhydroxide [kg/T]
    df['Natriumhydroxide mass flow [g/T]'] = (df['SP1: 1-000-FC028_01    NaOH-Dosierung Pulper :  MESSWERT']+df['SP1: 1-000-FC028_02    NaOH-Dosierung Kurzfaser-SF :  MESSWE']+df['SP1: 1-000-FC028_03    NaOH-Dosierung Langfaser-SF :  MESSWE'])/df['Production Rate [T/h]']*1000

    # #CO2 [g/T]
    df['CO2 mass flow [g/T]'] = (df['CO2-Dosierung Deculator 1 [0..60 kg/h]'] + df['CO2-Dosierung Deculator 2 [0..60 kg/h]'] + df['CO2-Dosierung Bio-Rückwasser [0..60 kg/h]'])/df['Production Rate [T/h]']*1000

    #===== Previous Costimiser Caculations

    #Delete Summe Energieverbrauch APA in kW ( Code used in the OT Electrical Calculations)
    df['Summe Energieverbrauch APA in kW'] = (df['APA 6KV [-1..10 MW]'] + df['APA 660V [-1..10 MW]'] + df['APA 400V [-1..10 MW]'] + df['APA 660V [-1..10 MW]~^0'])*1000

    #Delete Spezifischer Energieverbrauch APA (Code used in the OT Electrical Calculations)
    df['Spezifischer Energieverbrauch APA '] = df['Summe Energieverbrauch APA in kW']/df['Production Rate [T/h]']

    # #Delete Summe Energieverbrauch PM Hauptantriebe in kW
    # df['Summe Energieverbrauch PM Hauptantriebe in kW'] = (df['Mehrmotorenantrieb Presse [-1..10 MW]'] + df['Mehrmotorenantrieb Trockenpartie [-1..10 MW]'] + df['Mehrmotorenantrieb Former [-1..10 MW]'])*1000

    # #Delete Spezifischer Energieverbrauch PM Antriebe
    # df['Spezifischer Energieverbrauch PM Antriebe'] = df['Summe Energieverbrauch PM Hauptantriebe in kW'] / df['Production Rate [T/h]']

    # #Delete Summe PM Hauptaggregate in kW
    # df['Summe PM Hauptaggregate in kW'] = (df['PM 660V Festantriebe Verteiler3 [-1..10 MW]~^0'] + df['PM 660V Festantriebe Verteiler3 [-1..10 MW]'] + df['PM 660V Festantriebe Verteiler2 [-1..10 MW]'] + df['PM 660V Festantriebe Verteiler2 [-1..10 MW]~^1'] + df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^0'] + df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^1'])*1000

    # #Delete Spezifischer Energieverbrauch PM Hauptaggregate
    # df['Spezifischer Energieverbrauch PM Hauptaggregate'] = df['Summe PM Hauptaggregate in kW'] / df['Production Rate [T/h]']

    # #Delete Energieverbrauch RSM1 in kW
    # df['Energieverbrauch RSM1 in kW'] = df['VariSTEP div. Hauben+Lueftung Ventilatoren']*1000

    # #Delete Spezifischer Energieverbrauch RSM1
    # df['Spezifischer Energieverbrauch RSM1'] = df['Energieverbrauch RSM1 in kW'] / df['Production Rate [T/h]']

    # #Delete Energieverbrauch Bio in kW
    # df['Energieverbrauch Bio in kW'] = df['PM 400V [-1..10 MW]'] * 1000

    # #Delete Spezifischer Energieverbrauch Bio
    # df['Spezifischer Energieverbrauch Bio'] = df['Energieverbrauch Bio in kW'] / df['Production Rate [T/h]']

    # #Delete Energieverbrauch Logistik/Verwaltung in kW
    # df['Energieverbrauch Logistik/Verwaltung in kW'] = df['PM 400V [-1..10 MW]'] * 1000

    # #Delete Spezifischer Energieverbrauch Logistik/Verwaltung
    # df['Spezifischer Energieverbrauch Logistik/Verwaltung'] = df['Energieverbrauch Logistik/Verwaltung in kW'] / df['Production Rate [T/h]']

    # #Delete Energieverbrauch RSM2 in kW
    # df['Energieverbrauch RSM2 in kW'] = (df['Varisprint 660V [-10..10 MW]'] + df['Varisprint 400V [-10..10 MW]'])*1000

    # #Delete Spezifischer Energieverbrauch RSM2
    # df['Spezifischer Energieverbrauch RSM2'] = df['Energieverbrauch RSM2 in kW']/ df['Production Rate [T/h]']

    # #Electricity [kWh/T]
    # df['Electricity [kWh/T]'] = df['Spezifischer Energieverbrauch APA '] + df['Spezifischer Energieverbrauch PM Antriebe'] + df['Spezifischer Energieverbrauch PM Hauptaggregate'] + df['Spezifischer Energieverbrauch RSM1'] + df['Spezifischer Energieverbrauch Bio'] + df['Spezifischer Energieverbrauch Logistik/Verwaltung'] + df['Spezifischer Energieverbrauch RSM2']
    
    #======

    #===== OT Electrical Calulations
    # add Electrical consumption Wire Section in MW (OT Platform)
    df['Electrical Consumtion Wire Section'] = (df['Mehrmotorenantrieb Former [-1..10 MW]'])*1000/df['Production Rate [T/h]']

    # add Press and Steambox in MW (OT Platform)
    df['Press and Steambox'] = (df['Mehrmotorenantrieb Presse [-1..10 MW]'])*1000/df['Production Rate [T/h]']

    # add Winders + PM Freq Converters 1 to 3 in MW (OT Platform)
    df['PM Freq Converters 1 to 3'] = (df['VariSTEP div. Hauben+Lueftung Ventilatoren'] + df['Varisprint 660V [-10..10 MW]'] + df['Varisprint 400V [-10..10 MW]'])*1000/df['Production Rate [T/h]']

    # add Other PM 1 to 10 in kW (OT Platform)
    df['Other PM 1 to 10']= (df['Mehrmotorenantrieb Trockenpartie [-1..10 MW]'] + df['PM 400V [-1..10 MW]'] + df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^1'] 
                             + df['PM 660V Festantriebe Verteiler1 [-1..10 MW]~^0'] + df['PM 660V Festantriebe Verteiler2 [-1..10 MW]'] 
                             + df['PM 660V Festantriebe Verteiler2 [-1..10 MW]~^1'] + df['PM 660V Festantriebe Verteiler3 [-1..10 MW]~^0']+ df['PM 660V Festantriebe Verteiler3 [-1..10 MW]']
                             + df['MCC Einzel-FU 660V [-10..10 MW]~^0'] + df['MCC Einzel-FU 660V [-10..10 MW]~^1'])*1000/df['Production Rate [T/h]']

    # add Predryers sensor 10 to 25 in kW (OT Platform)
    df['Predryers sensor 10 to 25'] = (df ['Last Antrieb 10 1.TG [0..100 kW]'] + df['Last Antrieb 11 1.TG [0..100 kW]'] + df['Last Antrieb 12 2.TG [0..100 kW]']
                                       + df['Last Antrieb 13 2.TG [0..100 kW]'] + df['Last Antrieb 14 2.TG [0..100 kW]'] + df['Last Antrieb 15 3.TG [0..100 kW]'] 
                                       + df['Last Antrieb 16 3.TG [0..100 kW]'] + df['Last Antrieb 17 3.TG [0..100 kW]'] + df['Last Antrieb 18 4.TG [0..100 kW]']
                                       + df['Last Antrieb 19 4.TG [0..100 kW]'] + df['Last Antrieb 20 4.TG [0..100 kW]'] + df['Last Antrieb 21 4.TG [0..100 kW]']
                                       + df['Last Antrieb 22 5.TG [0..100 kW]'] + df['Last Antrieb 23 5.TG [0..100 kW]'] + df['Last Antrieb 24 5.TG [0..100 kW]']
                                       + df['Last Antrieb 25 5.TG [0..100 kW]']
                                       )/df['Production Rate [T/h]']

    # add Afterdryers sensor 1 to 9 in kW (OT Platform)
    df['Afterdryers sensor 1 to 9'] = (df['Last Antrieb 31 6.TG [0..100 kW]'] + df['Last Antrieb 32 6.TG [0..100 kW]'] + df['Last Antrieb 33 6.TG [0..100 kW]']
                                       + df['Last Antrieb 34 6.TG [0..100 kW]'] + df['Last Antrieb 35 6.TG [0..100 kW]'] + df['Last Antrieb 36 7.TG [0..200 kW]'] 
                                       + df['Last Antrieb 37 7.TG [0..200 kW]'] + df['Last Antrieb 38 7.TG [0..200 kW]'] + df['Last Antrieb 39 7.TG [0..200 kW]']
                                       )/df['Production Rate [T/h]']
    
    # Electricity [kWh/T] (Updated OT Platform)
    df['Electricity [kWh/T]'] = (df['Electrical Consumtion Wire Section'] + df['Press and Steambox'] + df['PM Freq Converters 1 to 3'] 
                                 + df['Other PM 1 to 10'] +  + df['Predryers sensor 10 to 25'] + df['Afterdryers sensor 1 to 9']
                                 + df['Spezifischer Energieverbrauch APA '])    

    #=======    
    
    # #Steam [kWh/T] (Previous Costimier Formula)
    # df['Steam [kWh/T]'] = (df['Steam energy from power plant to paper plant']-df['Condensate energy from paper plant to power plant'])/df['Production Rate [T/h]'] * 1000
    
    #Steam [kWh/T] (Updated OT Platform)
    
    cond = np.where(
        (df['Condensate energy from paper plant to power plant'] < 0) |
        (df['Condensate energy from paper plant to power plant'] > 10),
        df['Condensate energy from paper plant to power plant'].median(),
        df['Condensate energy from paper plant to power plant']
    )
    
    df['Steam [kWh/T]'] = (
        (
            (
                ((df['Steam flow to PM'] + df["Waste steam flow"]) * 0.788) - cond
            ) * 1.02 - (0.5938 / 24)
        )
        / np.where(df['Production Rate [T/h]'] > 45, df['Production Rate [T/h]'], 45)
    ) * 1000

    #df['Steam [kWh/T]'] = (((((df['Steam flow to PM'] + df["Waste steam flow"]) *0.788)-df['Condensate energy from paper plant to power plant'])*1.02 - (0.5938/24))/np.where(df['Production Rate [T/h]']>45,df['Production Rate [T/h]'],45))* 1000
   
    # #Fibre usage [kg/T] (Previous Costimier Formula)
    
    # df['Fibre usage [T/T]'] = df['Thick Stock Consistency [%]'] / 100 * df['Thick Stock Flow [l/min]'] * 60 / 1000 / df['Production Rate [T/h]'] # used for old fibre consumption

    #====== OT Fibre Calcualtions
    
    # Fibre Calculation in OT Platform
    df['MC_SF_LF_Demand'] = ((df['Short fibre flow'] * (df['Short fibre B06 consistency']/100))+(df['Long fibre flow']*(df['Long fibre consistency B07']/100)))*(60/1000)

    df['Whitewater_solids_flow'] = ((df['Short fibre flow']+df['Long fibre flow'])*(df['Consistency white water']/1000))*(60/1000)

    df['Approach_flow_returns'] = (df["Inlet Thickerner 2 [m3/h]"]*(1.74/100))

    #Fibre usage [T/T] (Updated OT Platform)
    df['Fibre usage [T/T]'] = ((((df["MC_SF_LF_Demand"] - df["Whitewater_solids_flow"] - df["Approach_flow_returns"])*1.5688)-23.1)*1.1)/df['Production Rate [T/h]'] 

    #======

    return df


def SQM_calc(df):
    df_sqm=df.copy()
    #Square meter cost
    df_sqm['Total SQM cost [€/hm2]'] = df_sqm['Production Rate [T/h]'] * df_sqm['Combined cost [€/T]']*df_sqm['Current basis weight']/(1-df_sqm['AB Grade ID'].astype(str).str.replace(".0","").str[-3:].astype(int))
    return df_sqm['Total SQM cost [€/hm2]']


def cost(df:pd.DataFrame)->pd.DataFrame:
    #Retention Aid [€/T]
    df['Retention Aid [€/T]'] = df['Retention Aid mass flow [g/T]'] * 4.08 / 1000

    #Bentonite 1 [€/T] 
    df['Bentonite 1 [€/T]'] = df['Bentonite 1 mass flow [g/T]'] * 0.28 / 1000

    #Bentonite 2 [€/T] 
    df['Bentonite 2 [€/T]'] = df['Bentonite 2 mass flow [g/T]'] * 0.28 / 1000

    # #Fixative 1 [€/T]
    # df['Fixative 1 [€/T]'] = df['Fixative 1 mass flow [g/T]'] * 3.62 / 1000

    #Fixative 2 [€/T]
    df['Fixative 2 [€/T]'] = df['Fixative 2 mass flow [g/T]'] * 3.58 / 1000

    #Deaerator [€/T]
    df['Deaerator [€/T]'] = df['Act Deaerator mass flow [g/T]'] * 1.53 / 1000

    #Starch [€/T]
    df['Starch [€/T]'] = df['Starch mass flow [kg/T]'] * 434.22 / 1000

    #Defoamer [g/T]
    df['Defoamer [€/T]'] = df['Defoamer mass flow [g/T]'] * 4.22 / 1000

    #Sizing Agent [€/T]
    df['Sizing Agent [€/T]'] = df['Sizing Agent [g/T]'] * 0.84 / 1000

    #Dry Strength Agent [€/T]
    df['Dry Strength Agent [€/T]'] = df['Dry Strength Agent mass flow [kg/T]'] * 0.30 

    # #Biozid - ClO2 [€/T]
    # df['Biozid - ClO2 [€/T]'] = df['Biozid - ClO2 mass flow [g/T]'] * 1.83 / 1000 #to confirm

    #Natriumhydroxide [€/T]
    df['Natriumhydroxide [€/T]'] = df['Natriumhydroxide mass flow [g/T]'] * 0.30 / 1000

    # #CO2 [€/T]
    # df['CO2 [€/T]'] = df['CO2 mass flow [g/T]'] * 0.32 / 1000

    # #Electricity [€/T]
    df['Electricity [€/T]'] = df['Electricity [kWh/T]'] * 113.66 / 1000

    #Steam [€/T]
    df['Steam [€/T]'] = df['Steam [kWh/T]'] * 89.03 / 1000

    #Fibre usage [€/T]
    df['Fibre cost [€/T]'] = df['Fibre usage [T/T]'] * 146.46

    #Combined cost [€/T]
    # df['Combined cost [€/T]'] = df['Retention Aid [€/T]'] + df['Bentonite 1 [€/T]'] + df['Bentonite 2 [€/T]'] + df['Fixative 1 [€/T]'] + df['Fixative 2 [€/T]'] + df['Deaerator [€/T]'] + df['Starch [€/T]'] + df['Defoamer [€/T]'] + df['Sizing Agent [€/T]'] + df['Dry Strength Agent [€/T]'] + df['Biozid - ClO2 [€/T]'] + df['Natriumhydroxide [€/T]'] + df['CO2 [€/T]'] + df['Electricity [€/T]'] + df['Steam [€/T]'] + df['Fibre usage [€/T]']
    df['Combined cost [€/T]'] = df['Retention Aid [€/T]'] + df['Bentonite 1 [€/T]'] + df['Bentonite 2 [€/T]']  + df['Fixative 2 [€/T]'] + df['Deaerator [€/T]'] + df['Starch [€/T]'] + df['Defoamer [€/T]'] + df['Sizing Agent [€/T]'] + df['Dry Strength Agent [€/T]'] +  df['Natriumhydroxide [€/T]'] + df['Electricity [€/T]'] + df['Steam [€/T]'] + df['Fibre cost [€/T]']
    
    df['Total SQM cost [€/hm2]'] = SQM_calc(df)

    return df


def summarize_dataframe(df: pd.DataFrame, name: str = "Dataframe"):
    numeric_df = df.select_dtypes(include=[np.number])  # Only consider numeric columns
    # Calculate percentage of 99999 and -99999 values
    extreme_values = ((numeric_df == 999999) | (numeric_df == -999999)).mean().mean() * 100 if not numeric_df.empty else 0
    summary = {
        "Name": name,
        "Shape": df.shape,
        "Missing Values (%)": df.isna().mean().mean() * 100,
        "Infinite Values (%)": np.isinf(numeric_df).mean().mean() * 100 if not numeric_df.empty else 0,
        "Substituted infinite values": extreme_values,
        "Duplicate Rows": df.duplicated().sum()
    }
    return summary


def compare_dataframes(before: pd.DataFrame, after: pd.DataFrame, name: str):
    before_summary = summarize_dataframe(before, f"Before {name}")
    after_summary = summarize_dataframe(after, f"After {name}")
    comparison_df = pd.DataFrame([before_summary, after_summary])

    print(comparison_df)


def combined_calculations(df:pd.DataFrame)->pd.DataFrame:
    df_initial = df.copy()
    df = paper_break(df)
    df = draws(df)
    df = starch_application(df)
    df = slurry(df)
    df = dewatering(df)
    df = multifractionator(df)
    df = steam(df)
    df = flows(df)
    df = cost(df)
    compare_dataframes(df_initial, df, "Combined calculations")
    df = df.replace([np.inf], 999999)
    df = df.replace([-np.inf], -999999)

    return df


def alligning_dataframe(df: pd.DataFrame, df_gloss) -> pd.DataFrame:

    ######## Drop this part when all tags will be available ###########
    # drop_list = ['Contact pressure support drum OS', 'Contact pressure support drum DS', 'Fixative 1 [€/T]', 'Biozid - ClO2 [€/T]', 'CO2 [€/T]', 'Electricity [€/T]', 'Fixative 1 mass flow [g/T]', 'Biozid - ClO2 mass flow [g/T]', 'Electricity [kWh/T]', 'Rodsize BottomRoll', 'Rod Size TopRoll', 'Power Vacuum pump press section 305', 'Power Vacuum pump 308 (reserve)', 'Power Vacuum pump press section 306', 'Ratio of Dewatering of Pick UP of Total Dewatering Press', 'Total Dewatering Top Felt', 'Ratio of Dewatering of Bottom Felt of Total Dewatering', 'Ratio of Double Felt Press on Total Dewatering', 'Total Felt Dewatering Press Two', 'Total Dewatering Press', 'Condensate cloudidity']
    drop_list = ['Contact pressure support drum OS', 'Contact pressure support drum DS', 'Fixative 1 [€/T]', 'Biozid - ClO2 [€/T]', 'CO2 [€/T]','Fixative 1 mass flow [g/T]', 'Biozid - ClO2 mass flow [g/T]', 'Rodsize BottomRoll', 'Rod Size TopRoll', 'Power Vacuum pump press section 305', 'Power Vacuum pump 308 (reserve)', 'Power Vacuum pump press section 306', 'Ratio of Dewatering of Pick UP of Total Dewatering Press', 'Total Dewatering Top Felt', 'Ratio of Dewatering of Bottom Felt of Total Dewatering', 'Ratio of Double Felt Press on Total Dewatering', 'Total Felt Dewatering Press Two', 'Total Dewatering Press', 'Condensate cloudidity']
    # Create a mask for rows to keep
    mask = ~df_gloss['Feature'].isin(drop_list)

    # Drop rows where the mask is False
    df_gloss = df_gloss[mask]

    ##################################

    df = df[df_gloss['Feature']]

    return df


def clean_column_names(df:pd.DataFrame) -> pd.DataFrame:
    regex = re.compile(r"[\[\]\<\>]")  # Regex pattern to match '[', ']', '<', '>'
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
    df.columns = [col.replace(" ", "_") if " " in col else col for col in df.columns]
    return df


def ingestion_pipeline_API_alternative(df:pd.DataFrame, df_tags, df_gloss) -> pd.DataFrame:
    df["R2NAE20CF905"]=df["R2NAE20CF905"].fillna(0)
    df_initial = df.copy()
    print("Step 1: Replace tags with corresponding names in PHD \n")
    df = replace_tags_with_name(df, df_tags)
    df = df.copy()
    print("Step 2: Perform calculations")
    df = combined_calculations(df)
    print("Step 3: Align dataframe with features from data glossary \n")
    df = alligning_dataframe(df, df_gloss)
    print("Step 4: Clean name of the columns \n")
    df = clean_column_names(df)
    print("Before and after of ingestion pipeline: \n")
    compare_dataframes(df_initial, df, "Ingestion Pipeline")
    return df