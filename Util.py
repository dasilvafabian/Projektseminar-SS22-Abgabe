import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from datetime import datetime 


"""
Diese Datei enthält die nötigen Funktionen zum Einlesen der Daten sowie Erstellung und
Testung der Prognosemodelle. Es werden die folgenden Dateien benötigt:
 - Unfallorte 2016 bis einschl. 2020
 - bevölkerung.csv
 - zulassungszahlen.csv
 - für den Testmodus: daten.csv
 
Der Parameter Testmodus (bool) ermöglicht es die Schritte, welche die Daten einlesen 
und vorbereiten zu überspringen und hat eine entsprechend kürzere Laufzeit zur Folge.
"""

Testmodus=False


def get_unfalldaten():  
    """Die Funktion liest die Unfalldaten ein, fügt sie in einem Dataframe zusammen und löscht nicht relevante Attribute.
    
    Parameter:
        -
    
    Returns:
        unfaelle_alles (DataFrame): Ein DataFrame mit den Unfalldaten für die Jahre 2016 bis 2020.
    """
    
    unfaelle_16 = pd.read_csv("Unfallorte_2016_LinRef.txt", delimiter=";")
    unfaelle_17 = pd.read_csv("Unfallorte2017_LinRef.txt", delimiter=";").reset_index(drop=True)
    unfaelle_18 = pd.read_csv("Unfallorte2018_LinRef.txt", delimiter=";").reset_index(drop=True)
    unfaelle_19 = pd.read_csv("Unfallorte2019_LinRef.txt", delimiter=";").reset_index(drop=True)
    unfaelle_20 = pd.read_csv("Unfallorte2020_LinRef.csv", delimiter=";").reset_index(drop=True)
    # Spaltennamen anpassen
    unfaelle_17.rename(columns = {'LICHT':'ULICHTVERH'}, inplace = True)
    unfaelle_16.rename(columns = {'IstStrasse':'STRZUSTAND'}, inplace = True)
    unfaelle_19.rename(columns = {'IstSonstige':'IstSonstig'}, inplace = True)
    unfaelle_20.rename(columns = {'IstSonstige':'IstSonstig'}, inplace = True)
    unfaelle_alle=pd.concat([unfaelle_16, unfaelle_17, unfaelle_18, unfaelle_19, unfaelle_20], axis=0).reset_index().fillna(0)
    # Khfz auf sonstige addieren
    unfaelle_alle["IstSonstig"] += unfaelle_alle["IstGkfz"]
    unfaelle_alle["IstSonstig"].replace(2,1)
    # Nicht benötigte Attribute löschen
    unfaelle_alle = unfaelle_alle.drop(["UIDENTSTLA","UIDENTSTLAE","IstGkfz","OBJECTID_1","FID","OBJECTID","XGCSWGS84","YGCSWGS84"],axis=1)
    # AGS erstellen
    unfaelle_alle["ULAND"] = unfaelle_alle["ULAND"].astype("string").str.zfill(2)
    unfaelle_alle["UKREIS"] = unfaelle_alle["UKREIS"].astype("string").str.zfill(2)
    unfaelle_alle["UGEMEINDE"] = unfaelle_alle["UGEMEINDE"].astype("string").str.zfill(3)
    unfaelle_alle["AGS"] = unfaelle_alle["ULAND"].map(str)+unfaelle_alle["UREGBEZ"].map(str)+unfaelle_alle["UKREIS"].map(str)+unfaelle_alle["UGEMEINDE"].map(str)
    unfaelle_alle = unfaelle_alle.drop(["UREGBEZ","UKREIS","UGEMEINDE"],axis=1)
    return(unfaelle_alle)


def get_regionalklassen():
    """Die Funktion erstellt ein Dataframe, welches die Regionalklassen und die Anzahl der zugelassenen Fahrzeuge
    je Zulassungsbezirk für das Bundesland Sachsen.
    
    Parameter:
        -
        
    Returns:
        regionalklassen_sn (DataFrame): Ein DataFrame mit den Regionalklassen (Stand Mai 2022) für das Bundesland Sachsen.
    """
    
    RKHaftpflicht = [2,6,10,2,1,2,9,2,1,2,3,4,3]
    RKVollkasko = [4,5,4,4,6,5,7,3,4,4,5,4,4]
    RKTeilkasko = [9,8,6,9,11,10,8,8,9,10,8,7,8]
    regionalklassen_sn = pd.DataFrame(list(zip(RKHaftpflicht, RKVollkasko, RKTeilkasko)), columns = ["RKHaftpflicht","RKVollkasko","RKTeilkasko"])
    return(regionalklassen_sn)


def get_bevölkerung_regionaklassen():
    """Die Fuktion liest die .csv-Datei mit den Bevökerungsdaten für das Bundesland Sachsen ein und erzeugt 
    anhand der darin enthaltenen Zuordnungen der AGS zu den Zulassungsbezirken ein DataFrame, welches die 
    Bevölkerungsdaten und Regionalklassen enthält.
    
    Parameter: 
        -
        
    Returns:
        bev_regkl_sn (DatFrame): Ein DataFrame mit den Bevölkerungszahlen und Regionalklassen des Bundeslandes Sachsen.
    """
    
    bev_regkl_sn = pd.read_csv("bevölkerung.csv",delimiter=";")
    regionalklassen = get_regionalklassen()
    dictRKH=dict(regionalklassen["RKHaftpflicht"].reset_index(level=0).values)
    dictRKV=dict(regionalklassen["RKVollkasko"].reset_index(level=0).values)
    dictRKT=dict(regionalklassen["RKTeilkasko"].reset_index(level=0).values)
    bev_regkl_sn["RKHaftpflicht"] = bev_regkl_sn["Bezirk"].map(dictRKH)
    bev_regkl_sn["RKVollkasko"] = bev_regkl_sn["Bezirk"].map(dictRKV)
    bev_regkl_sn["RKTeilkasko"] = bev_regkl_sn["Bezirk"].map(dictRKT)
    bev_regkl_sn["Kreis"] = bev_regkl_sn["Kreis"].astype("int").astype("string").str.zfill(2)
    bev_regkl_sn["Gem"] = bev_regkl_sn["Gem"].astype("int").astype("string").str.zfill(3)
    bev_regkl_sn["AGS"] = bev_regkl_sn["Land"].astype("int").map(str)+bev_regkl_sn["RB"].astype("int").map(str)+bev_regkl_sn["Kreis"].map(str)+bev_regkl_sn["Gem"].map(str)
    bev_regkl_sn = bev_regkl_sn.drop(["Land","Bezeichnung"],axis=1)
    bev_regkl_sn["Kreis"] = bev_regkl_sn["Kreis"].astype("int")
    bev_regkl_sn["Gem"] = bev_regkl_sn["Gem"].astype("int")
    return(bev_regkl_sn)


def aggregate_daten(daten, bundesland):
    """Die Funktion aggregiert ein gegebenes DF mit Unfalldaten und gibt die Anzahl aller Unfälle für
    jede Kombination aus Jahr/Monat/AGS für ein Bundesland ein an.
    
    Parameter:
        daten (DataFrame): Ein DataFrame mit Unfalldaten.
        bundesland (int): Die Schlüsselstelle für ein deutsches Bundesland.
        
    Returns:
        unfaelle_agg (DatFrame): Ein DataFrame mit aggregierten Unfalldaten für ein Bundesland.
    """  
    
    # Alle Bundesländer außer dem gewünschten löschen
    daten_bundesland = daten.loc[(daten["ULAND"] == str(bundesland))]
    temp1 = daten_bundesland[["AGS","UJAHR","UMONAT","USTUNDE"]].groupby(["AGS","UJAHR","UMONAT"]).count().reset_index()
    temp1.rename(columns = {'USTUNDE':'UANZAHL'}, inplace = True)
    # Datenpunkte ohne Unfälle erzeugen
    temp2 = pd.DataFrame(list(product(pd.unique(temp1["AGS"]), range(2016,2021), range(1,13))), columns=["AGS","UJAHR","UMONAT"])
    temp2["UANZAHL"] = 0
    # In einem DataFrame kombinieren
    unfaelle_agg = pd.concat([temp2, temp1], axis=0).groupby(["AGS","UJAHR","UMONAT"]).sum().reset_index()
    return(unfaelle_agg)


def merge_daten():
    """Die Funktion erzeugt ein DataFrame mit den aggregierten Unfalldaten, Regionaklassen und Bevölkerungszahlen für das
    Bundesland Sachsen.
    
    Parameter:
        -
    Returns:
        merged_sn (DatFrame): Ein DataFrame mit den aggregierten Unfalldaten, Regionaklassen und Bevölkerungszahlen.
    """
    
    bev_regkl = get_bevölkerung_regionaklassen() 
    unfaelle_sn_agg = aggregate_daten(get_unfalldaten(), 14)
    merged_sn = unfaelle_sn_agg.merge(right=bev_regkl, how="left", left_on="AGS", right_on="AGS")
    return(merged_sn)
            

def add_zulassungszahlen():
    """Die Funktion fügt einem DataFrame mit aggregierten Unfallzahlen die Zulassungszahlen je Zulassungsbezirk und Jahr hinzu.
    
    Parameter:
        -
        
    Returns:
        daten_sn (DatFrame): Ein DataFrame mit Unfalldaten, Regionalklassen, Bevölkerungs- und Zulassungszahlen für Sachsen.
    """
    
    daten_sn = merge_daten()
    # Jahresaktuelle Zulassungszahlen per Dictionary mappen
    daten_sn["Bez+Jahr"] = daten_sn["Bezirk"].astype("int").map(str)+daten_sn["UJAHR"].map(str)
    zulassungsahlen = pd.read_csv("zulassungszahlen.csv",delimiter=";")
    zulassungsahlen = zulassungsahlen.set_index(zulassungsahlen["Bezirk"].map(str)+zulassungsahlen["Jahr"].map(str))
    dict_zulassungen = dict(zulassungsahlen["Absolut"])
    daten_sn["Anz_Fahrzeuge"] = daten_sn["Bez+Jahr"].map(dict_zulassungen)
    daten_sn = daten_sn.drop(["Bez+Jahr"],axis=1)
    return(daten_sn)


def naiverPredictor1(jahr):
    """
    Die Funktion prognostiziert die Anzahl an Unfällen für ein bestimmtes Jahr je mit dem Wert null.
    
    Parameter:
        jahr (int): Das Jahr, für welches die Prognose erstellt werden soll.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.        
    """
    
    if Testmodus:
        daten = pd.read_csv("daten.csv")
    else:
        daten = add_zulassungszahlen()
    y_test = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr)]
    daten["UANZAHL"] = 0
    y_naiv = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr)]
    f_one_score = f1_score(y_test, y_naiv, average='micro')
    mse = mean_squared_error(y_test, y_naiv)
    return(f_one_score, mse)


def naiverPredictor2(jahr):
    """
    Die Funktion prognostiziert die Anzahl an Unfällen für ein bestimmtes Jahr je mit dem Wert zwei.
    
    Parameter:
        jahr (int): Das Jahr, für welches die Prognose erstellt werden soll.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.        
    """
    
    if Testmodus:
        daten = pd.read_csv("daten.csv")
    else:
        daten = add_zulassungszahlen()
    y_test = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr)]
    daten["UANZAHL"] = 2
    y_naiv = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr)]
    f_one_score = f1_score(y_test, y_naiv, average='micro')
    mse = mean_squared_error(y_test, y_naiv)
    return(f_one_score, mse)


def naiverPredictor3(jahr):
    """
    Die Funktion prognostiziert die Anzahl an Unfällen für ein bestimmtes Jahr, in dem die
    jeweiligen Vorjahreswerte als Vorhersage ausgegeben werden.
    
    Parameter:
        jahr (int): Das Jahr, für welches die Prognose erstellt werden soll.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.        
    """
    
    if Testmodus:
        daten = pd.read_csv("daten.csv")
    else:
        daten = add_zulassungszahlen()
    y_test = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr)]
    y_naiv = daten["UANZAHL"].loc[(daten["UJAHR"] == jahr-1)]
    f_one_score = f1_score(y_test, y_naiv, average='micro')
    mse = mean_squared_error(y_test, y_naiv)
    return(f_one_score, mse)


def random_forest(without, year, oversampling, weighted):
    """Die Funktion erstellt mittels Random Forest eine Vorhersage für die Anzahl an Unfällen und gibt den
    MSE, F1-Score sowie Laufzeit dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.
        oversampling (bool): Falls True, wird der Trainingsdatensatz voher mittels Oversampling bearbeitet.
        weighted (bool): Falls True, wird anstatt des RF-Regressors der RF-Klassifizierer genutzt
            und mit gewichteten Klassen trainiert.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv").drop(without,axis=1)
    else:
        daten = add_zulassungszahlen()
    X_train = daten.drop("UANZAHL",axis=1).loc[(daten["UJAHR"] < year)]
    X_test = daten.drop("UANZAHL",axis=1).loc[(daten["UJAHR"] == year)]
    y_train = daten["UANZAHL"].loc[(daten["UJAHR"] < year)]
    y_test = daten["UANZAHL"].loc[(daten["UJAHR"] == year)]
    if weighted: 
        rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    elif oversampling: 
        rus = RandomOverSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        rf = RandomForestRegressor(random_state=42)
    else:
        rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train,y_train)
    y_pred_test = rf.predict(X_test).round()
    f_one_score = f1_score(y_test, y_pred_test, average='micro')
    mse = mean_squared_error(y_test, y_pred_test)
    runtime = time.time() - start
    return(f_one_score, mse, runtime)


def random_forest_splitting_1(without, year):
    """Die Funktion erstellt mittels des Modells Splitting 1 (2x Random Forest) eine Vorhersage für die Anzahl an Unfällen und gibt den
    MSE und F1-Score dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.     
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv").drop(without,axis=1)
    else:
        daten = add_zulassungszahlen()
    # Neues Attribut hinzufügen
    daten.loc[daten['UANZAHL'] > 0, 'ZV_neu'] = 1
    daten.loc[daten['UANZAHL'] == 0, 'ZV_neu'] = 0
    # Klassifikation
    train = daten.loc[(daten["UJAHR"] < year)]
    test = daten.loc[(daten["UJAHR"] == year)].reset_index(drop=True)
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    rf.fit(train.drop(["UANZAHL","ZV_neu"],axis=1),train["ZV_neu"])
    test["pred_1"] = pd.Series(rf.predict(test.drop(["UANZAHL","ZV_neu"],axis=1)))
    train["pred_1"] = pd.Series(rf.predict(train.drop(["UANZAHL","ZV_neu"],axis=1)))
    # Regression
    testset_2 = test.loc[(test["pred_1"] == 1)].reset_index(drop=True)
    trainset_2 = train.loc[(train["pred_1"] == 1)]
    nicht_unfälle = test.loc[(test["pred_1"] == 0)].reset_index(drop=True)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(trainset_2.drop(["UANZAHL","ZV_neu","pred_1"],axis=1),trainset_2["UANZAHL"])
    testset_2["pred_1"] = pd.Series(rf.predict(testset_2.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)).round())
    # Zusammenführen und evaluieren 
    ganzes_testset = pd.concat([nicht_unfälle,testset_2], ignore_index = True, sort = False)
    mse = mean_squared_error(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"])
    f_one_score = f1_score(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"], average='micro')
    runtime = time.time() - start
    return mse, f_one_score, runtime



def random_forest_splitting_2(without, year):
    """Die Funktion erstellt mittels des Modells Splitting 2 (3x Random Forest) eine Vorhersage für die Anzahl an Unfällen und gibt den
    MSE und F1-Score dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.     
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv").drop(without,axis=1)
    else:
        daten = add_zulassungszahlen()
    # Neues Attribut hinzufügen
    daten.loc[daten['UANZAHL'] > 4, 'ZV_neu'] = 1
    daten.loc[daten['UANZAHL'] <= 4, 'ZV_neu'] = 0   
    # Klassifikation
    train = daten.loc[(daten["UJAHR"] < year)]
    test = daten.loc[(daten["UJAHR"] == year)].reset_index(drop=True)
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    rf.fit(train.drop(["UANZAHL","ZV_neu"],axis=1),train["ZV_neu"])
    test["pred_1"] = pd.Series(rf.predict(test.drop(["UANZAHL","ZV_neu"],axis=1)))
    train["pred_1"] = pd.Series(rf.predict(train.drop(["UANZAHL","ZV_neu"],axis=1)))
    # Regression 1 für Anz. Unfälle größer 4
    testset_2A = test.loc[(test["pred_1"] == 1)].reset_index(drop=True)
    trainset_2A = train.loc[(train["pred_1"] == 1)]
    rf = RandomForestRegressor(random_state=42)
    rf.fit(trainset_2A.drop(["UANZAHL","ZV_neu","pred_1"],axis=1),trainset_2A["UANZAHL"])
    testset_2A["pred_1"] = pd.Series(rf.predict(testset_2A.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)).round())
    # Regression 2 für Anz. Unfälle kl. gl. 4
    testset_2B = test.loc[(test["pred_1"] == 0)].reset_index(drop=True)
    trainset_2B = train.loc[(train["pred_1"] == 0)]
    rf = RandomForestRegressor(random_state=42)
    rf.fit(trainset_2B.drop(["UANZAHL","ZV_neu","pred_1"],axis=1),trainset_2B["UANZAHL"])
    testset_2B["pred_1"] = pd.Series(rf.predict(testset_2B.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)).round())
    # Zusammenführen und evaluieren 
    ganzes_testset = pd.concat([testset_2A,testset_2B], ignore_index = True, sort = False)
    mse = mean_squared_error(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"])
    f_one_score = f1_score(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"], average='micro')
    runtime = time.time() - start
    return mse, f_one_score, runtime
    

def knn(without, year):
    """Die Funktion erstellt mittels eines KNN eine Vorhersage für die Anzahl an Unfällen und gibt den
    MSE, F1-Score sowie Laufzeit dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv") 
    else:
        daten = add_zulassungszahlen()
    new_dum = pd.get_dummies(daten['AGS'])
    KNN_daten = pd.concat([daten,new_dum], axis=1)
    KNN_daten = KNN_daten.sort_values(by=['UJAHR', 'UMONAT'])
    daten = daten.drop(without,axis=1)
    X_train = daten.drop("UANZAHL",axis=1).loc[(daten["UJAHR"] < year)]
    X_test = daten.drop("UANZAHL",axis=1).loc[(daten["UJAHR"] == year)]
    y_train = daten["UANZAHL"].loc[(daten["UJAHR"] < year)]
    y_test = daten["UANZAHL"].loc[(daten["UJAHR"] == year)]
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Layer und Neuronen definieren
    model = Sequential()
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(1, activation="relu"))
    model.compile(optimizer="adam",loss="mse")
    # Training
    model.fit(X_train,y_train,epochs=50, batch_size=5124)
    # Loss ausgeben
    loss = pd.DataFrame(model.history.history)
    plt.figure()
    loss.plot()
    # Vorhersage mit tatsächlichen Werten vergleichen
    test_predictions = model.predict(X_test).round()
    pred_df = pd.DataFrame(y_test)
    test_predcitions = pd.Series(test_predictions.reshape(5124,))
    pred_df = pred_df.reset_index()
    pred_df = pred_df.drop(columns=["index"])
    pred_df = pd.concat([pred_df, test_predcitions],axis=1)
    pred_df.columns = ["Test Y","Model Predictions"]
    sns.set(rc={"figure.figsize":(12, 12)})
    plt.figure()
    sns.scatterplot(data=pred_df, x="Test Y", y="Model Predictions")
    plt.figure()
    # Evaluierung
    mse = round(mean_squared_error(pred_df["Test Y"], pred_df["Model Predictions"]), 4)
    f_one_score = round(f1_score(pred_df["Test Y"], pred_df["Model Predictions"], average='micro'), 4)
    runtime = round(time.time() - start, 4)
    return mse, f_one_score, runtime


def knn_oversampling(without,year):
    """Die Funktion erstellt mittels eines KNN und Oversamplings eine Vorhersage für die Anzahl an Unfällen und gibt den
    MSE, F1-Score sowie Laufzeit dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """

    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv") 
    else:
        daten = add_zulassungszahlen()
    daten = daten.sort_values(by=['UJAHR', 'UMONAT'])
    X_train_sam = daten.drop(['UANZAHL'], axis=1).loc[(daten['UJAHR'] < year)]
    y_train_sam = daten['UANZAHL'].loc[(daten['UJAHR'] < year)]
    ros = RandomOverSampler(sampling_strategy="not majority") # String
    Xsam, ysam = ros.fit_resample(X_train_sam, y_train_sam)
    X_test_sam = daten.drop(['UANZAHL'], axis=1).loc[(daten['UJAHR'] == year)]
    y_test_sam = daten['UANZAHL'].loc[(daten['UJAHR'] == year)]
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(Xsam)
    X_train = scaler.transform(Xsam)
    X_test = scaler.transform(X_test_sam)
    # Layer und Neuronen definieren
    model = Sequential()
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(1, activation="relu"))
    model.compile(optimizer="adam",loss="mse")
    # Training
    model.fit(X_train,ysam,epochs=10)
    # Loss im Verlauf betrachten
    loss = pd.DataFrame(model.history.history)
    plt.figure()
    loss.plot()
    # Vorhersage mit tatsächlichen Werten vergleichen
    test_predictions = model.predict(X_test)
    pred_df = pd.DataFrame(y_test_sam)
    test_predcitions = pd.Series(test_predictions.reshape(5124,))
    pred_df = pred_df.reset_index()
    pred_df = pred_df.drop(columns=["index"])
    pred_df = pd.concat([pred_df, test_predcitions],axis=1)
    pred_df.columns = ["Test Y","Model Predictions"]
    sns.set(rc={"figure.figsize":(12, 12)})
    plt.figure()
    sns.scatterplot(data=pred_df, x="Test Y", y="Model Predictions")
    plt.figure()
    mse = mean_squared_error(pred_df["Test Y"], pred_df["Model Predictions"])
    f_one_score = f1_score(pred_df["Test Y"], pred_df["Model Predictions"], average='micro')
    runtime = time.time() - start
    return mse, f_one_score, runtime


def knn_splitting_1(without, year):
    """Die Funktion erstellt mittels des Modells Splitting 1(1x Random Forest, 1x KNN) eine Vorhersage 
    für die Anzahl an Unfällen und gibt den MSE und F1-Score dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.     
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv") 
    else:
        daten = add_zulassungszahlen()
    # Neues Attribut hinzufügen
    daten.loc[daten['UANZAHL'] > 0, 'ZV_neu'] = 1
    daten.loc[daten['UANZAHL'] == 0, 'ZV_neu'] = 0
    # Klassifikation
    train = daten.loc[(daten["UJAHR"] < year)]
    test = daten.loc[(daten["UJAHR"] == year)].reset_index(drop=True)
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    rf.fit(train.drop(["UANZAHL","ZV_neu"],axis=1),train["ZV_neu"])
    test["pred_1"] = pd.Series(rf.predict(test.drop(["UANZAHL","ZV_neu"],axis=1)))
    train["pred_1"] = pd.Series(rf.predict(train.drop(["UANZAHL","ZV_neu"],axis=1)))
    # Regression
    testset_2 = test.loc[(test["pred_1"] == 1)].reset_index(drop=True)
    trainset_2 = train.loc[(train["pred_1"] == 1)]
    nicht_unfälle = test.loc[(test["pred_1"] == 0)].reset_index(drop=True)
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(trainset_2.drop(['UANZAHL', 'ZV_neu', 'pred_1'], axis=1))
    rf = Sequential()
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(1, activation="relu"))
    rf.compile(optimizer="adam",loss="mse")
    rf.fit(scaler.transform(trainset_2.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)),trainset_2["UANZAHL"],epochs=100)
    pred_nn_1 = rf.predict(scaler.transform(testset_2.drop(["UANZAHL","ZV_neu","pred_1"],axis=1))).round()
    testset_2["pred_1"] = pd.Series(pred_nn_1.reshape((pred_nn_1.shape[0],)))
    # Zusammenführen und evaluieren 
    ganzes_testset = pd.concat([nicht_unfälle,testset_2], ignore_index = True, sort = False)
    mse = mean_squared_error(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"])
    f_one_score = f1_score(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"], average='micro')
    runtime = time.time() - start
    return mse, f_one_score, runtime
    

def knn_splitting_2(without, year):
    """Die Funktion erstellt mittels des Modells Splitting 2 (1x Random Forest, 2x KNN) eine Vorhersage 
    für die Anzahl an Unfällen und gibt den MSE und F1-Score dieser Schätzung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.     
        
    Returns:
        f_one_score (float): Der F1-Score der Schätzung.
        mse (float): Der MSE der Schätzung.    
        runtime (float): Laufzeit der Funktion.
    """
    
    start = time.time()
    if Testmodus:
        daten = pd.read_csv("daten.csv") 
    else:
        daten = add_zulassungszahlen()
    # Neues Attribut hinzufügen
    daten.loc[daten['UANZAHL'] > 4, 'ZV_neu'] = 1
    daten.loc[daten['UANZAHL'] <= 4, 'ZV_neu'] = 0
    # Klassifikation
    train = daten.loc[(daten["UJAHR"] < year)]
    test = daten.loc[(daten["UJAHR"] == year)].reset_index(drop=True)
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
    rf.fit(train.drop(["UANZAHL","ZV_neu"],axis=1),train["ZV_neu"])
    test["pred_1"] = pd.Series(rf.predict(test.drop(["UANZAHL","ZV_neu"],axis=1)))
    train["pred_1"] = pd.Series(rf.predict(train.drop(["UANZAHL","ZV_neu"],axis=1)))
    # Regression 1
    testset_2A = test.loc[(test["pred_1"] == 1)].reset_index(drop=True)
    trainset_2A = train.loc[(train["pred_1"] == 1)]
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(trainset_2A.drop(['UANZAHL', 'ZV_neu', 'pred_1'], axis=1))
    rf = Sequential()
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(350, activation="relu"))
    rf.add(Dense(1, activation="relu"))
    rf.compile(optimizer="adam",loss="mse")
    rf.fit(scaler.transform(trainset_2A.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)),trainset_2A["UANZAHL"],epochs=100)
    pred_nn_1A = rf.predict(scaler.transform(testset_2A.drop(["UANZAHL","ZV_neu","pred_1"],axis=1))).round()
    testset_2A["pred_1"] = pd.Series(pred_nn_1A.reshape((pred_nn_1A.shape[0],)))
    # Regression 2
    testset_2B = test.loc[(test["pred_1"] == 0)].reset_index(drop=True)
    trainset_2B = train.loc[(train["pred_1"] == 0)]
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(trainset_2B.drop(['UANZAHL', 'ZV_neu', 'pred_1'], axis=1))
    rf.fit(scaler.transform(trainset_2B.drop(["UANZAHL","ZV_neu","pred_1"],axis=1)),trainset_2B["UANZAHL"])
    pred_nn_1B = rf.predict(scaler.transform(testset_2B.drop(["UANZAHL","ZV_neu","pred_1"],axis=1))).round()
    testset_2B["pred_1"] = pd.Series(pred_nn_1B.reshape((pred_nn_1B.shape[0],)))
    # Zusammenführen und evaluieren 
    ganzes_testset = pd.concat([testset_2A,testset_2B], ignore_index = True, sort = False)
    mse = mean_squared_error(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"])
    f_one_score = f1_score(ganzes_testset["UANZAHL"], ganzes_testset["pred_1"], average='micro')
    runtime = time.time() - start
    return mse, f_one_score, runtime


def knn_oversampling_prognose(without,year):
    """Die Funktion erstellt mittels eines KNN und Oversamplings eine Vorhersage für die Anzahl an Unfällen und gibt 
    ein DataFrame mit AGS, Monat, Jahr und der prognostizierten Anzahl der Unfälle zur weiteren Verwendung aus.
    
    Parameter:
        without (list): Liste mit Attributen, welche nicht für die Prognose genutzt werden sollen. Die 
            Zielvariable wird standardmässig nicht genutzt und muss nicht eingetragen werden.
        year (int): Das Jahr, welches vorhergesagt werden soll. Mögl. Werte: 2017 bis 2020.
        
    Returns:
        prognose (DataFrame): Ein DataFrame mit AGS, Monat, Jahr und geschätzter Anzahl der Unfälle.
    """

    if Testmodus:
        daten = pd.read_csv("daten.csv") 
    else:
        daten = add_zulassungszahlen()
    daten = daten.sort_values(by=['UJAHR', 'UMONAT'])
    X_train_sam = daten.drop(['UANZAHL'], axis=1).loc[(daten['UJAHR'] < year)]
    y_train_sam = daten['UANZAHL'].loc[(daten['UJAHR'] < year)]
    ros = RandomOverSampler(sampling_strategy="not majority") # String
    Xsam, ysam = ros.fit_resample(X_train_sam, y_train_sam)
    X_test_sam = daten.drop(['UANZAHL'], axis=1).loc[(daten['UJAHR'] == year)]
    # Daten Normalisieren
    scaler = MinMaxScaler()
    scaler.fit(Xsam)
    X_train = scaler.transform(Xsam)
    X_test = scaler.transform(X_test_sam)
    # Layer und Neuronen definieren
    model = Sequential()
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(350, activation="relu"))
    model.add(Dense(1, activation="relu"))
    model.compile(optimizer="adam",loss="mse")
    # Training
    model.fit(X_train,ysam,epochs=3)
    # Vorhersage mit tatsächlichen Werten vergleichen
    test_predictions = model.predict(X_test)
    X_test_sam = X_test_sam.reset_index(drop=True)
    X_test_sam["Unfälle"] = pd.Series(test_predictions.reshape(5124,)).round()
    prognose = X_test_sam[["AGS","UJAHR","UMONAT","Bevölkerung","PLZ","Unfälle"]]
    prognose.to_csv("prognose2020.csv",index=False)
    return prognose
