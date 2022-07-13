from Util.py import *
import math
import pandas as pd 
from ipywidgets import interactive 
from ipywidgets import widgets 
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

"""
Diese Datei enthält die Benutzeroberfläche und die Funktion zur Berechnung der Risikoklassen. Zur
Nutzung der Benutzoberfläche sollte der Code als Jupyter Notebook geöffnet werden.
Es werden die folgenden Dateien benötigt:
 - utils.py
"""


def get_risikoklasse(alter, sex, ags, year, month_1, month_2): 
    """Die Funktion berechnet auf Grundlage der eingegeben Parameter sowie der Prognose 
    der prognostizierten Unfallzahlen einen persönlichen Risikoscore, anhand derer
    die Einteilung in eine aus zehn Risikoklassen erfolgt.
    Parameter:
        alter (int): Das Alter des Nutzers in Jahren.
        ags (int): Der AGS des Wohnorts des Nutzers.
        month (int): Der Monat, für welchen der Nutzer seine Klasse ermittelt haben will.
        sex (int): Das Geschlecht des Ntuzers.
        year (int): Das Jahr, für welches der Nutzer seine Klasse ermittelt haben will.
        
    Returns:
        riskclass (int): Die Risikoklasse des Nutzers (zw. null u. neun).
    """
    try:
        if alter <=24: 
            alter = 1 
        elif 24<alter<65: 
            alter = 1.22 
        elif alter >64: 
            alter = 2
        year = int(year)
        ags = int(ags)
        risk_df = pd.read_csv("https://raw.githubusercontent.com/dasilvafabian/Projektseminar-SS22-Abgabe/main/prognose2020.csv")
        risk_ags = []
        for month in range(month_1,month_2+1):
            risk_ags.append(risk_df.loc[(risk_df['AGS'] == ags) & (risk_df['UJAHR'] == year) & (risk_df['UMONAT'] == month)]['Unfälle'].reset_index(drop=True)[0])
        anz_unf = sum(risk_ags)/len(risk_ags)
        bevölkerung = risk_df.loc[(risk_df['AGS'] == ags) & (risk_df['UJAHR'] == year) & (risk_df['UMONAT'] == month_1)]['Bevölkerung'].reset_index(drop=True)[0]
        riskscore = (anz_unf/bevölkerung) * alter * sex 
        riskclass = math.floor((riskscore/0.025132743362831857)*10)
        print("Ihre Risiklasse ist",riskclass)
        return riskclass
    except: 
        print("Die eingegebene AGS ist ungültig oder Ihre Eingaben sind unvollständig!")


# Interface erzeugen
my_result = interactive(get_risikoklasse,
                        alter = widgets.IntSlider(min=18, max=99, value=18, description='Alter: '), 
                        year = widgets.Dropdown( options=['Bitte Auswählen', '2020'], value='2020', description='Jahr:'),
                        month_1 = widgets.Dropdown( options=[('Bitte Auswählen', 0), ('Januar', 1), ('Feburar', 1), ('März', 1), ('April', 1), ('Mai', 1), ('Juni', 1), ('Juli', 1), ('August', 1), ('September', 1), ('Oktober', 1), ('November', 1), ('Dezember', 1)], value=0, description='Monat, Beginn:'), 
                        month_2 = widgets.Dropdown( options=[('Bitte Auswählen', 0), ('Januar', 1), ('Feburar', 1), ('März', 1), ('April', 1), ('Mai', 1), ('Juni', 1), ('Juli', 1), ('August', 1), ('September', 1), ('Oktober', 1), ('November', 1), ('Dezember', 1)], value=0, description='Monat, Ende:'), 
                        sex = widgets.Dropdown( options=[('Bitte Auswählen', 0), ('Weiblich', 1), ('Männlich', 2.13), ('Divers', 1)], value=0, description='Geschlecht:'), 
                        ags = widgets.Text(description= 'AGS',value='14522520', disabled=False))
display(my_result)
