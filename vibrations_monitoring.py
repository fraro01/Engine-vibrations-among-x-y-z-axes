# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:46:07 2025

@author: Francesco
"""

#IMPORTAZIONE LIBRERIE
import pandas as pd
from scipy.fft import rfft, rfftfreq
import numpy as np
#per visualizzazione dati dashboard
import dash 
from dash import dcc, html, Input, Output
import plotly.express as px



######################### DATASET E DATA PRE-PROCESSING #######################
#dataset importato
data = pd.read_csv("engine_failure_dataset.csv")

#suddivisione pandas series dei dati di interesse
data_vibration_x = data["Vibration_X [mm/s]"]
data_vibration_y = data["Vibration_Y [mm/s]"]
data_vibration_z = data["Vibration_Z [mm/s]"]
data_fault = data["Fault_Condition [-]"]

#data frame con solo i dati di interesse
combined_df = pd.concat([data_vibration_x, 
                         data_vibration_y, 
                         data_vibration_z,
                         data_fault], 
                        axis=1) #axis 1 indica l'orientamento di combinazione orizzontale

#raggruppamento dei valori con la severità di fault
#X
normal_x = combined_df["Vibration_X [mm/s]"][combined_df["Fault_Condition [-]"] == 0]
minor_fault_x = combined_df["Vibration_X [mm/s]"][combined_df["Fault_Condition [-]"] == 1]
moderate_fault_x = combined_df["Vibration_X [mm/s]"][combined_df["Fault_Condition [-]"] == 2]
severe_fault_x = combined_df["Vibration_X [mm/s]"][combined_df["Fault_Condition [-]"] == 3]
#Y
normal_y = combined_df["Vibration_Y [mm/s]"][combined_df["Fault_Condition [-]"] == 0]
minor_fault_y = combined_df["Vibration_Y [mm/s]"][combined_df["Fault_Condition [-]"] == 1]
moderate_fault_y = combined_df["Vibration_Y [mm/s]"][combined_df["Fault_Condition [-]"] == 2]
severe_fault_y = combined_df["Vibration_Y [mm/s]"][combined_df["Fault_Condition [-]"] == 3]
#Z
normal_z = combined_df["Vibration_Z [mm/s]"][combined_df["Fault_Condition [-]"] == 0]
minor_fault_z = combined_df["Vibration_Z [mm/s]"][combined_df["Fault_Condition [-]"] == 1]
moderate_fault_z = combined_df["Vibration_Z [mm/s]"][combined_df["Fault_Condition [-]"] == 2]
severe_fault_z = combined_df["Vibration_Z [mm/s]"][combined_df["Fault_Condition [-]"] == 3]

#################### FINE DATASET E DATA PRE-PROCESSING #######################



################################### RMS #######################################
#calcolo degli RMS
rms_values = {
    "Vibration_X": {
        0: np.sqrt(np.mean(normal_x**2)),
        1: np.sqrt(np.mean(minor_fault_x**2)),
        2: np.sqrt(np.mean(moderate_fault_x**2)),
        3: np.sqrt(np.mean(severe_fault_x**2)),
    },
    "Vibration_Y": {
        0: np.sqrt(np.mean(normal_y**2)),
        1: np.sqrt(np.mean(minor_fault_y**2)),
        2: np.sqrt(np.mean(moderate_fault_y**2)),
        3: np.sqrt(np.mean(severe_fault_y**2)),
    },
    "Vibration_Z": {
        0: np.sqrt(np.mean(normal_z**2)),
        1: np.sqrt(np.mean(minor_fault_z**2)),
        2: np.sqrt(np.mean(moderate_fault_z**2)),
        3: np.sqrt(np.mean(severe_fault_z**2)),
    },
}

################################### FINE RMS ##################################



############################### BASELINE & THRESHOLD ##########################
x_baseline = 0.9*np.mean(normal_x)
x_threshold = 1.0001*((np.mean(minor_fault_x) + np.mean(moderate_fault_x))/2)
y_baseline = 0.9*np.mean(normal_y)
y_threshold = 0.95*((np.mean(minor_fault_y) + np.mean(moderate_fault_y))/2)
z_baseline = 0.9*np.mean(normal_z)
z_threshold = 1.0001*((np.mean(minor_fault_z) + np.mean(moderate_fault_z))/2)

############################### FINE BASELINE & THRESHOLD #####################



################################## DASHBOARD ##################################
app = dash.Dash(__name__) #dash constructor for intializing the app
server = app.server #THIS IS FOR RENDER DEPLOY

app.layout = html.Div([ 
    # Contenitore flex per il titolo e il menu a tendina
    html.Div([
            # Titolo della dashboard
            html.H1(children='Engine Failure Detection',
                    style={
                        'height': '100%', 
                        'width':'50%',
                        'padding':'0',
                        'margin':'0',
                        'border':'0'
                        }
                    ),
            
            # Dropdown per selezionare l'asse di vibrazione
            dcc.Dropdown(
                        id='vibration-axis-dropdown',
                        options=[
                                {'label': 'Vibration X', 'value': 'Vibration_X'},
                                {'label': 'Vibration Y', 'value': 'Vibration_Y'},
                                {'label': 'Vibration Z', 'value': 'Vibration_Z'},
                                ],
                        value='Vibration_X',  # Valore di default
                        style={
                            'height': '100%', 
                            'width':'50%',
                            'padding': '0', 
                            'margin': '0', 
                            'border': '0', 
                            },
                        )
            ], 
            #stile generale di titolo + menu a tendina
            style={
                'display': 'flex',  # Usa flexbox per disporre gli elementi orizzontalmente
                'width': '100%'  # Occupa tutta la larghezza disponibile
                }
        ),
    
        # Grafico a barre con gli RMS per ciascun livello di fault
        dcc.Graph(id='rms-bar-graph',   
            style={'height': '49%', 
                   'width': '100%',
                   'padding':'0', 
                   'margin':'0',
                   'border':'0'
                   }
                ),
        
        # Grafico per la FFT
        dcc.Graph(id='fft-graph',
            style={'height': '49%', 
                   'width': '100%',
                   'padding':'0', 
                   'margin':'0',
                   'border':'0'
                   }
                ),
    
    ],  #stile Globale
    style={'height':'100vh', 
          'overflow':'hidden', 
          'padding':'0', 
          'margin':'0',
          'border':'0'
          }
          
 )

################################ FINE DASHBOARD ###############################



############################### LOGICHE DASHBOARD #############################
#decoratore per azioni dell'utente
@app.callback(
                Output('rms-bar-graph', 'figure'),
                Input('vibration-axis-dropdown', 'value')
                )    #input value
def update_rms_graph(selected_axis):
    
    rms_data = rms_values[selected_axis] #dal dizionario prendiamo i dati di interesse sulla selezione
    
    # Definizione dei colori per ogni livello di guasto
    color_map = {
        'Normal': 'green',
        'Minor Fault': 'yellow',
        'Moderate Fault': 'orange',
        'Severe Fault': 'red'
        }
    
    # Creazione del DataFrame per il grafico
    df = pd.DataFrame({
        'Fault Condition': ['Normal', 'Minor Fault', 'Moderate Fault', 'Severe Fault'],
        'RMS Value': [rms_data[0], rms_data[1], rms_data[2], rms_data[3]],
    })

    # Creazione del grafico a barre
    figure = px.bar(
        df,
        x='Fault Condition',
        y='RMS Value',
        color='Fault Condition',
        color_discrete_map=color_map,
        labels={'Fault Condition': 'Fault Condition', 'RMS Value': 'RMS Value'},
        title=f'RMS for {selected_axis}'
    )

    return figure


@app.callback(
                Output('fft-graph', 'figure'),
                Input('vibration-axis-dropdown', 'value')
                )    #input value
def update_fft_graph(selected_axis):
    
    # Estrazione dei dati selezionati e trasformazione di essi in un array numpy
    signal = combined_df[selected_axis + ' [mm/s]'].to_numpy()
    
    # Calcolo della FFT
    N = len(signal)  # Numero di campioni
    fft_values = rfft(signal)
    fft_freqs = rfftfreq(N, d=1)  #Assumendo una frequenza di campionamento di 1 Hz
    #a causa dell'unità di misura della vibrazione [mm/s]
    
    # Creazione del DataFrame per la FFT
    df_fft = pd.DataFrame({
        'Frequency': fft_freqs[1:],  # Escludiamo la componente DC (frequenza 0)
        'Amplitude': np.abs(fft_values)[1:]
    })
    
    # Creazione del grafico
    figure = px.line(
        df_fft,
        x='Frequency',
        y='Amplitude',
        labels={'Frequency': 'Frequency (Hz)', 'Amplitude': 'Amplitude'},
        title=f'FFT of {selected_axis} frequency spectrum' 
    )
    
    if selected_axis == 'Vibration_X':
        selected_baseline = x_baseline
        selected_threshold = x_threshold
        
    elif selected_axis == 'Vibration_Y':
        selected_baseline = y_baseline
        selected_threshold = y_threshold
        
    else:
        selected_baseline = z_baseline
        selected_threshold = z_threshold
    
    # Aggiunta delle barre verticali Baselin e Threshold
    figure.update_layout(
                        shapes=[
                            # Barra verde Baseline
                            dict(
                                type="line",
                                x0=selected_baseline,
                                y0=0,
                                x1=selected_baseline,
                                y1=df_fft['Amplitude'].max(),  # Altezza massima dell'asse Y
                                line=dict(color="green", width=2),
                            ),
                            # Barra rossa Threshold
                            dict(
                                type="line",
                                x0=selected_threshold,
                                y0=0,
                                x1=selected_threshold,
                                y1=df_fft['Amplitude'].max(),
                                line=dict(color="red", width=2),
                            )
                        ]
                    )
    
    # Aggiunta di tracce fittizie per la legenda
    figure.add_trace(
        dict(
            x=[None],  # Nessun dato effettivo da plottare
            y=[None],
            mode='markers',
            marker=dict(color="green"),
            name="Baseline"  # Etichetta della legenda
        )
    )
    
    figure.add_trace(
        dict(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(color="red"),
            name="Threshold"  # Etichetta della legenda
        )
    )
   
    
    
    return figure

############################ FINE LOGICHE DASHBOARD ###########################


#running
if __name__ == '__main__':
    app.run_server(debug=False)






















