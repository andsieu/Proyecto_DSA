#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

website = Socrata('www.datos.gov.co', None)
table = website.get("8yi9-t44c", limit=200000)
df = pd.DataFrame.from_records(table)
df.desde = pd.to_datetime(df.desde)
df.hasta = pd.to_datetime(df.hasta)
df.valortarifa = df.valortarifa.astype('int')
df.cantidadtrafico = df.cantidadtrafico.astype('int')
df.cantidadevasores = df.cantidadevasores.apply(pd.to_numeric, errors='coerce')
df.cantidadexentos787 = df.cantidadexentos787.apply(pd.to_numeric, errors='coerce')
df['mes'] = df.hasta.dt.strftime('%Y-%m')
df['recaudo'] = df.valortarifa * df.cantidadtrafico
df = df.drop(['desde', 'hasta', 'valortarifa'], axis=1).groupby([
    'idpeaje', 'peaje', 'categoriatarifa', 'mes']).sum().reset_index()
df = df[['peaje', 'categoriatarifa', 'mes', 'recaudo']]

# Iniciar la aplicación Dash
app = dash.Dash(__name__)

# Obtener la lista de peajes y meses
peajes = df['peaje'].unique()
meses = df['mes'].unique()

# Diseño del filtro y del gráfico
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='peaje-dropdown',
            options=[{'label': peaje, 'value': peaje} for peaje in peajes],
            multi=True,
            value=[np.sort(df.peaje)[0]]
        )
    ], style={'width': '32%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Dropdown(
            id='mes-dropdown',
            options=[{'label': mes, 'value': mes} for mes in meses],
            multi=True,
            value=[df.mes.min()]
        )
    ], style={'width': '32%', 'display': 'inline-block'}),

    html.Div([
        # Eliminado el componente Dropdown para categoriatarifa
    ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'}),
    
    dcc.Graph(id='recaudo-line-chart'),
    dcc.Graph(id='recaudo-bar-chart')
])

# Callback para actualizar el gráfico de línea en función del filtro de peaje
@app.callback(
    Output('recaudo-line-chart', 'figure'),
    [Input('peaje-dropdown', 'value')]
)
def update_line_chart(selected_peajes):
    if not selected_peajes:
        return px.line(title='Selecciona al menos un peaje')

    filtered_df = df[df['peaje'].isin(selected_peajes)]
    summed_df = filtered_df.groupby(['mes']).sum().reset_index()

    fig = px.line(summed_df, x='mes', y='recaudo',
                  labels={'recaudo': 'Recaudo Total', 'mes': 'Mes'},
                  title='Recaudo Total por Mes')
    
    return fig

# Callback para actualizar el gráfico de barras en función del filtro de mes
@app.callback(
    Output('recaudo-bar-chart', 'figure'),
    [Input('mes-dropdown', 'value')]
)
def update_bar_chart(selected_meses):
    if not selected_meses:
        return px.bar(title='Selecciona al menos un mes')

    filtered_df = df[df['mes'].isin(selected_meses)]
    
    summed_categoria_recaudo = filtered_df.groupby('categoriatarifa')['recaudo'].sum().reset_index()

    fig = px.bar(summed_categoria_recaudo, x='categoriatarifa', y='recaudo', color='categoriatarifa',
                 labels={'recaudo': 'Recaudo Total', 'categoriatarifa': 'Categoría de Tarifa'},
                 title='Recaudo Total por Categoría de Tarifa')
    
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

