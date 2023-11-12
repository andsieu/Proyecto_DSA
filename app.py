import numpy as np
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
import geopandas as gpd
from sodapy import Socrata
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

fs_url = 'https://aniscopiosig-server.ani.gov.co/arcgisserver/rest/services/OpenData/OpenDataICDE/FeatureServer/4'
layer = FeatureLayer(fs_url)
sdf = layer.query().sdf
gdf = gpd.GeoDataFrame(sdf, geometry='SHAPE')
gdf = gdf.rename(columns={'IdPea': 'idpeaje',
                          'Departamento': 'departamento',
                          'Municipio': 'municipio',
                          'Longitud': 'longitud',
                          'Latitud': 'latitud',
                          'SHAPE': 'shape'})

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
df['año'] = df.hasta.dt.year
df['recaudo'] = df.valortarifa * df.cantidadtrafico
df = df.drop(['desde', 'hasta', 'valortarifa'], axis=1).groupby([
    'idpeaje', 'peaje', 'categoriatarifa', 'mes', 'año']).sum().reset_index()

df = pd.merge(df, gdf[['idpeaje', 'departamento', 'municipio', 'longitud', 'latitud', 'shape']], on='idpeaje', how='left')
df = df.dropna()
df = df.reset_index(drop='index')

data = df.copy()
df = df[['peaje', 'categoriatarifa', 'mes', 'recaudo']]

# Funciones para transformar las variables categóricas
def transform_peaje(value):
    return label_encoder_peaje.transform([value])[0]

def transform_mes(value):
    return label_encoder_mes.transform([value])[0]

def transform_categoria(value):
    return label_encoder_categoria.transform([value])[0]

def transform_inverse(column, encoder):
    try:
        return encoder.inverse_transform(column)
    except ValueError as e:
        print(f'Error en la transformación inversa: {str(e)}')
        unseen_labels = set(column) - set(encoder.classes_)
        print(f'Categorías no vistas en el entrenamiento: {unseen_labels}')
        return [-1] * len(column)

# Usar Label Encoder para convertir las variables categóricas
label_encoder_peaje = LabelEncoder()
label_encoder_categoria = LabelEncoder()
label_encoder_mes = LabelEncoder()

# Ajustar el LabelEncoder con las categorías únicas
label_encoder_peaje.fit(df['peaje'])
label_encoder_categoria.fit(df['categoriatarifa'])
label_encoder_mes.fit(df['mes'])

df['peaje'] = df['peaje'].apply(transform_peaje)
df['categoriatarifa'] = df['categoriatarifa'].apply(transform_categoria)
df['mes'] = df['mes'].apply(transform_mes)

# Separar los datos en conjuntos de entrenamiento y prueba
X = df.drop('recaudo', axis=1)
y = df['recaudo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

datos = data[['peaje',
              'año',
              'recaudo',
              'longitud',
              'latitud']].groupby(['peaje', 'año', 'longitud', 'latitud']).sum().reset_index()
datos['texto'] = 'Recaudo Peaje ' + datos.peaje + ': ' + datos.recaudo.astype('str')

# Crear un gráfico Scattergeo
fig = go.Figure(data=go.Scattergeo(
    lat=datos.latitud,
    lon=datos.longitud,
    text = datos.texto,
    mode='markers',
    marker = dict(
        size = 8,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = False,
        symbol = 'circle',
        line = dict(
            width=1,
            color='rgba(102, 102, 102)'
        ),
        colorscale = 'Blues',
        cmin = 0,
        color = datos.recaudo,
        cmax = datos.recaudo.max(),
        colorbar_title='Recaudo'
    )))

# Diseño del punto medio de Colombia
fig.update_geos(
    center=dict(lon=-74, lat=6),
    lataxis_range=[1, 15],
    lonaxis_range=[-85, -65]
)

# Ajustes de diseño
fig.update_layout(
    title='Recaudo por Peaje',
    geo=dict(
        showland=True,
        landcolor='rgb(250, 250, 250)',
        subunitcolor='rgb(217, 217, 217)',
        countrycolor='rgb(217, 217, 217)',
        countrywidth=0.5,
        subunitwidth=0.5
    ),
    height=900,
    margin=dict(l=20, r=20, t=60, b=20)
)

# Obtener la lista de peajes, meses, y años categorías de tarifas
peajes = df['peaje'].unique()
meses = df['mes'].unique()
categorias_tarifa = df['categoriatarifa'].unique()

# Crear diccionarios de transformación inversa
peaje_inverse_dict = dict(zip(label_encoder_peaje.transform(label_encoder_peaje.classes_),
                              label_encoder_peaje.classes_))
mes_inverse_dict = dict(zip(label_encoder_mes.transform(label_encoder_mes.classes_),
                            label_encoder_mes.classes_))
categoria_inverse_dict = dict(zip(label_encoder_categoria.transform(label_encoder_categoria.classes_),
                                  label_encoder_categoria.classes_))

# Iniciar la aplicación Dash
app = dash.Dash(__name__)

# Establecer el título general de la aplicación
app.title = "SITUACIÓN ACTUAL PEAJES DE LA AGENCIA NACIONAL DE INFRAESTRUCTURA"

# Diseño del dashboard
app.layout = html.Div([
    
    html.H1("SITUACIÓN ACTUAL PEAJES DE LA AGENCIA NACIONAL DE INFRAESTRUCTURA"),

    # Fila superior
    html.Div([
        # Gráfico de líneas y Gráfico de barras con el mismo ancho (50% cada uno)
        html.Div([
            # Gráfico de líneas
            html.Div([
                html.H3("RECAUDO TOTAL POR MES"),
                # Filtros
                html.Div([
                    dcc.Dropdown(
                        id='peaje-dropdown',
                        options=[{'label': peaje, 'value': peaje} for peaje in data.peaje.unique()],
                        multi=True,
                        value=[np.sort(data.peaje)[0]]
                    )
                ], style={'width': '100%', 'display': 'inline-block'}),
                # Gráfico
                dcc.Graph(id='recaudo-line-chart')
            ], style={'width': '100%', 'display': 'inline-block', 'height': '50%'}),
            
            # Gráfico de barras
            html.Div([
                html.H3("RECAUDO TOTAL POR CATEGORÍA DE TARIFA"),
                # Filtros
                html.Div([
                    dcc.Dropdown(
                        id='mes-dropdown',
                        options=[{'label': mes, 'value': mes} for mes in data.mes.unique()],
                        multi=True,
                        value=[data.mes.min()]
                    )
                ], style={'width': '100%', 'display': 'inline-block'}),
                # Gráfico
                dcc.Graph(id='recaudo-bar-chart')
            ], style={'width': '100%', 'display': 'inline-block', 'height': '50%'}),
        ], style={'width': '50%', 'display': 'inline-block'}),

        # Mapa a la derecha con altura ajustada
        html.Div([
            html.H3("RECAUDO POR PEAJE"),
            dcc.Graph(id='map', figure=fig),
            dcc.Slider(
                id='year-slider',
                min=data['año'].min(),
                max=data['año'].max(),
                marks={str(year): str(year) for year in range(data['año'].min(), data['año'].max() + 1)},
                step=None,
                value=data['año'].max()
    )
        ], style={'width': '50%', 'float': 'right', 'height': '100%'}),
    ], style={'margin-bottom': '20px'}),
    
    # Sección de predicciones en la parte inferior (ajustado al 100% del ancho)
    html.Div([
        html.H3("REALIZAR PREDICCIONES"),
        # Filtros
        html.Div([
            dcc.Dropdown(
                id='prediccion-peaje-dropdown',
                options=[{'label': peaje_inverse_dict[peaje], 'value': peaje} for peaje in peajes],
                multi=True,
                value=[np.sort(df.peaje)[0]]
            )
        ], style={'width': '32%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='prediccion-mes-dropdown',
                options=[{'label': mes_inverse_dict[mes], 'value': mes} for mes in df['mes'].unique()],
                multi=True,
                value=[df.mes.max()]
            )
        ], style={'width': '32%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='prediccion-categoria-dropdown',
                options=[{'label': categoria_inverse_dict[categoria], 'value': categoria} for categoria in categorias_tarifa],
                multi=True,
                value=[np.sort(df.categoriatarifa)[0]]
            )
        ], style={'width': '32%', 'display': 'inline-block'}),
        # Botón de predicción
        html.Button('Realizar Predicción', id='button-predict'),
        # Resultado de predicción
        html.Div(id='prediction-output')
    ], style={'margin-top': '20px', 'width': '100%'}),
])

# Callback para actualizar el gráfico de línea en función del filtro de peaje
@app.callback(
    Output('recaudo-line-chart', 'figure'),
    [Input('peaje-dropdown', 'value')]
)
def update_line_chart(selected_peajes):
    if not selected_peajes:
        return px.line(title='Selecciona al menos un peaje')

    filtered_df = data[data['peaje'].isin(selected_peajes)]
    summed_df = filtered_df.groupby(['mes'])['recaudo'].sum().reset_index()
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

    filtered_df = data[data['mes'].isin(selected_meses)]
    summed_categoria_recaudo = filtered_df.groupby('categoriatarifa')['recaudo'].sum().reset_index()
    fig = px.bar(summed_categoria_recaudo, x='categoriatarifa', y='recaudo', color='categoriatarifa',
                 labels={'recaudo': 'Recaudo Total', 'categoriatarifa': 'Categoría de Tarifa'},
                 title='Recaudo Total por Categoría de Tarifa')
    
    return fig

# Callback para actualizar el mapa en función del año seleccionado en el slider
@app.callback(
    Output('map', 'figure'),
    [Input('year-slider', 'value')]
)
def update_map(selected_year):
    filtered_data = datos[datos['año'] == selected_year]
    fig = go.Figure(data=go.Scattergeo(
        lat=filtered_data.latitud,
        lon=filtered_data.longitud,
        text=filtered_data.texto,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            symbol='circle',
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale='Blues',
            cmin=0,
            color=filtered_data.recaudo,
            cmax=filtered_data.recaudo.max(),
            colorbar_title='Recaudo'
        )
    ))

    # Diseño del punto medio de Colombia
    fig.update_geos(
        center=dict(lon=-74, lat=6),
        lataxis_range=[1, 15],
        lonaxis_range=[-85, -65]
    )

    # Ajustes de diseño
    fig.update_layout(
        title='Recaudo por Peaje',
        geo=dict(
            showland=True,
            landcolor='rgb(250, 250, 250)',
            subunitcolor='rgb(217, 217, 217)',
            countrycolor='rgb(217, 217, 217)',
            countrywidth=0.5,
            subunitwidth=0.5
        ),
        height=900,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

# Callback para realizar predicciones
@app.callback(
    Output('prediction-output', 'children'),
    [Input('button-predict', 'n_clicks')],
    [State('prediccion-peaje-dropdown', 'value'),
     State('prediccion-mes-dropdown', 'value'),
     State('prediccion-categoria-dropdown', 'value')]
)
def make_prediction(n_clicks, selected_peajes, selected_meses, selected_categorias):
    if n_clicks is None:
        return ''

    # Crear un DataFrame para la predicción
    prediction_data = {'mes': selected_meses,
                       'peaje': selected_peajes,
                       'categoriatarifa': selected_categorias}

    prediction_df = pd.DataFrame(prediction_data)

    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    prediction_df = prediction_df[X_train.columns]

    # Realizar la predicción
    try:
        prediction = model.predict(prediction_df)
    except Exception as e:
        print('Error al realizar la predicción:', str(e))
        return f'Error al realizar la predicción: {str(e)}'

    return f'Predicción de Recaudo: {prediction[0]:.2f}'

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)