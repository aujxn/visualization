import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import unicodedata
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import subprocess

from numpy import *
import numpy as np
import psycopg2
#try:
connection = psycopg2.connect(user = "austen",
                              host = "127.0.0.1",
                              port = "5432",
                              database = "recipes")

cursor = connection.cursor()
# Print PostgreSQL Connection properties
print ( connection.get_dsn_parameters(),"\n")

# Print PostgreSQL version
cursor.execute("SELECT version();")
record = cursor.fetchone()
print("You are connected to - ", record,"\n")

cursor.execute("SELECT tags FROM nyt")
query = cursor.fetchall()

tag_counts = {}
for row in query:
    for item in row:
        for tag in item:
            tag_count = tag_counts.get(tag)
            if tag_count is None:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] = tag_count + 1

sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
sorted_tags = [i[0] for i in sorted_tags]

cursor.execute("SELECT name FROM ingredients")
query = cursor.fetchall()
ingredients = [unicodedata.normalize('NFC', j[0]) for j in [i for i in query]]

#print(ingredients)

cursor.close()

#except (Exception, psycopg2.Error) as error :
#    print ("Error while connecting to PostgreSQL", error)
#finally:
#closing database connection.
if(connection):
    cursor.close()
    connection.close()
    print("PostgreSQL connection is closed")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1(children='Recipe Analyzer and Generator'),
    html.H3(children='''
        A tool for investigating relationships between ingredients in cooking recipes
    '''),

    html.Div([
        html.Label([
            'Tags:',
            dcc.Dropdown(
                id='tags-dropdown',
                options=[{'label': i, 'value': i} for i in sorted_tags],
            ),
        ]),
    ],
    style={'width': '24%', 'display': 'inline-block'},
    ),

    html.Div([
        html.Label([
            'Ingredients:',
            dcc.Dropdown(
                id='ingredients-dropdown',
                options=[{'label': i, 'value': i} for i in ingredients],
                multi=True
            ),
        ]),
    ],
    style={'width': '74%', 'display': 'inline-block'}
    ),

    html.Div([
        html.Label([
            'Community detection algorithm:',
            dcc.RadioItems(
                id='algorithm',
                options=[{'label': 'Parallel Modularity Maximisation', 'value': 'parallel'}, {'label': 'Louvain\'s', 'value': 'louvains'}],
                labelStyle={'display': 'inline-block'}
            ),
        ]),
        html.Button('Embed!', id='embed', n_clicks=0),
    ]),

    html.Div([dcc.Loading(id='loading', type='default', children= html.Div(id='embedding'))], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20', 'border': 'double'}),
    ],
    style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'},
)

@app.callback(Output('embedding', 'children'),
               [Input('embed', 'n_clicks'),
                Input('algorithm', 'value'),
                Input('ingredients-dropdown', 'value'),
                Input('tags-dropdown', 'value'),
               ])
def make_embedding(n_clicks, algorithm, ingredients, tag):
    trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'embed' in trigger:
        #subprocess.run(["../recipe_analysis/target/debug/main", "analyze-nyt", "--tags", tag]);
        return embed(),
    else:
        raise PreventUpdate

from numpy import *
from scipy.sparse import csr_matrix
def embed():
    import numpy as np
    import sys
    import plotly.graph_objs as go
    import time
    

    random.seed(0);

    print('loading data')
    start = time.time()

    graphpath = 'temp/mat.temp'
    partpath = 'temp/part.temp'
    coordspath = 'temp/coords.temp'
    ingpath = 'temp/ing.temp'

    ingfile = open(ingpath)
    ingredients = [ing for ing in ingfile.readlines()]

    coordsfile = open(coordspath)
    coords = [[float(i) for i in line.split(" ")] for line in coordsfile.readlines()]
    coords = [coord if len(coord) > 2 else [coord[0], coord[1], 0.0] for coord in coords]

    graphfile = open(graphpath)
    edges = [(int(line.split(" ")[0]), int(line.split(" ")[1])) for line in graphfile.readlines()]

    partfile = open(partpath)
    # n is number of vertices in graph, K is number of levels in hierarchy
    n, K = partfile.readline().split(" ")
    n = int(n)
    K = int(K)
    partition_sizes = [int(i) for i in partfile.readline().strip().split(" ")]

    partitions = []
    for i in range(K):
        partition = []
        for j in range(partition_sizes[i]):
            partition.append([int(i) for i in partfile.readline().strip().split(" ")])
        partitions.append(partition)

    end = time.time()
    print("loaded in: " + str(end - start) + " seconds")
    print("building interpolation matrix")
    start = end
    ## TODO start as sparse matrix
    interpolation = [np.zeros((n, partition_sizes[0]), dtype=int)]
    for size in range(K-1):
        interpolation.append(np.zeros((partition_sizes[size], partition_sizes[size + 1]), dtype=int))

    for i in range(K):
        for j in range(partition_sizes[i]):
            for node in partitions[i][j]:
                interpolation[i][node][j] = 1

    interpolation = [csr_matrix(sparse) for sparse in interpolation]

    end = time.time()
    print(str(end - start) + " seconds")
    print("creating partition")
    start = end

    level = len(partitions)
    n_aggs = partition_sizes[level-1]
    part = interpolation[0]
    for i in range(level - 1):
        part = part.dot(interpolation[i+1])

    end = time.time()
    print(str(end - start) + " seconds")
    print("creating metadata")
    start = end

    texts = ["" for i in range(n_aggs)]
    vertices = [[] for i in range(n_aggs)]
    part_coo = part.tocoo()
    for i,j in zip(part_coo.row, part_coo.col):
        texts[j] = texts[j] + "<br>" + ingredients[i]
        vertices[j].append(i)

    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = go.Layout(
           title='',
           height=700,
           showlegend=False,
           scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
           margin=dict(t=100),
           #hovermode='closest',
           )

    plot_datas = []

    colors = ['rgb(' + str(int(256 * color[0])) + ',' + str(int(256 * color[1])) + ',' + str(int(256 * color[2])) + ')' for color in random.rand(n_aggs, 3)]

    DO_VERTICES = True
    if DO_VERTICES:
        for i in range(n_aggs):
            plot_datas.append(
                    go.Scatter3d(
                        x=[coords[j][0] for j in vertices[i]],  
                        y=[coords[j][1] for j in vertices[i]],  
                        z=[coords[j][2] for j in vertices[i]], 
                        visible=True,
                        mode='markers', 
                        marker=dict(
                            size=3,
                            opacity=1.0, 
                            color=colors[i],
                            line = dict(
                                color = 'rgb(0, 0, 0)',
                                width = 1
                                ),
                            ),
                        #group=[i for i in range(n)],
                        #opacity = [0.1 for i in range(n)],
                        #text=texts[i],
                        #hoverinfo="text"))
                        ))

        
    # add edges
    DO_EDGES = False
    if DO_EDGES:
        Xe = []
        Ye = []
        Ze = []
        print(len(coords))
        DO_SHIFT = False
        eps = 0.0
        if DO_SHIFT:
            eps = 0.001
        for (i,j) in edges:
            Xe += [coords[i][0], coords[j][0], None]
            Ye += [coords[i][1], coords[j][1], None]
            Ze += [coords[i][2] - eps, coords[j][2] - eps, None]
        linecolor = 'rgb(75,75,75)'
        #linewidth = 4.5 #2*math.sqrt(math.sqrt(1000/E))
        lineopacity = 0.01 #1.0 #math.sqrt(200/E)
        plot_datas.append(go.Scatter3d(x=Xe, 
                                       y=Ye, 
                                       z=Ze,
                                       visible=True,
                                       mode='lines', 
                                       line=dict(color=linecolor, width=linewidth),
                                       hoverinfo='none', 
                                       opacity=lineopacity))


    end = time.time()
    print(str(end - start) + " seconds")
    print('plotting')
    start = end
    fig = go.Figure(data=plot_datas, layout=layout)
    end = time.time()
    print(str(end - start) + " seconds")

    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
