import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
from test import run_get_graph
import dash_table
from graph_to_recommend import graph_to_recommend
import os


if __name__ == '__main__':
    host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
    port=3306
    dbname="pubmed"
    user="jks17"
    password="password"
    app = dash.Dash()

    # Loading screen CSS
    #css_directory = os.getcwd()
    #stylesheets = ['loading.css']
    #static_css_route = '/static/'
    #for stylesheet in stylesheets:
    #    app.css.append_css({"external_scripts": "/static/{}".format(stylesheet)})

    df1 = {'columns': ['Recommended Papers 📄']}
    df2 =  {'columns': ['Recommended People 👩‍🔬👨‍🔬', 'Affiliation 🏫']}

    app.layout = html.Div([
    dcc.Input(id='username', value='', type='text', size='80'),
    html.Button(id='submit-button', type='submit', children='Submit'),
    html.Div(id='output_div')
        ])


    @app.callback(Output('output_div', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('username', 'value')],
                                                                              )
    def update_output(clicks, input_value):
        if clicks is not None:
            my_file = open("test.txt","w+")
            my_file.write(input_value)
            my_file.close()
            #print('getting graph')
            #graph, descriptors = run_get_graph(host, port, dbname, user, password)
            #print('obtained graph')
            #top_k_papers, top_k_people = graph_to_recommend(graph, host, port, dbname, user, password)
            top_k_papers = ['covid', 'test2', 'test3']
            top_k_people = ['Guada', 'Josh']
            affiliation_dict = {
                'Guada': 'Imperial',
                'Josh': 'Imperial'
            }

            layout = [dash_table.DataTable(
            id='table',
                columns=[{"name": i, "id": i} for i in df1['columns']],
                    data=[{'Recommended Papers 📄': x} for x in top_k_papers],
                    style_header={'backgroundColor': '#E0FFFF'},
                    style_cell={'textAlign': 'left'},
                    style_table={"margin-top": "25px"},
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Region'},
                    'textAlign': 'left'
                }]
            ),
            dash_table.DataTable(
                    id='table2',
                        columns=[{"name": i, "id": i} for i in df2['columns']],
                            data=[{'Recommended People 👩‍🔬👨‍🔬': x, 'Affiliation 🏫': affiliation_dict[x]} for x in top_k_people],
                            style_header={'backgroundColor': '#E6E6FA'},
                            style_cell={'textAlign': 'left'},
                            style_table={"margin-top": "25px"},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Region'},
                            'textAlign': 'left'
                        }
                    ]
            )]
            
            return layout

    

    app.run_server(host="127.0.0.1", debug=True)
