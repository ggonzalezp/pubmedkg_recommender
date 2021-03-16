import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
from test import run_get_graph
import dash_table
from graph_to_recommend import graph_to_recommend
import os
import networkx as nx
import plotly.graph_objs as go


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

    df1 = {'columns': ['Recommended Papers 📄', 'URL', 'Last Author']}
    df2 =  {'columns': ['Recommended People 👩‍🔬👨‍🔬', 'Affiliation 🏫', 'Citations', 'Number of Papers']}
    df3 = {'columns': ['Affiliation 🏫', 'Number of Papers']}

    app.layout = html.Div([
    dcc.Input(id='username', value='', type='text', size='80'),
    html.Button(id='submit-button', type='submit', children='Submit'),
    html.Div(id='output_div'),])


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


            graph = nx.read_gexf("test_graph.gexf")
            top_k_papers, top_k_papers_pmids, top_k_people, top_k_people_ids, authors_to_affiliation, papers_to_author, citation_dict, number_papers_dict, affiliation_paper_count, pmid_to_title = graph_to_recommend(graph, host, port, dbname, user, password)
            title_to_pmid = dict([(value, key) for key, value in pmid_to_title.items()]) 
            #get a x,y position for each node
            pos = nx.layout.spring_layout(graph)

            #Create Edges
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5,color='#888'),
                hoverinfo='none',
                mode='lines')
            for edge in graph.edges():
                x0, y0 = pos[graph.nodes[edge[0]]['label']]
                x1, y1 = pos[graph.nodes[edge[1]]['label']]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

            node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),  
                line=dict(width=2)))
            for node in graph.nodes():
                x, y = pos[graph.nodes[node]['label']]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])

            #add color to node points
            for node, adjacencies in enumerate(graph.adjacency()):
                node_trace['marker']['color']+=tuple([len(adjacencies[1])])
                if str(adjacencies[0]) in pmid_to_title:
                    node_info = 'Name: ' + pmid_to_title[str(adjacencies[0])] + '<br># of connections: '+str(len(adjacencies[1]))
                else:
                    node_info = 'Name: ' + str(adjacencies[0]) + '<br># of connections: '+str(len(adjacencies[1]))
                node_trace['text']+=tuple([node_info])

            fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Citation graph of related papers',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )



            #top_k_papers = ['covid', 'test2', 'test3']
            #top_k_people = ['Guada', 'Josh']
            #affiliation_dict = {
            #    'Guada': 'Imperial',
            #    'Josh': 'Imperial'
            #}


            layout = [dash_table.DataTable(
            id='table1',
                columns=[{"name": i, "id": i} for i in df1['columns']],
                    data=[{'Recommended Papers 📄': x, 'URL': 'https://pubmed.ncbi.nlm.nih.gov/{}/'.format(title_to_pmid[x]), 'Last Author': papers_to_author[x]} for x in top_k_papers],
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
                            data=[{'Recommended People 👩‍🔬👨‍🔬': x, 'Affiliation 🏫': authors_to_affiliation[x], 'Citations': citation_dict[top_k_people_ids[idx]], 'Number of Papers': number_papers_dict[top_k_people_ids[idx]]} for idx, x in enumerate(top_k_people)],
                            style_header={'backgroundColor': '#E6E6FA'},
                            style_cell={'textAlign': 'left'},
                            style_table={"margin-top": "25px"},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Region'},
                            'textAlign': 'left'
                        }
                    ]
            ),
            dash_table.DataTable(
                    id='table3',
                        columns=[{"name": i, "id": i} for i in df3['columns']],
                            data=[{'Affiliation 🏫': key, 'Number of Papers': affiliation_paper_count[key]} for key in affiliation_paper_count.keys()],
                            style_header={'backgroundColor': '#68FFB3'},
                            style_cell={'textAlign': 'left'},
                            style_table={"margin-top": "25px"},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Region'},
                            'textAlign': 'left'
                        }
                    ]
            ),

            html.Div(dcc.Graph(id='Graph',figure=fig))]
            
            return layout

    #@app.callback(
    #Output('selected-data', 'children'),
    #[Input('Graph','selectedData')])
    #def display_selected_data(selectedData):
    #    num_of_nodes = len(selectedData['points'])
    #    text = [html.P('Num of nodes selected: '+str(num_of_nodes))]
    #    for x in selectedData['points']:
    #        material = int(x['text'].split('<br>')[0][10:])
    #        text.append(html.P(str(material)))
    #return text

    

    app.run_server(host="127.0.0.1", debug=True)
