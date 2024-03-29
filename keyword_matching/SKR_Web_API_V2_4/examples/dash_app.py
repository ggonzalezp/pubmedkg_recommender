import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
from test import run_get_graph_2
import dash_table
from graph_to_recommend import graph_to_recommend
from embeddings_to_visualize import get_embeddings_to_visualize
import os
import networkx as nx
import plotly.graph_objs as go
from transformers import pipeline
import pickle
import time
import json

import plotly.express as px
# Initialize the HuggingFace summarization pipeline
summarizer = pipeline("summarization")


from paper_nodes_to_knowledge import articles_to_knowledge
from paper_nodes_to_keywords import extract_overlapping_keywords
from paper_nodes_to_summary import articles_to_summary

global title_to_pmid
title_to_pmid = {}

# with open("pmid_to_summary_demo.pickle", 'rb') as f:
#     pmid_to_sentence = pickle.load(f)


global mesh_info

# with open('../../../graph_embedding/dataset/dict_pmid_refs.json') as json_file:
#     dict_pmid_refs = json.load(json_file)


# with open('../../../graph_embedding/dataset/dict_mesh_pmids.json') as json_file:
#     dict_mesh_pmids = json.load(json_file)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()



    host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
    port=3306
    dbname="pubmed"
    user="jks17"
    password="password"
    app = dash.Dash()


    df1 = {'columns': ['Recommended Papers 📄', 'PMID', 'Last Author', 'Article Summary']}
    df2 =  {'columns': ['Recommended People 👩‍🔬👨‍🔬', 'Latest Affiliation 🏫', 'Citations', 'Number of Papers']}
    df3 = {'columns': ['Affiliation 🏫', 'Number of Papers']}

    app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Main', children=[
                                                html.Div(id="banner", children=[html.H1("PubMed Explorer", id='title')],),
                                                html.Div(children=[
                                                                        html.H3(children='Introduce keywords or text:'),
                                                                        dcc.Textarea(id='username', value='', style={'width': '100%', 'height': 200}),
                                                                        html.Button(id='submit-button', type='submit', children='Get recommendations'),
                                                                        html.Button(id='submit-button-2', type='submit', children='Get recommendations and embeddings')

                                                                    ], id='div_main'),
                    
                                                html.Div(
                                                            dcc.Loading(
                                                                id="loading-2",
                                                                children=[html.Div(id='output_div')],
                                                                type="circle",
                                                        )),
                                                html.Div([dcc.Graph(id='Graph', figure=go.Figure(data=[],
                                                                                layout=go.Layout(
                                                                                title='',
                                                                                titlefont_size=16,
                                                                                )), style={"backgroundColor": "#F2F3F4", 'color': '#F2F3F4'}),
                                                                                #dcc.Graph(id='Graph', figure=None),
                                                                                html.Div(id='selected-data')
                                                        ], id='div_graph'),
                                                html.Div([dcc.Graph(id='Graph_Embeddings', figure=go.Figure(data=[],
                                                                                layout=go.Layout(
                                                                                title='',
                                                                                titlefont_size=16,
                                                                                )), style={"backgroundColor": "#F2F3F4", 'color': '#F2F3F4'})
                                                        ], id='div_graph_embeddings'
                                                    )
                                            
                                            ]),
            dcc.Tab(label='About', children=[
                                                html.H2('Pubmed Graph Explorer'),
                                                html.P(
                                                                                     """
                                                            The PubMed knowledge graph explorer is designed to be a useful tool for promising students to explore new topics and to find a department and supervisor 
                                                            at the forefront of their field of interest. The tool provides a simple interface, whereby a user can input text or keywords. The input is then used to obtain paper recommendations for a specified topic as well as to get information regarding the 
                                                            key authors and departments in the area.
                                                            """
                                                                                       ),
                                                html.H3('1) Extracting keywords and Querying the Database'),
                                                html.P('''  Mesh terms are extracted from the input using the MeSH on Demand API.'''),
                                                html.H3('2) Creating the Citation Graph'),
                                                html.P('''  The selected papers form a citation graph where edges between
                                                            the paper nodes correspond to a reference. This citation graph is supplied interactively to the user. The user can hover
                                                            over paper nodes to find their title; select clusters of nodes to which we list their common keywords as well as easily visualise which papers have the most connections in the citation graph.'''),
                                                html.H3('3) Paper Recommendations'),

                                                html.P('''  We apply the page-rank algorithm on the citation graph in order to get the top k paper recommendations for the user. We additionally filter on these recommendations so that the suggested papers have the maximal number of corresponding keywords. The recommended papers are outlined to the user in a table format. We obtain the article summary by applying an implementation of the Bidirectional and Auto-Regressive Transformer to the paper's abstract. The model is implemented in the HuggingFace library and has been pre-trained on text summarization using the CNN/DailyMail summarization dataset.
                                                '''),
                                                html.H3('4) Author and Department Recommendations'),
                                                html.P('''  We are also able to give author and department recommendations based on the citation graph. We introduce an author metric (number of citations x number of papers) and list the authors with the highest value of this metric in the graph. Additionally, we provide the affiliation for these authors based on their latest paper.


                                                            ''')
                                            ])
        ])
    ])
        

    @app.callback(Output('output_div', 'children'),
    Output('Graph', 'style'),
    Output('Graph', 'figure'),
    Output('Graph_Embeddings', 'figure'),
    Output('Graph_Embeddings', 'style'),
    Input('submit-button', 'n_clicks'),
    Input('submit-button-2', 'n_clicks'),
    [State('username', 'value')],
                     
                                                                        )
    def update_output(clicks, clicks2, input_value):
        ctx = dash.callback_context
        if (clicks is not None) or (clicks2 is not None):
            my_file = open("test.txt","w+")
            my_file.write(input_value)
            my_file.close()
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            print(button_id)

            if button_id == 'submit-button':
                print('selected first submit button')
                print('getting graph')
                ################
                #COMPUTATIONS
                ################
                graph, descriptors, dict_pmid_count_mesh = run_get_graph_2()
                print('obtained graph')


                # import pdb; pdb.set_trace()
                graph = nx.read_gexf("test_graph.gexf")
                top_k_papers, top_k_papers_pmids, top_k_people, top_k_people_ids, authors_to_affiliation, papers_to_author, citation_dict, number_papers_dict, affiliation_paper_count, pmid_to_title, graph, pagerank_ordered = graph_to_recommend(graph, dict_pmid_count_mesh, host, port, dbname, user, password)
                # import pdb; pdb.set_trace()
                global title_to_pmid
                title_to_pmid = dict([(value, key) for key, value in pmid_to_title.items()]) 
                # sentences = articles_to_knowledge(top_k_papers_pmids, host, port, dbname, user, password)
                sentences = articles_to_summary(top_k_papers_pmids, host, port, dbname, user, password, summarizer)


                #################
                #CITATION NETWORK
                ##################
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
                    size=20,
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
                        node_info =  pmid_to_title[str(adjacencies[0])] + '<br># of connections: '+str(len(adjacencies[1]))
                    else:
                        node_info =  str(adjacencies[0]) + '<br># of connections: '+str(len(adjacencies[1]))
                    node_trace['text']+=tuple([node_info])

                fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='',
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
                fig.update_layout(
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=16,
                        font_family="Roboto"
                    ))



                style = {'display':'block'}
    




                #################
                #TABLES WITH DATA
                ##################

                layout = [html.Div(children=[

                    html.H2(children='Recommendations', id='title_rec'),

                                                dash_table.DataTable(
                                                                        id='table1',
                                                                        columns=[{"name": i, "id": i} for i in df1['columns']],
                                                                        data=[{'Recommended Papers 📄': x[0:-6], 'PMID': title_to_pmid[x], 'Last Author': papers_to_author[x], 'Article Summary':sentences[title_to_pmid[x]]} for x in top_k_papers],

                                                                        style_header={'backgroundColor': '#f2f2f2', 'whiteSpace': 'normal','height': 'auto'},
                                                                        style_cell={'textAlign': 'left'},

                                                                        style_data={'whiteSpace': 'pre-wrap','height': 'auto'},
                                                                        style_table={"margin-top": "25px", 'whiteSpace': 'normal', 'height': 'auto'},
                                                                        style_cell_conditional=[

                                                                                                    {'if': {'column_id': 'Recommended Papers 📄'},
                                                                                                    'width': '50%'},
                                                                                                    {'if': {'column_id': 'PMID'},


                                                                                                    'width': '5%'},
                                                                                                    {'if': {'column_id': 'Last Author'},
                                                                                                    'width': '10%'},
                                                                                                    {'if': {'column_id': 'Article Summary'},
                                                                                                    'width': '35%'}
                                                                                                ]
                                                                    ),

                                                dash_table.DataTable(
                                                                        id='table2',
                                                                        columns=[{"name": i, "id": i} for i in df2['columns']],
                                                                        data=[{'Recommended People 👩‍🔬👨‍🔬': x, 'Latest Affiliation 🏫': authors_to_affiliation[x], 'Citations': citation_dict[top_k_people_ids[idx]], 'Number of Papers': number_papers_dict[top_k_people_ids[idx]]} for idx, x in enumerate(top_k_people)],
                                                                        style_header={'backgroundColor': '#f2f2f2', 'textColor':'pink', 'whiteSpace': 'normal','height': 'auto'},
                                                                        style_cell={'textAlign': 'left'},

                                                                        style_data={'whiteSpace': 'normal','height': 'auto'},
                                                                        style_table={"margin-top": "40px", 'whiteSpace': 'normal', 'height': 'auto', 'align': 'center'},
                                                                        style_cell_conditional=[

                                                                                                    {'if': {'column_id': 'Recommended People 👩‍🔬👨‍🔬'},
                                                                                                    'width': '25%'},
                                                                                                    {'if': {'column_id': 'Latest Affiliation 🏫'},
                                                                                                    'width': '55%'},
                                                                                                    {'if': {'column_id': 'Citations'},


                                                                                                    'width': '8%'},
                                                                                                    {'if': {'column_id': 'Number of Papers'},
                                                                                                    'width': '12%'}
                                                                                                ]
                                                                    )
                                                                    #,
                                                #dash_table.DataTable(
                                                #                        id='table3',
                                                #                        columns=[{"name": i, "id": i} for i in df3['columns']],
                                                #                        data=[{'Affiliation 🏫': key, 'Number of Papers': affiliation_paper_count[key]} for key in affiliation_paper_count.keys()],

                                                #                      style_header={'backgroundColor': '#f2f2f2', 'whiteSpace': 'normal','height': 'auto'},
                                                #                       style_cell={'textAlign': 'left'},

                                                #                      style_data={'whiteSpace': 'normal','height': 'auto'},
                                                                        #style_table={"margin-top": "40px", 'whiteSpace': 'normal', 'height': 'auto'},
                                                                        #style_cell_conditional=[

                                                                                                    #{'if': {'column_id': 'Affiliation 🏫'},


                                                                                                    #'width': '85%'},
                                                                                                    #{'if': {'column_id': 'Number of Papers'},
                                                                                                    #'width': '15%'}
                                                                                                #]
                                                                    #)

                                                ], id='div_table_analytics'),
                                                html.H2(children='Citation graph of related papers', id='title_graph_div')]

                
                return layout, style, fig, go.Figure(data=[],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    )), {'display':'none'}

            elif button_id == 'submit-button-2':
                print('selected second submit button')
                print('getting graph')
                ################
                #COMPUTATIONS
                ################
                graph, descriptors, dict_pmid_count_mesh = run_get_graph_2()
                print('obtained graph')


                # import pdb; pdb.set_trace()
                graph = nx.read_gexf("test_graph.gexf")
                top_k_papers, top_k_papers_pmids, top_k_people, top_k_people_ids, authors_to_affiliation, papers_to_author, citation_dict, number_papers_dict, affiliation_paper_count, pmid_to_title, graph, pagerank_ordered = graph_to_recommend(graph, dict_pmid_count_mesh, host, port, dbname, user, password)
                title_to_pmid = dict([(value, key) for key, value in pmid_to_title.items()]) 
                # sentences = articles_to_knowledge(top_k_papers_pmids, host, port, dbname, user, password)
                sentences = articles_to_summary(top_k_papers_pmids, host, port, dbname, user, password, summarizer)


                #################
                #CITATION NETWORK
                ##################
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
                    size=20,
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
                        node_info =  pmid_to_title[str(adjacencies[0])] + '<br># of connections: '+str(len(adjacencies[1]))
                    else:
                        node_info =  str(adjacencies[0]) + '<br># of connections: '+str(len(adjacencies[1]))
                    node_trace['text']+=tuple([node_info])

                fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='',
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
                fig.update_layout(
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=16,
                        font_family="Roboto"
                    ))

                #################
                #EMBEDDING VISUALIZATION
                ##################
                t0 = time.time()
                emb = get_embeddings_to_visualize(descriptors, pagerank_ordered)
                t1 = time.time()
                print('got embeddings: {} secs'.format(t1-t0))

                fig_emb = px.scatter(emb,
                                            x='x',
                                            y='y',
                                            color='Node type',
                                            opacity=0.8,
                                            hover_data={'x':False,
                                                        'y':False,
                                                        'Name': True
                                            }
                                        )


                fig_emb.update_layout(
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=16,
                        font_family="Roboto"
                    ),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )

                fig_emb.update_traces(marker=dict(size=30),
                                selector=dict(mode='markers'))


                style = {'display':'block'}
                style_emb = {'display':'block'}




                #################
                #TABLES WITH DATA
                ##################

                layout = [html.Div(children=[

                    html.H2(children='Recommendations', id='title_rec'),

                                                dash_table.DataTable(
                                                                        id='table1',
                                                                        columns=[{"name": i, "id": i} for i in df1['columns']],
                                                                        data=[{'Recommended Papers 📄': x[0:-6], 'PMID': title_to_pmid[x], 'Last Author': papers_to_author[x], 'Article Summary':sentences[title_to_pmid[x]]} for x in top_k_papers],

                                                                        style_header={'backgroundColor': '#f2f2f2', 'whiteSpace': 'normal','height': 'auto'},
                                                                        style_cell={'textAlign': 'left'},

                                                                        style_data={'whiteSpace': 'pre-wrap','height': 'auto'},
                                                                        style_table={"margin-top": "25px", 'whiteSpace': 'normal', 'height': 'auto'},
                                                                        style_cell_conditional=[

                                                                                                    {'if': {'column_id': 'Recommended Papers 📄'},
                                                                                                    'width': '50%'},
                                                                                                    {'if': {'column_id': 'PMID'},


                                                                                                    'width': '5%'},
                                                                                                    {'if': {'column_id': 'Last Author'},
                                                                                                    'width': '10%'},
                                                                                                    {'if': {'column_id': 'Article Summary'},
                                                                                                    'width': '35%'}
                                                                                                ]
                                                                    ),

                                                dash_table.DataTable(
                                                                        id='table2',
                                                                        columns=[{"name": i, "id": i} for i in df2['columns']],
                                                                        data=[{'Recommended People 👩‍🔬👨‍🔬': x, 'Latest Affiliation 🏫': authors_to_affiliation[x], 'Citations': citation_dict[top_k_people_ids[idx]], 'Number of Papers': number_papers_dict[top_k_people_ids[idx]]} for idx, x in enumerate(top_k_people)],
                                                                        style_header={'backgroundColor': '#f2f2f2', 'textColor':'pink', 'whiteSpace': 'normal','height': 'auto'},
                                                                        style_cell={'textAlign': 'left'},

                                                                        style_data={'whiteSpace': 'normal','height': 'auto'},
                                                                        style_table={"margin-top": "40px", 'whiteSpace': 'normal', 'height': 'auto', 'align': 'center'},
                                                                        style_cell_conditional=[

                                                                                                    {'if': {'column_id': 'Recommended People 👩‍🔬👨‍🔬'},
                                                                                                    'width': '25%'},
                                                                                                    {'if': {'column_id': 'Latest Affiliation 🏫'},
                                                                                                    'width': '55%'},
                                                                                                    {'if': {'column_id': 'Citations'},


                                                                                                    'width': '8%'},
                                                                                                    {'if': {'column_id': 'Number of Papers'},
                                                                                                    'width': '12%'}
                                                                                                ]
                                                                    )
                                                                    #,
                                                #dash_table.DataTable(
                                                #                        id='table3',
                                                #                        columns=[{"name": i, "id": i} for i in df3['columns']],
                                                #                        data=[{'Affiliation 🏫': key, 'Number of Papers': affiliation_paper_count[key]} for key in affiliation_paper_count.keys()],

                                                #                      style_header={'backgroundColor': '#f2f2f2', 'whiteSpace': 'normal','height': 'auto'},
                                                #                       style_cell={'textAlign': 'left'},

                                                #                      style_data={'whiteSpace': 'normal','height': 'auto'},
                                                                        #style_table={"margin-top": "40px", 'whiteSpace': 'normal', 'height': 'auto'},
                                                                        #style_cell_conditional=[

                                                                                                    #{'if': {'column_id': 'Affiliation 🏫'},


                                                                                                    #'width': '85%'},
                                                                                                    #{'if': {'column_id': 'Number of Papers'},
                                                                                                    #'width': '15%'}
                                                                                                #]
                                                                    #)

                                                ], id='div_table_analytics'),
                                                html.H2(children='Citation graph of related papers', id='title_graph_div')]

                

                return layout, style, fig, fig_emb, style_emb

            else:
                return [], {'display':'none'}, go.Figure(data=[],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    )), go.Figure(data=[],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    )), {'display':'none'}
        else:
                return [], {'display':'none'}, go.Figure(data=[],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    )), go.Figure(data=[],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    )), {'display':'none'}


    @app.callback(
    Output('selected-data', 'children'),
    [Input('Graph','selectedData')],)
    def display_selected_data(selectedData):
        if selectedData:
            selectedData['points']
            num_of_nodes = len(selectedData['points'])
            text = []
            pmids = []
            for x in selectedData['points']:
                if x['text'][:x['text'].find('<br>')] in title_to_pmid:
                    pmids.append(title_to_pmid[x['text'][:x['text'].find('<br>')]])
                else:
                    pmids.append(x['text'][:x['text'].find('<br>')])
            keywords = extract_overlapping_keywords(pmids)
            keyword_string = 'Keywords:'
            for idx, keyword in enumerate(keywords):
                if idx > 0:
                    keyword_string += (', ' + str(keyword))
                else:
                    keyword_string += (' ' + str(keyword))
            text.append(html.P(keyword_string))

            return text
        else:
            return None


    

    app.run_server(host="127.0.0.1", debug=True)
