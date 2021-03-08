import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
from test import run_get_graph

if __name__ == '__main__':
    app = dash.Dash()

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
            print('created file')
            graph, descriptors = run_get_graph()
            return u'nodes are {}, descriptors are {}'.format(graph.nodes(), descriptors)

    

    app.run_server(debug=True)
