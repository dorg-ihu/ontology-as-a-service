import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
import flask
from flask import request, session, redirect, url_for, render_template, jsonify, send_file
from datetime import date, datetime
from json2html import *
import warnings
import spacy
import faiss
from search.search_engine import FaissSearchEngine
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, MATCH, State, ALLSMALLER, ALL
import os
import re
import time
import uuid
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = stopwords.words('english') + [',', '?', '.', '!', '(', ')', '{', '}']

#Ignore warnings 
warnings.simplefilter("ignore")

# Initializing Dash app 
app = dash.Dash(__name__, 
    suppress_callback_exceptions=True, 
    external_stylesheets=[dbc.themes.LUMEN],
    prevent_initial_callbacks=True)
server = app.server

nlp = spacy.load("en_core_web_md")

# Initialize and train faiss search engine
faiss_search_obj = FaissSearchEngine()
faiss_search_obj.train_index()


def _filter_stop_words(text: str):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    #
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    text = text.rstrip().strip()
    text = nlp(text)
    words = [n.text for n in text if not n.is_punct]
    filtered_words = []
    for word in words:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_words.append(word)
    return filtered_words


def filter_stopwords(text: str):
    return _filter_stop_words(text)



text_input = dbc.Form(
    [dbc.FormGroup(
        [

            dbc.Input(id="input", placeholder="Input goes here...", type="text"),
            html.P(id="output"),

        ]
    )])
navbar = html.Div([dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    # dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Ontology as a service", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="",
        ),
        dbc.NavbarToggler(id="navbar-toggler")
        # dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),

    ],
    color="dark",
    dark=True,
),

])


inputs = html.Div(
    [
        dbc.Form(id='centrality-filters'),

        dbc.Form(id='dynamic-filters'),
        # dbc.Form([radioitems, checklist, switches]),
        dbc.Form(id='outliers-filters'),
        html.P(id="radioitems-checklist-output"),

    ]
)

def serve_layout():
    session_id = str(uuid.uuid4())

    row = html.Div(
        [
            html.Div(dcc.Input(id="input_session_id",type="text",value=session_id), hidden=True),
            dbc.Row(dbc.Col(html.Div([navbar]))),
            dbc.Row([dbc.Col(
                html.Div([text_input, dbc.Button("Search", id='search_button', color="primary", className="ml-2")]),
                width={"size": 8, "offset": 1})], ),
            dbc.Row(
                [
                    dbc.Col(inputs, width={"size": 3, "offset": 1}),
                    dbc.Col(html.Div(id='return_result'), width={"size": 12, })

                ]
            ),
            html.Data(id='source-profile',
                children="",
                hidden=True
            ),
        ]
    )
    return row

app.layout = serve_layout


def clear_text(n):
    text = re.sub(r"([?.!,¿])", r" \1 ", n)
    text = re.sub(r'[" "]+', " ", text)
    #
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

    text = text.rstrip().strip()
    return text


@app.callback([Output('return_result', 'children')],
              [State("input", "value"), Input("search_button", "n_clicks")])
def search(value, n_clicks):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if n_clicks is not None and triggered_id == 'search_button':

        # print(value, n_clicks)

        
        # Get most relevant 
        profiles = faiss_search_obj.search(str(value))

        # Convert to dictionary
        profiles = profiles.to_dict(orient='records')

        
        val_no_st = ' '.join(filter_stopwords(str(value)))
        for r in profiles:
            # print(r)
            highliedted = []

            if r['comment']=='' or not pd.isnull(r['comment']):
                # print(r['comment'])
                for n in r['comment'].split(' '):
                    text = re.sub(r"([?.!,¿])", r" \1 ", n)
                    text = re.sub(r'[" "]+', " ", text)
                    #
                    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
                    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)

                    text = text.rstrip().strip()
                    if text.lower() in [clear_text(q.lower()) for q in str(val_no_st).strip().split(' ')]:
                        n = '**' + text + '**'
                    highliedted.append(n)
                r['comment'] = dcc.Markdown(""" """.join(highliedted))

        cards = [
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [

                                html.H6(str(r['uri']), className="card-title",
                                    style={'font_size': '5px'}),
                                
                                html.H6(["Label"] + [
                                    dbc.Badge(r['label'], color="dark",
                                              className="ml-1")
                                    ]
                                ),

                                html.H6(
                                    ["Comment"] + [
                                        html.P(r['comment'], className="card-text")
                                    ]
                                ),
                                

                                dbc.Row(id={'type': 'feedback-choice', 'index': str(index)},
                                    children=[
                                    dbc.Col([
                                        dbc.Button('Relevant',
                                        id={'type': 'thumbs_up', 'index': str(index)},
                                        size='sm',
                                        style={'margin-top': 15}, color='success')], width=1),
                                    dbc.Col([
                                        dbc.Button('Not relevant',
                                        id={'type': 'thumbs_down', 'index': str(index)},
                                        size='sm',
                                        style={'margin-top': 15}, color='danger')], width=1),
                                    dbc.Col([
                                        dbc.Button('Neutral',
                                        id={'type': 'neutral', 'index': str(index)},
                                        size='sm',
                                        style={'margin-top': 15}, color='warning')], width=1),
                                    ]
                                ),

                                # html.Br(),
                                # html.Br(),

                                # dcc.RadioItems(
                                #     id={'type': 'feedback-choice', 'index': str(index)},
                                #     options=[
                                #         {'label': 'Thumbs up', 'value': '1'},
                                #         {'label': 'Thumbs down', 'value': '0'},
                                #         {'label': 'Neutral', 'value': '-1'},
                                #     ],
                                # ),

                                html.Div(id={

                                    'type': 'empty-response',
                                    'index': str(index)

                                    }
                                ),

                                
                                html.Div(id={

                                    'type': 'sim-profile',
                                    'index': str(index)

                                }, children=str(r['uri']), hidden=True
                                )
                            ]
                        ),

                    )
                )
            )
            for index, r in enumerate(profiles)
        ]     

        return [cards]
    else:
        return [[]]


@app.callback(
    Output({'type': 'empty-response', 'index': MATCH}, 'children'),
     [Input({'type': 'thumbs_up', 'index': MATCH}, 'n_clicks'),
      Input({'type': 'thumbs_down', 'index': MATCH}, 'n_clicks'),
      Input({'type': 'neutral', 'index': MATCH}, 'n_clicks'),
      State("input", "value"),
      State("input_session_id", "value"),
      State({'type': 'sim-profile', 'index': MATCH}, 'children')], prevent_initial_callbacks=True,)
def get_feedback(n_clicks_1, n_clicks_2, n_clicks_3, query, session_id, target_id):
    # print(n_clicks)
    # print(value)
    # if n_clicks:
    #     print(value)
    ctx = dash.callback_context
    ctrl_id = ctx.triggered[0]['prop_id'].split('.')[0]
    ctrl_id = str(ctrl_id)

    if 'thumbs_up' in ctrl_id:
        feedback = '1'
    elif 'thumbs_down' in ctrl_id:
        feedback = '-1'
    else:
        feedback = '0'

    data =[{
        'userID':session_id,
        'query': query,
        'targetID': target_id,
        'feedback': feedback,
        'timestamp': time.time()
        }
    ]
    data = pd.DataFrame(data)
    
    filepath = '../evaluation/search_feedback.csv' # TODO change the path
    if os.path.exists(filepath):
        data.to_csv(filepath, mode='a', header=False, index=False)
    else:
        data.to_csv(filepath, mode='w+', header=True, index=False)

    return ""


if __name__ == "__main__":
    # Get port and debug mode from environment variables
    port = 5000 # os.environ.get('dash_port')
    debug = True # os.environ.get('dash_debug') == "True"
    app.run_server(debug=debug, host="0.0.0.0", port=port)




# # Creating FastAPI instance 
# app = flask.Flask(__name__)
# app.secret_key = "ostool"

# app.config["DEBUG"] = True


# @app.route('/', methods=['GET'])
# def home():
#     return "<h1>Ontology Scout Tool API</h1><p>This site is a prototype API for Ontology Scout Tool. Have fun!</p>"

# @app.route('/search', methods=["POST",'GET'])
# def search():
#     if request.method == "POST":
#         query = str(request.form["query"])
#         session["query"] = query
#         df_result = search_faiss(query)  
#         print(query)    
#         session["json_result"] = df_result.to_json(orient = "records")
#         json_result = json2html.convert(json = session["json_result"])
#         return f"""<h1>Search Results </h1>\
#                 <p>Query: {query}\
#                 <br>\
#                 <br>{json_result} </p>"""
#     else:
#         return render_template("search.html")


# if __name__ == '__main__':
#   app.run(host='0.0.0.0')
