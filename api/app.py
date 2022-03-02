import pandas as pd
import numpy as np 
from scipy import spatial
from sentence_transformers import SentenceTransformer
import flask
from flask import request, session, redirect, url_for, render_template, jsonify, send_file
from datetime import date, datetime
from json2html import *

import warnings
# from numba.errors import NumbaPerformanceWarning
warnings.simplefilter("ignore")

b_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

df_details = pd.read_csv('sbert_pretrained/df_details.csv')


#Load dictionaries to dataframes
np_label = np.load('sbert_pretrained/numpy_sbert_label_embeddings_dictionary.npy',allow_pickle=True)
df_label_emb = pd.DataFrame(np_label.flat[0].items(), columns=['uri', 'vector_label']) 

np_comment = np.load('sbert_pretrained/numpy_sbert_comment_embeddings_dictionary.npy',allow_pickle=True)
df_comment_emb = pd.DataFrame(np_comment.flat[0].items(), columns=['uri', 'vector_comment']) 

df_details_vector_temp = pd.merge(df_details, df_label_emb, how='left', left_on='uri', right_on='uri' )

df_details_vector = pd.merge(df_details_vector_temp, df_comment_emb, how='left', on='uri' )

print(df_details_vector.shape)


def closest(q):

    df_results = pd.DataFrame(columns=['uri','type','label','comment', 'label_vector','cosine','cosine_label','cosine_comment']) #create results dataframe 
    
    emb = b_model.encode(str(q)) #calculate embedding of query 

    for i in range(0,len(df_details_vector)):    # for each row in df_all
      sp_dist_label = spatial.distance.cosine(emb, df_details_vector.iloc[i,4])  #spatial distance query and label
      sp_dist_comment = spatial.distance.cosine(emb, df_details_vector.iloc[i,5])  #spatial distance query and comment
      cosine_label   = 0 if np.isnan(sp_dist_label) else 1 - sp_dist_label #cosine query and label - if nan then 0 since some label values may be be nan  
      cosine_comment = 0 if np.isnan(sp_dist_comment) else 1 - sp_dist_comment #cosine query and comment - if nan then 0 since some comment values may be be nan  
      
      # Keep the max cosine similarity between query-label and query comment 
      cosine_max = max(cosine_label, cosine_comment)
      
      # Load everything in the results dataframe 
      df_results = df_results.append([{'uri': df_details_vector.iloc[i,0],
                                       'type': df_details_vector.iloc[i,1],
                                       'label': df_details_vector.iloc[i,2],
                                       'comment': df_details_vector.iloc[i,3],
                                       'label_vector': df_details_vector.iloc[i,4],
                                       'cosine': cosine_max,
                                       'cosine_label': cosine_label ,
                                       'cosine_comment': cosine_comment}], ignore_index = True)
      
      # Sort the results by cosine_max column
      df_results.sort_values(by=['cosine'], ascending=False, inplace = True)
      df_disp_drop = df_results.drop(['label_vector','cosine','cosine_label','cosine_comment'], axis=1)
      df_disp_results = df_disp_drop[['label','comment','type','uri']]

    return df_disp_results

# Creating FastAPI instance 
app = flask.Flask(__name__)
app.secret_key = "ostool"

app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Ontology Scout Tool API</h1><p>This site is a prototype API for Ontology Scout Tool. Have fun!</p>"

@app.route('/search', methods=["POST",'GET'])
def search():
    if request.method == "POST":
        query = str(request.form["query"])
        session["query"] = query
        df_result = closest(query)     
        session["json_result"] = df_result.to_json(orient = "records")
        json_result = json2html.convert(json = session["json_result"])
        return f"""<h1>Search Results </h1>\
                <p>Query: {query}\
                <br>\
                <br>{json_result} </p>"""
    else:
        return render_template("search.html")


if __name__ == '__main__':
  app.run()
