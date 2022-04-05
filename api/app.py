import pandas as pd
import numpy as np 
from scipy import spatial
from sentence_transformers import SentenceTransformer
import flask
from flask import request, session, redirect, url_for, render_template, jsonify, send_file
from datetime import date, datetime
from json2html import *
import warnings
import faiss

#Ignore warnings 
warnings.simplefilter("ignore")

#Set the pre-trained embedding model
b_model = SentenceTransformer('distilbert-base-nli-mean-tokens')


#IMPORT the essential data
df_details = pd.read_csv('sbert_pretrained/df_details.csv')
#Load dictionaries to dataframes
#Vlabel
np_label = np.load('sbert_pretrained/numpy_sbert_label_embeddings_dictionary.npy',allow_pickle=True)
df_label_emb = pd.DataFrame(np_label.flat[0].items(), columns=['uri', 'vector_label']) 
#Comment 
np_comment = np.load('sbert_pretrained/numpy_sbert_comment_embeddings_dictionary.npy',allow_pickle=True)
df_comment_emb = pd.DataFrame(np_comment.flat[0].items(), columns=['uri', 'vector_comment']) 

#Converting URIs to str and strip is essential for merging  
df_details['uri'] = df_details['uri'].str.strip()
df_label_emb['uri'] = df_label_emb['uri'].str.strip()
df_comment_emb['uri'] = df_comment_emb['uri'].str.strip()

#Merge all in one dataframe
df_details_vector_temp = pd.merge(df_details, df_label_emb, how='left', left_on='uri', right_on='uri' )
df_details_vector = pd.merge(df_details_vector_temp, df_comment_emb, how='left', on='uri' )

#Set the array that holds the embeddings from the labels to use it in faiss
vl_arr  = np.array(df_details_vector['vector_label'].values.tolist())



# Faiss initialisation for indexing embeddings        
dim = 768        
index = faiss.IndexFlatL2(dim)   # build the index
print(index.is_trained)
index.add(vl_arr)                # add vectors to the index
print(index.ntotal)


#Search function
def search_faiss(query):
    #Set the embedding 
    emb = np.array([b_model.encode(str(query))]) #calculate embedding of query 
    D, ids = index.search(emb, 20)
    return df_details_vector.iloc[ids[0]][['uri','label','comment']]


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
        df_result = search_faiss(query)  
        print(query)    
        session["json_result"] = df_result.to_json(orient = "records")
        json_result = json2html.convert(json = session["json_result"])
        return f"""<h1>Search Results </h1>\
                <p>Query: {query}\
                <br>\
                <br>{json_result} </p>"""
    else:
        return render_template("search.html")


if __name__ == '__main__':
  app.run(host='0.0.0.0')
