# -*- coding: utf-8 -*-
"""tool_SBERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gsqWU6KO6XZT6YRb-Ah1xp-aMi2--8jJ
"""

import rdflib
from rdflib import Graph
import pandas as pd
import en_core_web_sm
from scipy import spatial
import random 
import re
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from scipy.spatial.distance import cosine
import sklearn.feature_extraction.text as sktf
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import tkinter as tk


#load pre-trained bert embedding model
b_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

#regular expresions, stop words removal and stemmer
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english')) 

def str_cleaner_stemmer(s):
    s = str(s)
    s_clean = (re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'," ", s))
    return " ".join([stemmer.stem(word) for word in s_clean.lower().split() if word not in stop_words])

#count number of common words
def str_common_word(s1, s2):
	return sum(int(s2.find(word)>=0) for word in s1.split())

#fuzzy logic ratio
def fuzzy_WRatio(s1,s2):
    return fuzz.WRatio(s1,s2)/100

#tf-idf cosine similarity
def dist_cosine(data,col1,col2):   
    cos=[]
    for i in range(len(data.id)):        
        st=data[col1][i]
        title=data[col2][i] 
        if len(title)!=0:       
          tfidf = sktf.TfidfVectorizer().fit_transform([st,title])
          c=((tfidf * tfidf.T).A)[0,1]
        else:
          c=0       
        cos.append(c)                                   
    return cos
    
#Jaccard similarity 
def jaccard_similarity(s1,s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if len(set1.union(set2))!=0:
      j_sim = len(set1.intersection(set2)) / len(set1.union(set2))
    else:
      j_sim = 0
    return j_sim

pd.set_option('max_colwidth', 400)

######### IMPORT ALL DATAFRAMES ############

g = Graph()
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\cdm.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\eli.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\European_Commission_Conceptual_Framework.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\dcat.rdf", format = 'xml') 
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\european-qualification-framework-skos-ap-eu.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\DET-skos-ap-eu.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\Core_Public_Service_Vocabulary.ttl", format = 'turtle')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\Core_Public_Organisation_Vocabulary_v1.ttl", format = 'turtle')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\Core_person_vocabulary-v1.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\Core_location_vocabulary-v1.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\Core_Criterion_and_Core_Evidence_Vocabulary-v1.ttl", format = 'turtle')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\core_business_vocabulary.rdf", format = 'xml')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\BITS_ttl.ttl", format = 'turtle')
g.parse(r"C:\Users\patrikios\.spyder-py3\project\data\adms_v1.rdf", format = 'xml')

print("graph has {} statements.".format(len(g)))

qres = g.query(
    """SELECT ?uri (sample(?typeX) as ?type) (sample(?labelX) as ?label) (sample(?commentX) as ?comment) WHERE
{
   optional{?uri rdfs:label ?labelX}
   optional{?uri rdfs:comment ?commentX}
   optional{?uri rdfs:comment ?commentX}
   optional{?uri a ?typeX}

} GROUP BY ?uri 
  """)

df_details = pd.DataFrame(columns=['uri','type','label','comment'])
for row in qres:
    df_details = df_details.append([{'uri': row[0],'type':row[1],'label': row[2],'comment': row[3]}], ignore_index = True)

df_details = df_details.astype(str)
df_details

# Create the pre-trained embeddings for the labels in a new dataframe

df_vec_label = pd.DataFrame(columns=['uri','vector_label'])

for i in range(len(df_details)) : 
    if df_details.iloc[i, 2] in [None, 'None', np.nan, '','nan']:
        vec = [0] * 768 #size of pre-trained vectors
    else:
        vec = b_model.encode(str(df_details.iloc[i, 2])) # takes the mean vector for the entire sentence https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
    df_vec_label = df_vec_label.append([{'uri': df_details.iloc[i,0],'vector_label': vec}], ignore_index = True)


# Create the pre-trained embeddings for the comments in a new dataframe

df_vec_comment = pd.DataFrame(columns=['uri','vector_comment'])

for i in range(len(df_details)) : 
    if df_details.iloc[i, 3] in [None, 'None', np.nan, '','nan']:
        vec = [0] * 768 #size of pre-trained vectors
    else:
        vec = b_model.encode(str(df_details.iloc[i, 3])) # takes the mean vector for the entire sentence https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
    df_vec_comment = df_vec_comment.append([{'uri': df_details.iloc[i,0],'vector_comment': vec}], ignore_index = True)

#merge with vector of labels and comments to create a dataframe which contains vector for labels and comments
df_details_vector_temp = pd.merge(df_details, df_vec_label, how='left', left_on='uri', right_on='uri' )

df_details_vector = pd.merge(df_details_vector_temp, df_vec_comment, how='left', on='uri' )

########################## feature engineering ################################

def closest(q):

    df_results = pd.DataFrame(columns=['uri','type','label','comment', 'label_vector','cosine','cosine_label','cosine_comment']) #create results dataframe 
    
    emb = b_model.encode(str(q)) #calculate embedding of query 

    df_results = df_results.append([{'uri': 'q_uri',
                                  'type': 'q_type',
                                  'label': str(q),
                                  'comment': 'q_comment',
                                  'label_vector': emb,
                                  'cosine': 1,
                                  'cosine_label': 1 ,
                                  'cosine_comment': 0}], ignore_index = True)

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

    return df_results.iloc[0:15]

df_c = closest('Identity card')
df_c




######### EXPORT ###################

#Create dictionary uri:label_vector and export it to numpy file
sbert_label_embeddings_dictionary = dict(zip(df_vec_label['uri'], df_vec_label['vector_label']))
np.save(r'C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\numpy_sbert_label_embeddings_dictionary.npy',sbert_label_embeddings_dictionary)

#Create dictionary uri:comment_vector and export it to numpy file
sbert_comment_embeddings_dictionary = dict(zip(df_vec_comment['uri'], df_vec_comment['vector_comment']))
np.save(r'C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\numpy_sbert_comment_embeddings_dictionary.npy',sbert_comment_embeddings_dictionary)


df_details.to_csv(r'C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\df_details.csv', index = False)



######### IMPORT ##############
#df_details = pd.read_csv(r"C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\df_details.csv")
#
#
##Load dictionaries to dataframes
#np_label = np.load(r'C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\numpy_sbert_label_embeddings_dictionary.npy',allow_pickle=True)
#df_label_emb = pd.DataFrame(np_label.flat[0].items(), columns=['uri', 'vector_label']) 
#
#np_comment = np.load(r'C:\Users\patrikios\.spyder-py3\project\sbert_pretrained\numpy_sbert_comment_embeddings_dictionary.npy',allow_pickle=True)
#df_comment_emb = pd.DataFrame(np_comment.flat[0].items(), columns=['uri', 'vector_comment']) 
#
#df_details_vector_temp = pd.merge(df_details, df_label_emb, how='left', left_on='uri', right_on='uri' )
#
#df_details_vector = pd.merge(df_details_vector_temp, df_comment_emb, how='left', on='uri' )
#












































# =============================================================================
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# 
# labels = []
# tokens = []
# 
# for i in range(len(df_c)):
#     tokens.append(df_c.iloc[i,4])
#     labels.append(df_c.iloc[i,2])
# 
# tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=False)
# new_values = tsne_model.fit_transform(tokens)
# 
# x = []
# y = []
# for value in new_values:
#     x.append(value[0])
#     y.append(value[1])
#     
# plt.figure(figsize=(16, 16)) 
# for i in range(len(x)):
#     plt.scatter(x[i],y[i])
#     plt.annotate(labels[i],
#                   xy=(x[i], y[i]),
#                   xytext=(5, 2),
#                   textcoords='offset points',
#                   ha='right',
#                   va='bottom')
# plt.title('Sentence Bert Model')
# plt.show()
# =============================================================================

