import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
import warnings
import faiss


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
vl_arr  = np.array(df_details_vector['vector_label'].values.tolist()).astype(np.float32)


class FaissSearchEngine():
    """
    A search engine class that uses the Faiss algorithm for searching on embeddings
    https://github.com/facebookresearch/faiss/wiki
    
    """
    def __init__(self, dim=768):

        # Faiss initialisation for indexing embeddings        
        self.dim = dim        
        self.index = faiss.IndexFlatL2(dim)   # build the index
        print(self.index.is_trained)
    

    def train_index(self, embs=vl_arr):
        self.index.add(embs)                # add vectors to the index
        print(self.index.ntotal)


    def search(self, query, topk=20):
        emb = np.array([b_model.encode(str(query))]) #calculate embedding of query 
        D, ids = self.index.search(emb, topk)
        return df_details_vector.iloc[ids[0]][['uri','label','comment']]
