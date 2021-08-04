print()
print("Initiating script please wait...")
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

#We will train the model on this dataset
#This dataset is a json file collected from the web

print()
print("Loading Dataset from local machine ...")



dataset = pd.read_json(".\\News_Category_Dataset_v2.json", lines=True)
news_description = dataset["short_description"]                          #Training the data on the short description part of the loaded dataset

print("Dataset loaded: News_Category_Datset_v2.json")

start_time = time.time()
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')        #making a model out of distilBERT


print("\nEncoding BERT model on provided dataset. Please Wait ...")
print()

encoding_number = int(input("Enter the SIZE of data to encode the model from: "))

print("Please Wait...")
print()

embeddings_distilbert = model.encode(news_description.values[0:encoding_number])      #encoding the model on the dataset collected from web
                                                                           #creating sentence embeddings


end_time = time.time()

time_taken = str(round((end_time - start_time),1)) + " second"

print("Time taken to train model - "+ time_taken) 

def find_similar(vector_representation, all_representations, k=1):
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])


descriptions = [input("\nEnter Sentence: ")]

no_similar = int(input("\nEnter no. of similar sentence: "))
print()

cnt = 1
for description in descriptions:
    #print(f"Description: {description}")
    #print()
    
    distilbert_similar_indexes = find_similar(model.encode([description]), embeddings_distilbert, no_similar)
    print("\nMost similar descriptions using Sentence-Bert ---- \n")
    for index in distilbert_similar_indexes:
        print(str(cnt)+ ". " + news_description[index])
        print()
        cnt +=1