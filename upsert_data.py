# Import the Pinecone library
from pinecone import Pinecone
import json
import numpy as np


pc = Pinecone(api_key=" pcsk_6zrVT1_FC6L9Qo6ZPYu79a3kUJSKTsgas9NdViKMq7bUfAG3xuMz6RNTLTR5pZb7V4ZMLg")
index = pc.Index('master-rag')

data = []
with open('../data/cars_data.txt', 'r') as file:
    data = file.read()
    data = json.loads(data)
    
dense_embeddings = np.load('../embeddings/dense_embeddings.npy')
print(dense_embeddings)

# for i in range(len(data)):
    