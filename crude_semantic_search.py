#python3 crude_semantic_search.py "Machine learning is so easy."

import json
import sys
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

query = sys.argv[1]

#Swap out with the model of your choice
model = SentenceTransformer("all-MiniLM-L6-v2")

#Can play with different sentence/text lengths
#ref: https://huggingface.co/tasks/sentence-similarity
sentences = ['Deep learning is so straightforward.',
    'This is so difficult, like rocket science.',
    'I cannot believe how much I struggled with this.'
]

#If you have a large corpus that doesn't change this saves a lot of time
if os.path.exists('./sentence_embeddings.pickled') :
    with open('./sentence_embeddings.pickled', 'rb') as f:
        sentence_embeddings = pickle.load(f)
else :
    sentence_embeddings = model.encode(sentences[0:100])

    with open('./sentence_embeddings.pickled', 'wb') as f:
         pickle.dump(sentence_embeddings, f)

#You can also load previously indexed sentences
if os.path.exists('./faiss_index.pickled') :
    with open('./faiss_index.pickled', 'rb') as f:
        index = pickle.load(f)
else :
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(sentence_embeddings)

    with open('./faiss_index.pickled', 'wb') as f:
         pickle.dump(index, f)

xq = model.encode([query])

D, I = index.search(xq, 1)
result_indexes = np.array(I).flatten().tolist()

for index in result_indexes :
    print(sentences[index])

