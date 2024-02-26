import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pprint
import streamlit as st
import random
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


def shuffle_string(s):
    # Convert the string into a list of characters
    char_list = list(s)
    # Shuffle the list in place
    random.shuffle(char_list)
    # Join the list back into a string
    shuffled_string = ''.join(char_list)
    return shuffled_string

def wrap_text(text, width=50):
    text = shuffle_string(text)
    return '<br>'.join([text[i:i+width] for i in range(0, len(text), width)])

# Load embeddings and their information
def load_embeddings(filename):
    with open(filename, 'r') as file:
        paragraphs = json.load(file)

    with open('./data/paragraphs.json', 'r') as file:
        pt = json.load(file)
    
    pt_lu = {}
    for row in pt :
        pt_lu[str(row['index'])] = {'text' : row['paragraph']}

    embeddings = np.array([paragraph['values'] for paragraph in paragraphs])
    texts = [pt_lu[str(paragraph['id'])]['text'] for paragraph in paragraphs]
    ids = [str(paragraph['id']) for paragraph in paragraphs]

    return embeddings, texts, ids

# Perform PCA on embeddings
def reduce_to_2d(embeddings):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    return embeddings_2d

# Create a DataFrame for easier handling with plotly
def create_dataframe(embeddings_2d, texts, ids, labels):
    print(len(embeddings_2d), len(texts), len(ids), len(labels))
    
    df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df['text'] = texts
    df['id'] = ids
    df['label'] = labels

    return df

# Main script starts here
if __name__ == '__main__':
    
    # Simulate some results and query embedding for demonstration purposes
    # In practice, these should come from your actual query and results processing
    #query_embedding = np.random.rand(1, 300)  # Simulated query embedding
    #results_embeddings = np.random.rand(5, 300)  # Simulated results embeddings

    filename = './data/rag-output.json'
    with open(filename, 'r') as file:
        rag_output = json.load(file)

    results_embeddings = [row['values'] for row in rag_output['results']]
    results_ids = [row['id'] for row in rag_output['results']]
    results_texts = [wrap_text(row['text']) for row in rag_output['results']]

    query = rag_output['query']
    query_embedding = rag_output['query_embedding']

    filename = './data/paragraphs-embeddings.json'
    embeddings, texts, ids = load_embeddings(filename)
    embeddings_texts = [wrap_text(text) for text in texts]
    embs = np.vstack([query_embedding, results_embeddings, embeddings])

    tsne = TSNE(n_components=3, random_state=42, perplexity=5)
    reduced_vectors = tsne.fit_transform(embs)

    # Create a 3D scatter plot
    scatter_plot = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color='grey', opacity=0.5, line=dict(color='lightgray', width=1)),
        text=[f"Point {i}" for i in range(len(reduced_vectors))]
    )

    # Highlight the first point with a different color
    highlighted_point = go.Scatter3d(
        x=[reduced_vectors[0, 0]],
        y=[reduced_vectors[0, 1]],
        z=[reduced_vectors[0, 2]],
        mode='markers',
        marker=dict(size=8, color='yellow', opacity=0.8, line=dict(color='lightgray', width=1)),
        text=[wrap_text(query)]
        
    )

    rend = len(results_embeddings) + 1

    blue_points = go.Scatter3d(
        x=reduced_vectors[1:rend, 0],
        y=reduced_vectors[1:rend, 1],
        z=reduced_vectors[1:rend, 2],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.8,  line=dict(color='black', width=1)),
        text=results_texts
    )

    # Create the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title=f'3D Representation after t-SNE (Perplexity=5)'
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Add the scatter plots to the Figure
    fig.add_trace(scatter_plot)
    fig.add_trace(highlighted_point)
    fig.add_trace(blue_points)

    fig.update_layout(layout)

    pio.write_html(fig, 'interactive_paragraph_embedding_3d_plot.html')
    fig.show()