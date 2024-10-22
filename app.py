# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import plotly.express as px
import pyLDAvis
import pyLDAvis.lda_model
import joblib
import streamlit.components.v1 as components

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings('ignore')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("metadata.csv")
    return data

data = load_data()
st.title("Medical Journal Articles Topic Modelling")
st.write("### Data Preview")
st.dataframe(data.head())

# Data Cleaning
columns_to_drop = ['cord_uid', 'sha', 'source_x', 'doi', 'pmcid', 'pubmed_id', 'license',
                    'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse',
                    'has_pmc_xml_parse', 'full_text_file', 'url', 'journal', 'publish_time', 'authors']
new_data = data.drop(columns=columns_to_drop, axis=1).dropna()

# Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [PorterStemmer().stem(token) for token in lemmatized_tokens]
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Applying text preprocessing
new_data_30k = new_data.head(2000)
new_data_30k['title'] = new_data_30k['title'].apply(preprocess_text)
new_data_30k['abstract'] = new_data_30k['abstract'].apply(preprocess_text)

# TF-IDF Feature Extraction
corpus = new_data_30k['title'].tolist() + new_data_30k['abstract'].tolist()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# LDA Model Training
param_grid = {'n_components': [5, 10, 15, 20]}
lda_model = LatentDirichletAllocation()
grid_search = GridSearchCV(lda_model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(tfidf_matrix)

optimal_lda_model = LatentDirichletAllocation(n_components=grid_search.best_params_['n_components'], random_state=42)
optimal_lda_output = optimal_lda_model.fit_transform(tfidf_matrix)

# Save the LDA model and vectorizer
joblib.dump(optimal_lda_model, 'lda_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Display Topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

st.write("### LDA Topics")
topics = display_topics(optimal_lda_model, tfidf_vectorizer.get_feature_names_out(), 10)
topic_descriptions = {
    0: "Cardiovascular and Biochemical Studies",
    1: "Public Health and Epidemiology",
    2: "Respiratory Viruses",
    3: "Virology and Immunology",
    4: "Miscellaneous Terms"
}

for i, topic in enumerate(topics):
    st.write(f"**Topic {i+1}:** {topic}")
    st.write(f"*Description:* {topic_descriptions.get(i, 'No description available')}")

# K-Means Clustering
st.write("### K-Means Clustering")
num_clusters = st.slider("Select Number of Clusters for K-Means", min_value=2, max_value=10, value=3, step=1)
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_model.fit(optimal_lda_output)
cluster_labels = kmeans_model.labels_
new_data_30k['cluster'] = cluster_labels[:2000]

# PCA for Visualization
pca = PCA(n_components=2, random_state=42)
lda_output_2d = pca.fit_transform(optimal_lda_output)
centroids_2d = pca.transform(kmeans_model.cluster_centers_)

# Plot Clusters
fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i in range(num_clusters):
    ax.scatter(lda_output_2d[cluster_labels == i, 0], lda_output_2d[cluster_labels == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=300, c='yellow', marker='x', label='Centroids')
ax.set_title('Cluster Plot with Centroids')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
st.pyplot(fig)

# Cluster Validation
st.write("### Cluster Validation")
silhouette_avg = silhouette_score(optimal_lda_output, cluster_labels)
davies_bouldin_index = davies_bouldin_score(optimal_lda_output, cluster_labels)
st.write(f"**Silhouette Score:** {silhouette_avg}")
st.write(f"**Davies-Bouldin Index:** {davies_bouldin_index}")

# Plotly for Interactive Visualization
pca_output = pca.fit_transform(optimal_lda_output)
cluster_labels_str = [str(label) for label in cluster_labels]
fig = px.scatter(x=pca_output[:, 0], y=pca_output[:, 1], color=cluster_labels_str)
st.plotly_chart(fig)

# pyLDAvis visualization
st.write("### pyLDAvis Visualization")
panel = pyLDAvis.lda_model.prepare(optimal_lda_model, tfidf_matrix, tfidf_vectorizer)
html_string = pyLDAvis.prepared_data_to_html(panel)
components.html(html_string, width=1300, height=800)

# User input for text classification
st.write("### Predict Topic for Input Text")
user_input = st.text_area("Enter text (abstract or part of an article):", "")
if st.button("Predict Topic"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_tfidf = tfidf_vectorizer.transform([processed_input])
        lda_output = optimal_lda_model.transform(input_tfidf)
        topic_distribution = lda_output[0]
        predicted_topic = topic_distribution.argmax()
        
        # Check if input contains any keywords from the topics
        contains_keywords = any(keyword in processed_input for topic in topics for keyword in topic.split())
        
        if contains_keywords:
            st.write(f"**Predicted Topic:** Topic {predicted_topic + 1}")
            st.write(f"**Topic Keywords:** {topics[predicted_topic]}")
            st.write(f"**Topic Description:** {topic_descriptions.get(predicted_topic, 'No description available')}")

            # Display topic probabilities
            st.write("### Topic Probabilities")
            for i, prob in enumerate(topic_distribution):
                st.write(f"**Topic {i+1} Probability:** {prob:.4f}")

            # Predict cluster
            cluster_label = kmeans_model.predict(lda_output)[0]
            st.write(f"**Predicted Cluster:** Cluster {cluster_label + 1}")
        else:
            st.write("The input text does not contain keywords that determine its topic. It might not be a medical paper.")
    else:
        st.write("Please enter some text to predict its topic and cluster.")

