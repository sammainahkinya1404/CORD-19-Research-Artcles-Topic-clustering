import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv("metadata.csv")

# Dropping unnecessary columns
columns_to_drop = ['cord_uid', 'sha', 'source_x', 'doi', 'pmcid', 'pubmed_id', 'license', 'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse', 'has_pmc_xml_parse', 'full_text_file', 'url', 'journal', 'publish_time', 'authors']
new_data = data.drop(columns_to_drop, axis=1)

# Dropping null values
new_df = new_data.dropna()

# Text preprocessing function
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

new_data_40k = new_df.head(5000)
new_data_40k['title'] = new_data_40k['title'].apply(preprocess_text)
new_data_40k['abstract'] = new_data_40k['abstract'].apply(preprocess_text)

# Feature extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = new_data_40k['title'].tolist() + new_data_40k['abstract'].tolist()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Topic modeling using LDA
from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_output = lda_model.fit_transform(tfidf_matrix)

# Clustering using K-means on LDA output
from sklearn.cluster import KMeans

num_clusters = 3
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_model.fit(lda_output)
cluster_labels = kmeans_model.labels_
new_data_40k['cluster'] = cluster_labels[:5000]

# Coherence score computation
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import gensim

tokenized_texts = [text.split() for text in new_data_40k['abstract']]
dictionary = corpora.Dictionary(tokenized_texts)
corpus_gensim = [dictionary.doc2bow(text) for text in tokenized_texts]

lda_model_gensim = gensim.models.ldamodel.LdaModel(corpus=corpus_gensim, num_topics=5, id2word=dictionary, random_state=42)
coherence_model = CoherenceModel(model=lda_model_gensim, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
lda_coherence = coherence_model.get_coherence()

# Streamlit app
st.title("Abstract Topic Prediction and Coherence Score")

abstract = st.text_area("Paste the abstract here:")

if st.button("Predict Topic and Compute Coherence Score"):
    if abstract:
        processed_abstract = preprocess_text(abstract)
        
        # Feature extraction
        tfidf_matrix_abstract = tfidf_vectorizer.transform([processed_abstract])
        
        # Predict topic using LDA
        lda_output_abstract = lda_model.transform(tfidf_matrix_abstract)
        topic = lda_output_abstract.argmax(axis=1)[0]
        
        # Compute coherence score
        tokenized_text = processed_abstract.split()
        corpus_gensim_abstract = [dictionary.doc2bow(tokenized_text)]
        coherence_model_abstract = CoherenceModel(model=lda_model_gensim, texts=[tokenized_text], dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_abstract.get_coherence()
        
        # Display topic and coherence score
        topic_descriptions = {
            0: "Medical Research (focus on angiotensin, renin, cardiovascular or biochemical studies)",
            1: "Public Health and Epidemiology (focus on disease, health, COVID-19, influenza, public health emergencies, infectious diseases, epidemiological studies)",
            2: "Respiratory Viruses (focus on respiratory infections, viruses, SARS, coronavirus, acute respiratory syndrome)",
            3: "Virology and Immunology (focus on viral infections, cellular mechanisms, immune responses)",
            4: "Miscellaneous Terms (diverse and less coherent, range of subjects or noise in the dataset)"
        }
        
        st.write(f"Predicted Topic: {topic} - {topic_descriptions[topic]}")
        st.write(f"Coherence Score: {coherence_score:.4f}")
    else:
        st.write("Please paste an abstract to predict its topic and compute the coherence score.")

