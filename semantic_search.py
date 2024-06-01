import streamlit as st
import pandas  as pd
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(
    page_title="Movie Search",
    page_icon="🏳️",
    layout="wide"
)

def load_css(css_file):
    """Load the CSS file"""
    with open(css_file, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Header
st.markdown("""
    <div class="header-container">
        <header class="header-text">IMT589D: Building and Applying Large Language Models</header>
    </div>
""", unsafe_allow_html=True)

# Content
st.markdown("""
    <div class="content-container">
        <h1 class="title">Movie Search Box</h1>
        <div class="instruction">Type questions or keywords in the search box to find related movies.</div>
    </div>
""", unsafe_allow_html=True)

query = st.text_input("Search for Movies")


# model set up
df = pd.read_csv("/Users/jiashu/Documents/UW_IMT589D/project/train.csv", index_col =0)

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "movie_reco"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# hit enter to initiate the search. 
# previously use button #st.button("Search"):
if query:
    st.write(f"You searched for: {query}")

    query_results = collection.query(
        query_texts= query,
        n_results=10,
    )
    
    movie_synopsis = query_results["documents"][0]
    movie_meta= query_results['metadatas'][0]
    movie_id = [int(iid) for iid in query_results["ids"][0]]

    if movie_id:
        st.write("Results:")
        st.write("**Movie Name**<span style='padding-left: 20px'></span>**Movie Synopsis**", unsafe_allow_html=True)
        for idx, mid in enumerate(movie_id):
            movie_name = movie_meta[idx]['movie_name']
            movie_genre = movie_meta[idx]['genre']
            msynopsis = movie_synopsis[idx]
            st.markdown(f"{idx+1}: **{movie_name}**<span style='padding-left: 20px'></span>{msynopsis}", unsafe_allow_html=True)
    else:
        st.write("No results found")