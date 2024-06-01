import pandas  as pd
import chromadb
from chromadb.utils import embedding_functions


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

documents = []
metadatas = []
ids = []

for index, data in df.iloc[:40000].iterrows():
    documents.append(data['synopsis'])
    metadatas.append({"movie_name": data['movie_name'],
                      "genre": data['genre']
                     })
    ids.append(str(index+1))

collection.add(
    documents=documents,
    metadatas = metadatas,
    ids=ids
)