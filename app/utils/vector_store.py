from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from uuid import uuid4
from functools import lru_cache


client = MongoClient(os.getenv("CONNECTION_STRING"))
DB_NAME = "market_minds_ai"
COLLECTION_NAME = "tech_news_vectorstore"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "tech_news_vectorstore_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

@lru_cache(maxsize=1)
def get_vector_store():
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=MONGODB_COLLECTION,
        embedding=GoogleGenerativeAIEmbeddings(
            model= "models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    # vector_store.create_vector_search_index(dimensions=768)
    return vector_store



def add_to_vector_store(docs, vector_store):
    print("Adding docs to vector store ...")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    print(f"Documents successfully added to collection")
