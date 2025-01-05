from .vector_store import get_vector_store


def get_retriever(vector_store):
    
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 10, 'fetch_k': 20}
        # search_type="similarity_score_threshold",
        # search_kwargs={"k": 1, "score_threshold": 0.2},
    )
