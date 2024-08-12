from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_CHROMA_FILE = "instruct-embeddings-public-crypto"


model_kwargs = {'device': 'cpu'}
# hkunlp/instructor-large
embeddings_model = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large", model_kwargs=model_kwargs)

# Load from file
vectorstore_chroma = Chroma(
    persist_directory=INDEX_CHROMA_FILE,
    embedding_function=embeddings_model
)

retriever_chroma = vectorstore_chroma.as_retriever(
    search_kwargs={'k': 3}
)

relevant_docs = retriever_chroma.invoke("What are the recent advances on public key cryptography?")
print(len(relevant_docs))
print(relevant_docs)
print('-'*100)



