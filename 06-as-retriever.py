import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

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
    search_kwargs={'k': 3}  # 3 relevant documents
)

relevant_docs = retriever_chroma.invoke("What are the recent advances on public key cryptography?")
print(len(relevant_docs))
print(relevant_docs)
print('-'*100)

# Integrate with openai
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    n=1,
    temperature=0.3
)

qa_chain_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_chroma
)

query = 'What is public key cryptography important for?'
answer = qa_chain_with_sources.invoke(query)
print(answer)
