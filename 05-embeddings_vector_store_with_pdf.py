# -*- coding: utf-8 -*-
"""embeddings-vector-store.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1D6YiIYV8SS-Pjook3co-KPk8XBzWANqS
"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install langchain_community langchain-huggingface chromadb requests pypdf

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

loader = PyPDFLoader("./public_key_cryptography.pdf")
pdf_data = loader.load()

print(pdf_data[0])

"""**Embeddings model from HuggingsFace**"""

model_kwargs = {'device': 'cpu'}
# hkunlp/instructor-large
embeddings_model = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large", model_kwargs=model_kwargs)

print(embeddings_model)

embeddings = embeddings_model.embed_query(pdf_data[0].page_content)

# embeddings

print(len(embeddings))  # embeddings length

"""**Split documents from pdfs**"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install langchain

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,
  chunk_overlap=50,
  length_function=len,
)

documents_split = text_splitter.split_documents(pdf_data)

print(len(pdf_data))

print(len(documents_split))

INDEX_CHROMA_FILE = "instruct-embeddings-public-crypto"

vectorstore_chroma = Chroma.from_documents(
    documents=documents_split,
    embedding=embeddings_model,
    persist_directory=INDEX_CHROMA_FILE
)

vectorstore_chroma.persist()

# Load from file
vectorstore_chroma = Chroma(
    persist_directory=INDEX_CHROMA_FILE,
    embedding_function=embeddings_model
)

query = "What is public key cryptography?"
docs = vectorstore_chroma.similarity_search_with_score(query, k=5)

print(docs[3])
