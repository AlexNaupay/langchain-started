from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

documents_to_embedded = [
    "¡Hola parce!",
    "¡Uy, hola!",
    "¿Cómo te llamas?",
    "Mis parceros me dicen Omar",
    "¡Hola Mundo!"
]
model_kwargs = {'device': 'cpu'}
# https://huggingface.co/models?library=sentence-transformers  More models
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=model_kwargs)
print(embeddings_model)

embeddings = embeddings_model.embed_documents(documents_to_embedded)

print(len(embeddings[0]))
# print(embeddings[0])
