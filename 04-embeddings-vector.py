from langchain_huggingface import HuggingFaceEmbeddings

documents_to_embedded = [
    "¡Hola parce!",
    "¡Uy, hola!",
    "¿Cómo te llamas?",
    "Mis parceros me dicen Omar",
    "¡Hola Mundo!"
]
model_kwargs = {'device': 'cpu'}
# https://huggingface.co/models?library=sentence-transformers  More models
embeddings_model = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large", model_kwargs=model_kwargs)
print(embeddings_model)

embeddings = embeddings_model.embed_documents(documents_to_embedded)

print(len(embeddings[0]))
# print(embeddings[0])

embeddings_query = embeddings_model.embed_query("Hello, my friend")
print(embeddings_query)
