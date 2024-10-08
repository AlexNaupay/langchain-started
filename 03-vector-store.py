from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_pdf_file():
    loader = PyPDFLoader("./public_key_cryptography.pdf")
    data_ = loader.load()
    print("documento cargado")
    return data_


def generate_document_chunks(data, chunk_size, overlap, function):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=function,
    )
    documents = text_splitter.split_documents(data)
    print(documents[0].page_content)
    return documents


def instantiate_embedding_model(model, device):
    # A junio de 2023 no hay modelos Instruct para español
    embedding_instruct = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device}
    )
    print("Tipo de Embedding:")
    print(type(embedding_instruct))
    return embedding_instruct


def generate_vectorstore(documents, embedding_instruct, vectorstore_name):
    vectorstore_chroma = Chroma.from_documents(
        documents=documents,
        embedding=embedding_instruct,
        persist_directory=vectorstore_name
    )
    print("Vector Store generado")
    return vectorstore_chroma


def save_vectorstore(vectorstore):
    vectorstore.persist()
    print("Vector store almacenado con éxito")


def load_vectorstore(vectorstore_name, embedding_instruct):
    vectorstore_chroma = Chroma(
        persist_directory=vectorstore_name,
        embedding_function=embedding_instruct
    )
    print("Vector Store cargado")
    return vectorstore_chroma


if __name__ == '__main__':
    # https://python.langchain.com/v0.2/docs/how_to/embed_text/

    data = load_pdf_file()

    # Generamos un Document con una partición de 500 carácteres y un overlap del 10%
    documents = generate_document_chunks(data, chunk_size=500, overlap=50, function=len)

    # Si es la primera vez que se instancia el modelo entonces este se va a descargar y se va a cargar en el device CUDA
    embedding_model = instantiate_embedding_model(model="hkunlp/instructor-large", device="cpu")

    # Ya teniendo el modelo de embedding entonces podemos generar el Vector Store, le asignamos el siguiente nombre:
    vectorstore_name = "instruct-embeddings-public-crypto"

    # Y lo vamos a generar con la información de `Documents` utilizando el modelo `embedding_model` y nombre
    # `vectorstore_name`
    vectorstore = generate_vectorstore(documents, embedding_model, vectorstore_name)

    # Guardamos el modelo en disco
    save_vectorstore(vectorstore)

    # Cargamos el modelo para trabajar con él
    vectorstore_loaded = load_vectorstore(vectorstore_name, embedding_model)

    # Vamos a hacer una petición que dada la siguiente pregunta obtenga 5 Chunks con información similar
    query = "What is public key cryptography?"
    docs = vectorstore_loaded.similarity_search_with_score(query, k=5)

    # Veamos cuantos documentos generó y un ejemplo del mismo
    print(len(docs))
    print(docs[3])
