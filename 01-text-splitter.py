from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    length_function=len,
    chunk_overlap=100
)

documents = text_splitter.split_documents(data)

print(f"Data len= {len(data)}")
print(f"Split data len = {len(documents)}")
print(documents[0])

