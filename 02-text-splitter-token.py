from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

# all_text_list = map(lambda x: x.page_content, data)
all_text = '\n'.join(map(lambda x: x.page_content, data))
# print(all_text)

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=100, chunk_overlap=10
# )
# texts = text_splitter.split_text(all_text)

text_splitter_recursive = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
)
texts = text_splitter_recursive.split_text(all_text)

print(texts[0])
print(f"All Text len = {len(all_text)} letters")
print(f"Split data len each one = {len(texts[0])} letters")
print(f"Splits = {len(texts)}")
