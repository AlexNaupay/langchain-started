from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("./public_key_cryptography.pdf")
# loader = PyPDFLoader('/home/alexh/Desktop/boletin_ubinas_202416.pdf')

data = loader.load()
content = data[0].page_content
print(content)
print(len(content))
print(content.strip())

replaced = content.replace('•', '')

if len(replaced) < 50:
    print('Not Readable PDF')
else:
    print(content)

