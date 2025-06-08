from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("public_key_cryptography.pdf")
# loader = PyPDFLoader('/home/alexh/Desktop/boletin_ubinas_202416.pdf')

data = loader.load()
content = data[1].page_content  # 1 = page number 2
print(content)
print(len(content))
print(content.strip())

replaced = content.replace('•', '')

if len(replaced) < 50:
    print('Not Readable PDF')
else:
    print(content)

