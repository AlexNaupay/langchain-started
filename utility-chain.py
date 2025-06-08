import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

chat_mini = ChatOpenAI(
    model='gpt-4o-mini',
    n=1,
    temperature=0.3
)

# Download and store PDF
url = 'https://cseweb.ucsd.edu/classes/wi22/cse127-a/scribenotes/14-pubkeycrypto-notes.pdf'
response = requests.get(url)
with open('public_key_cryptography.pdf', 'wb') as f:
    f.write(response.content)

# Load PDF
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

# Save on Chroma DB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(data, embeddings)

# Summarize every page and then a final summarize
# summarize_map_reduce = load_summarize_chain(
#     chat_mini,
#     chain_type="map_reduce"
# )
# response = summarize_map_reduce.run(data)
# print(response)


plantilla = """Escribe un resumen bien chido del siguiente rollo:

{text}

RESUMEN CORTO CON SLANG MEXICANO:"""

prompt = PromptTemplate(
    template=plantilla,
    input_variables=["text"]
)

# stuff=Limit to prompt limit
cadena_que_resume_con_slang = load_summarize_chain(
    llm=chat_mini,
    chain_type="stuff",
    prompt=prompt,
    # verbose=True
)

# response = cadena_que_resume_con_slang.run(data[:2]) # Deprecated
response = cadena_que_resume_con_slang.invoke(data[:2])
print(response['output_text'])


# response = chat_mini.invoke("Cómo puedo lograr una clase más interactiva para estudiantes virtuales?")
# print(response)

