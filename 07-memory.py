import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Integrate with openai
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    n=1,
    temperature=0.3
)

messages = [
    SystemMessage(content="Eres un asistente en un Call Center de reparación de lavadoras."),
    HumanMessage(content="Cómo estás? Necesito ayuda."),
    AIMessage(content="Estoy bien, gracias. En qué puedo ayudar?"),
    HumanMessage(content="Quiero reparar mi lavadora.")
]

res = llm.invoke(messages)

print(res.content)

# Append el res a nuestra serie de mensajes
messages.append(res)

# Agregamos un nuevo mensaje del humano
messages.append(
    HumanMessage(
        content="Ningún botón funciona"
    )
)

# send to chat-gpt
res = llm(messages)
