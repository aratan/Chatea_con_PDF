import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub
load_dotenv()

huggingfacehub_api_token = os.getenv("api_token")

from langchain.text_splitter import CharacterTextSplitter

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.6, "max_new_tokens":500})


#
from langchain import PromptTemplate, LLMChain
# fin prueba pizza

# Cargamos fichero Online
print("Cargamos pdf")
from langchain.document_loaders import OnlinePDFLoader

loader = OnlinePDFLoader("https://arxiv.org/pdf/1911.01547.pdf")
document = loader.load()

# print("tamaño del doc: " + str(len(document[0].page_content)) )

# Troceamos el documento para que entre en el modelo
from langchain.text_splitter import CharacterTextSplitter
# print("troceamos *chunks* los datos del pdf para que los ingiera el modelo")
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

# texts = text_splitter.split_text(raw_text)
documents = text_splitter.split_documents(document)
# print("han generado 211 documentos de una longitud aproximada de 1024 tokens con un solapamiento de 64 tokens entre ellos para evitar que se pierda información.")
# print(len(documents))

# prueba si se ve el doc
# print(documents[10].page_content)
# prueba si se ve el doc
# print(documents[11].page_content)

# print("Cargamos los transformes y arxiv")
import sentence_transformers
import arxiv

# 
from langchain.embeddings import HuggingFaceEmbeddings
print("Empezamos con los Embedding...")
embeddings = HuggingFaceEmbeddings()

query_result = embeddings.embed_query(documents[0].page_content)
# print("Para saber qué trozo de texto le tendremos que pasar al modelo en el prompt, generaremos embeddings de cada documento, una representación numérica en forma de vector.")
# print(query_result)

# print("Estos embedding serán guardados e indexados en una base de datos vectorial, lo cual nos permitirá una búsqueda y extracción eficiente de documentos pasando otro embedding como consulta. Como puedes imaginar, el objetivo será el de recuperar aquellos documentos más similares al prompt, los cuales (supuestamente), contendrán la información que buscamos. Chroma como base de datos vectorial.")
from langchain.vectorstores import Chroma
#print("Cargamos vectores")
vectorstore = Chroma.from_documents(documents, embeddings)

# base de datos vectorial
import chromadb
# Chatear
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

chat_history = []
query = "Cual es la definición de inteligencia?"
result = qa({"question": query, "chat_history": chat_history})
print("Respuesta:")
print(result["answer"])
# prueba
# result['source_documents'][0].page_content

print("**** hasta aqui sin errores ****")






'''
from langchain.agents import load_tools, initialize_agent, AgentType
print("Usamos agentes")
tools = load_tools(
    ["arxiv"],
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Agente
print("Corremos agente")
agent_chain.run("¿De qué trata el artículo Sobre la medida de la inteligencia, de François Chollet?")
'''


