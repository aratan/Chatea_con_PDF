import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType
import sentence_transformers
import arxiv

# Cargamos variables de entorno
load_dotenv()
huggingfacehub_api_token = os.getenv("api_token")

# Creamos el objeto LLM y lo configuramos
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.6, "max_new_tokens":500})

# Cargamos un documento PDF de manera online
print("Cargamos PDF")
loader = OnlinePDFLoader("https://arxiv.org/pdf/1911.01547.pdf")
document = loader.load()

# Troceamos el documento para que entre en el modelo
print("Troceamos el documento en chunks")
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
documents = text_splitter.split_documents(document)

# Generamos embeddings de cada documento
print("Generamos embeddings de cada documento")
embeddings = HuggingFaceEmbeddings()
query_result = embeddings.embed_query(documents[0].page_content)

# Los guardamos e indexamos en una base de datos vectorial
print("Creamos la base de datos vectorial")
vectorstore = Chroma.from_documents(documents, embeddings)

# Creamos un modelo conversacional de recuperación de documentos
print("Inicializamos el modelo conversacional")
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

# Realizamos una consulta de ejemplo
print("Ejemplo de consulta")
chat_history = []
query = "¿Cuál es la definición de inteligencia?"
result = qa({"question": query, "chat_history": chat_history})
print("Respuesta:")
print(result["answer"])

'''
# Cargamos herramientas y corremos un agente de ejemplo
print("Cargamos herramientas y corremos un agente")
tools = load_tools(["arxiv"])
agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent_chain.run("¿De qué trata el artículo Sobre la medida de la inteligencia, de François Chollet?")
'''