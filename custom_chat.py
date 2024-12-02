# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
# CODE UNIQUEMENT EN PROD SUR STREAMLIT
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
#import sqlite3 #UNIQUEMENT EN PROD SUR STREAMLIT
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
# Définir le chemin du répertoire chroma_db relatif au script
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
# Créer le répertoire s'il n'existe pas
os.makedirs(persist_directory, exist_ok=True)



# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=3072, openai_api_key=st.secrets["OPENAI_API_KEY"])  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model='gpt-4o', temperature=1.2, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': k}) #similarity
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    answer = chain.invoke(q)
    #st.write(answer['source_documents'] )
    return answer['result'] 


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


def stream_data():
    for word in REPONSE.split(" "):
        yield word + " "
        time.sleep(0.02)


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    #from dotenv import load_dotenv, find_dotenv
    #load_dotenv(find_dotenv(), override=True)

    #api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
    gauche , milieu, droite = st.columns([0.1,0.03,0.1])
    
    with milieu:
        st.image("https://www.insi.mg/wp-content/uploads/2022/07/cropped-Modiff-B.png", use_container_width=True)
    
    gauche1 , milieu1, droite1 = st.columns([1,6,1])
    with milieu1: 
        st.subheader("Discutez avec l'IA : Votre Assistant à l'INSI")
    
    #st.image('img.png')
    
    chunk_size = 512
    k = 10
    file_name = os.path.join('./', 'ia.txt')
    data = load_document(file_name)
    chunks = chunk_data(data, chunk_size=chunk_size)
    
    tokens, embedding_cost = calculate_embedding_cost(chunks)
    
    # creating the embeddings and returning the Chroma vector store
    vector_store = create_embeddings(chunks)

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vector_store
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # composant pour la question de l'utilisateur
    if q := st.chat_input("Posez votre question à propos de l'INSI ?"):
        
         # Ajout du message utilisateur dans l'historique
        st.session_state.messages.append({"role": "user", "content": q})
        
        # Affichage du message utilisateur en markdown dans l'interface
        with st.chat_message("user"):
            st.markdown(q)
        
        standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                        "If you can't answer then return `Vous pouvez vous adresser à l'INSI Ambanidia pour avoir plus de precision`." \
                        "If you can't answer then Give the reason in French and return a pertinent question" \
                        "If the text you received as input contains `Bonjour` then return Bonjour, qu'est ce que je peux faire pour vous ?"
                            
        
        q = f"{q} {standard_answer}"
        
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            
            #st.write(chunks)
            
            answer = ask_and_get_answer(vector_store, q, k)

            REPONSE = answer
            
            # Recuperation du Bot INSI
            with st.chat_message("assistant", avatar='https://www.insi.mg/wp-content/uploads/2022/07/cropped-Modiff-B.png'):
                response = st.write_stream(stream_data)
            
            #st.divider()

            # Ajout de la reponse du BOT INSI dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            #st.write(st.session_state)
            
# run the app: streamlit run ./chat_with_documents.py

