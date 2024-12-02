# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

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
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4o', temperature=1)
    
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': k}) #similarity
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
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
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #st.image('img.png')
    st.subheader('Bienvenu sur le CHATBOT DE INSI')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        #api_key = st.text_input('OpenAI API Key:', type='password')
        #if api_key:
        #    os.environ['OPENAI_API_KEY'] = api_key

        # upload de fichier BDD statique pour le FAQ
        #uploaded_file = st.file_uploader('Ajouter le fichier pour le FAQ :', type=['pdf', 'docx', 'txt'])

        # Definition du chunck size
        chunk_size = 512
        #chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # Definition du nombre de K-voisin pour la recherche de similarité
        k = 5
        #k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        #add_data = st.button('Add Data', on_click=clear_history)

        #if uploaded_file and add_data: # if the user browsed a file
            #with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
        #bytes_data = uploaded_file.read()
    file_name = os.path.join('./', 'ia.txt')
    #with open(file_name, 'wb') as f:
        #f.write(bytes_data)

    data = load_document(file_name)
    chunks = chunk_data(data, chunk_size=chunk_size)
    #st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
    
    tokens, embedding_cost = calculate_embedding_cost(chunks)
    #st.write(f'Embedding cost: ${embedding_cost:.4f}')

    # creating the embeddings and returning the Chroma vector store
    vector_store = create_embeddings(chunks)

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vector_store
    #st.success('File uploaded, chunked and embedded successfully.')

    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # user's question text input widget
    #q = st.text_input('Ask a question about the content of your file:')
    if q := st.chat_input("Posez votre question à propos de l'INSI ?"):
        
         # Ajout du message utilisateur dans l'historique
        st.session_state.messages.append({"role": "user", "content": q})
        
        # Affichage du message utilisateur en markdown dans l'interface
        with st.chat_message("user"):
            st.markdown(q)
        
        standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                        "If you can't answer then return `Vous pouvez vous adresser à l'INSI Ambanidia pour avoir plus de precision`." \
                        "If the text you received as input contains `Bonjour` then return Bonjour, qu'est ce que je peux faire pour vous ?"
                            
        
        q = f"{q} {standard_answer}"
        
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            
            answer = ask_and_get_answer(vector_store, q, k)

            REPONSE = answer
            
            # Recuperation du Bot INSI
            with st.chat_message("assistant"):
                response = st.write_stream(stream_data)
            
            st.divider()

            # Ajout de la reponse du BOT INSI dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": response})
            
# run the app: streamlit run ./chat_with_documents.py

