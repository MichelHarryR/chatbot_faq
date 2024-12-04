# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
# CODE UNIQUEMENT EN PROD SUR STREAMLIT
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
#import sqlite3 #UNIQUEMENT EN PROD SUR STREAMLIT //
import time

#chargement du core LMM / CHATBOT
from query import query


def stream_data():
    for word in REPONSE.split(" "):
        yield word + " "
        time.sleep(0.02)


if __name__ == "__main__":
    import os

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    gauche , milieu, droite = st.columns([0.1,0.03,0.1])
    
    with milieu:
        st.image("./public/logo_insi.png", use_container_width=True)
    
    gauche1 , milieu1, droite1 = st.columns([1,6,1])
    with milieu1: 
        st.subheader("Discutez avec l'IA : Votre Assistant à l'INSI")
        
    # creating the embeddings and returning the Chroma vector store
    #vector_store = vectorstore #from query

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    #st.session_state.vs = vector_store
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message['role'] == "assistant":
            with st.chat_message(message["role"], avatar='./public/logo_insi.png'):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    
    # composant pour la question de l'utilisateur
    if q := st.chat_input("Posez votre question à propos de l'INSI ?"):
        
         # Ajout du message utilisateur dans l'historique
        st.session_state.messages.append({"role": "user", "content": q})
        
        # Affichage du message utilisateur en markdown dans l'interface
        with st.chat_message("user"):
            st.markdown(q)
        
        st.write(st.session_state)
       
        #vector_store = st.session_state.vs
        
        #recuperation de la reponse
        answer = query(q)

        REPONSE = answer
        
        # Recuperation du Bot INSI
        with st.chat_message("assistant", avatar='./public/logo_insi.png'):
            response = st.write_stream(stream_data)
        
        # Ajout de la reponse du BOT INSI dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
            