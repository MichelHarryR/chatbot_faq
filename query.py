import streamlit as st
import bs4
from dotenv import load_dotenv
from langchain import hub
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#fichier personnel
from utils import format_qa_pair, format_qa_pairs

from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

import os
# Définir le chemin du répertoire chroma_db relatif au script
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
# Créer le répertoire s'il n'existe pas
os.makedirs(persist_directory, exist_ok=True)

#load_dotenv()

# LLM : instanciation du modele
llm = ChatOpenAI(model='gpt-4o', temperature=1.2, openai_api_key=st.secrets["OPENAI_API_KEY"])

# Initialisation de la liste pour stocker tous les documents
docs = []

# Liste des URLs à charger
urls = [
    "https://www.insi.mg/",
    "https://www.insi.mg/reseaux-et-systeme/",
    "https://www.insi.mg/developpement/",
    "https://www.insi.mg/formation-longue/",
    "https://www.insi.mg/formation-certifiante/",
    "https://www.insi.mg/ia-et-data-science/",
    "https://www.insi.mg/reseaux-et-systeme-universite/",
    "https://www.insi.mg/integration-et-genie-logiciel/",
    "https://www.insi.mg/langage-r/",
    "https://www.insi.mg/python/",
    "https://www.insi.mg/data-engineer/",
    "https://www.insi.mg/machine-laerning/",
    "https://www.insi.mg/deep-learning/",
    "https://www.insi.mg/power-bi-et-tableaux/",
    "https://www.insi.mg/data-analytics/",
    "https://www.insi.mg/data-science/",
    "https://www.insi.mg/intelligence-artificielle/",
    "https://www.insi.mg/architectures-reseaux-et-systemes-cfp/",
    "https://www.insi.mg/developpement-web-cfp/",
    "https://www.insi.mg/architecture-et-cybersecurite-en-reseaux-systemes/",
    "https://www.insi.mg/ingenerie-en-inteligence-artificielle-et-data-science/",
    "https://www.insi.mg/managment-systeme-dinformation/",
    "https://www.insi.mg/condition-dadmission/",
    "https://www.insi.mg/dossier-a-fournir/"
]

# Charger chaque URL et ajouter les documents à la liste pour l'INSI
i = 0
for url in urls:
    loader = WebBaseLoader(url)  # Créer un loader pour chaque URL
    docs.extend(loader.load())     # Charger et ajouter les documents à la liste
    i += 1
    if (i > 2):
        break

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

#file_name = os.path.join('./', 'ia.txt')
#docs = load_document(file_name) #utilisation du fichier ia.txt

# Split du document en 512 morceau avec 50 caractere de chevauchement
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, 
    chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Index and load embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=3072, openai_api_key=st.secrets["OPENAI_API_KEY"]),
                                    persist_directory=persist_directory)

st.session_state.vs = vectorstore

# Create the vector store
k = 10 #nombre de k-voisin
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': k}) #on utilise le type de recherche mmr au lieu de similarity

st.write("Retriever OKk")

# 1. DECOMPOSITION - CI DESSOUS LE CONTENU DU PROMPT QUI VA EFFECTUER LA DECOMPOSITION DE LA QUESTION PRIMITIF; ICI ON VA GENERER 3 QUESTION BIEN REFORMULER
template = """You are a helpful assistant trained to generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
The context always speaks of INSI which is an institute specialized in the IT field. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)


def generate_sub_questions(query):
    """ generate sub questions based on user query"""
    # Chain
    generate_queries_decomposition = (
        prompt_decomposition
        | llm 
        | StrOutputParser()
        | (lambda x: x.split("\n")) #cette indication de retourner le resultat en tableau de 3 element et que ce tableau est disponible en sortie
    )
    # Run
    sub_questions = generate_queries_decomposition.invoke({"question":query})
    questions_str = "\n".join(sub_questions)
    return sub_questions
    
     
   
      


# 2. ANSWER SUBQUESTIONS RECURSIVELY 
template = """Here is the question you need to answer:

\n --- \n {sub_question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {sub_question}
"""
prompt_qa = ChatPromptTemplate.from_template(template)


def generate_qa_pairs(sub_questions):
    """ Demandez au LLM de générer une paire de questions et de réponse en fonction de la requête utilisateur d'origine """
    q_a_pairs = "" #variable pair de Q/A
    
    #pour chaque sous question
    for sub_question in sub_questions:
        #creation de la chain
        generate_qa = (
            {
                "context": itemgetter("sub_question") | retriever,
                "sub_question": itemgetter("sub_question"), 
                "q_a_pairs": itemgetter("q_a_pairs")
            }
            | prompt_qa
            | llm
            | StrOutputParser()
        )
        answer = generate_qa.invoke({"sub_question": sub_question, "q_a_pairs" : q_a_pairs})
        #creation du paire Q/A
        q_a_pair = format_qa_pair(question=sub_question,answer=answer)
        q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair
    

# 3. REPONDRE 1 PAR 1 LES SOUS QUESTIONS GENERER AFIN DE GENERER LE CONTEXTE FINAL

#st.write("Prompt RAG debut")

# RAG prompt = https://smith.langchain.com/hub/rlm/rag-prompt
prompt_rag = hub.pull("rlm/rag-prompt") #ceci utilise un HUB de prompt specialisé pour le RAG

#st.write(prompt_rag)
#st.write("Prompt RAG OK")

def retrieve_and_rag(prompt_rag, sub_questions):
    """RAG on each sub-question"""
    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_question) # recupere le document basé sur le subquestion parmis les document retrieved sur l'embedding
        
        answer_chain = (
            prompt_rag
            | llm
            | StrOutputParser()
        )
        answer = answer_chain.invoke({"question": sub_question, "context": retrieved_docs})
        rag_results.append(answer)
    return rag_results, sub_questions
    

# SUMMARIZE AND ANSWER 

# Prompt
template = """Here is a set of Q+A pairs:

{context}

to use these to synthesize an answer to the question: {question} \n
Short and concise sentences not exceeding 50 words. \n
If the question contains 'Bonjour' or 'Hello' or other greeting then return 'Bonjour, qu\'est ce que je peux faire pour vous ?'. \n
If the question contains a polite phrase such as 'goodbye', 'thank you' , 'Au revoir', 'Bye' or other then return 'Au revoir et à bientôt'. \n
If the question contains 'Merci' or other then return 'Je vous en prie'. \n
If you do not know the answer to a question then you direct the person to contact INSI directly. \n
If the question has nothing to do with INSI then you guide the person by asking them questions only about INSI. \n
All answers are always based on the language used in the question.
"""

prompt = ChatPromptTemplate.from_template(template)


# Query
def query(query):
    
    mots_a_verifier = ["bye","au revoir"]
    
    if 'bonjour' in query.lower():
        final_answer = "Bonjour, qu'est ce que je peux faire pour vous ?"
        return final_answer
    elif 'merci' in query.lower():
        final_answer = "Je vous en prie"
        return final_answer
    elif any(mot.lower() in query.lower() for mot in mots_a_verifier):
        final_answer = "Au revoir et à bientôt"
        return final_answer
    else:
          
        #creation de sous question par rapport au question principale de l'utilisateur pour augmenter l'accuracy de la reponse / recherche
        sub_questions = generate_sub_questions(query)     
        #repondre 1 à 1 à tous les sub question avec un prompt rag
        answers, questions = retrieve_and_rag(prompt_rag=prompt_rag,sub_questions=sub_questions)
        #creation du context finale
        context = format_qa_pairs(questions,answers)
        
        final_rag_chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        
        final_answer = final_rag_chain.invoke({"question":query, "context":context})
        
        return final_answer 
    
    
