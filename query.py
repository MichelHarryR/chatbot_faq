
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

load_dotenv()

# LLM
llm = ChatOpenAI(model='gpt-4o')

# Initialisation de la liste pour stocker tous les documents
docs = []

# Liste des URLs à charger
urls = [
    "https://www.insi.mg/",
    "https://www.insi.mg/reseaux-et-systeme/",
    "https://www.insi.mg/developpement/",
    "https://www.insi.mg/projet-okile/",
    "https://www.insi.mg/agile-scrum/",
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

# Charger chaque URL et ajouter les documents à la liste
i = 0
for url in urls:
    loader = WebBaseLoader(url)  # Créer un loader pour chaque URL
    docs.extend(loader.load())     # Charger et ajouter les documents à la liste
    i += 1
    if (i > 1):
        break

# À ce stade, docs contient tous les documents chargés des URLs
#print(f"Total documents loaded: {len(docs)}")

#  split documents, create vector store and load embeddings
'''loader = WebBaseLoader(
    web_paths=("https://blog.langchain.dev/reflection-agents/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header section", "article-header__content", "article-header__footer")
        )
    ),
)
blog_docs = loader.load()'''

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Index and load embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

# Create the vector store
retriever = vectorstore.as_retriever()


# 1. DECOMPOSITION - CI DESSOUS LE CONTENU DU PROMPT QUI VA EFFECTUER LA DECOMPOSITION DE LA QUESTION PRIMITIF; ICI ON VA GENERER 3 QUESTION BIEN REFORMULER
template = """You are a helpful assistant trained to generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
The context always speaks of INSI which is an institute specialized in the IT field. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

'''template = """Tu es un assistant entrainé pour generer plusieurs sous question relatif à la question principale.\n
Ton but c'est de diviser la question principale en des sous-questions qui peut etre repondu 1 à 1.\n
Le sujet ne concerne que l'INSI qui est une Institut Specialisé dans le domaine de l'informatique. \n
Genere des requetes multiples rapport a : {question} \n
Output (3 requetes):"""'''

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
    #print(sub_questions)
    questions_str = "\n".join(sub_questions)
    #print(Fore.MAGENTA + "===== SUBQUESTIONS: =====" + Fore.RESET)
    #print(Fore.WHITE + questions_str + Fore.RESET + "\n")
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
    

# 3. ANSWER INDIVIDUALLY

# RAG prompt = https://smith.langchain.com/hub/rlm/rag-prompt
prompt_rag = hub.pull("rlm/rag-prompt") #ceci utilise un HUB de prompt specialisé pour le RAG


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
If the question is a greeting, confirmation or politeness then you should answer accordingly. \n
If you do not know the answer to a question then you direct the person to contact INSI directly. \n
If the question has nothing to do with INSI then you guide the person by asking them questions only about INSI. \n
All answers are always based on the language used in the question.
"""

prompt = ChatPromptTemplate.from_template(template)


# Query
def query(query):
    # generate optimized answer for a given query using the improved subqueries
    question = "What are the main components of an LLM-powered autonomous agent system?"
    queries = [
        "How is context improving AI systems",
        "What are the two main components involved in Basic Reflection",
        "Explain the steps involved in the Reflexion loop"
    ]
    #creation de sous question par rapport au question principale de l'utilisateur
    sub_questions = generate_sub_questions(query)
    #generate_qa_pairs(sub_questions=sub_questions)
    
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
    
    
