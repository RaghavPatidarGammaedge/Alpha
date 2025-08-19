from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
import os



# db connection
vector_store = Chroma(
    persist_directory="chroma_db",
    collection_name="doc",
    embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
)

#Constant info carrier
class AlphaState(TypedDict):
    question: str
    context: List[Document]
    gemini_answer: str
    groq_answer: str


def retrieve(state: AlphaState):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def gemini_node(state: AlphaState):
    gemini = ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"), model="gemini-2.5-flash")
    gemini_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are gemini. Use the provided context to answer the question concisely.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
    answer = gemini.invoke(gemini_prompt.format(context=state["context"], question=state["question"]))
    return {**state, "gemini_answer": answer}


#graph mapping
def groq_node(state: AlphaState):
    groq = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="llama3-8b-8192",)

    groq_prompt = PromptTemplate(
        input_variables=["gemini_answer"],
        template="""
        You are groq. Translate the following answer to Victorian English.
        Critical Instruction:
        1. Give me a clear and display ready response, only the meaningful output. 
        2. Nothing like 'Here is the translated answer in Victorian English:' should be added at any cost in response
        3. Answer in the form of paragraph,characters or character sequences like line changing sequence or * should not be included
        4. All of the response should be in a single paragraph
        
        Original Answer:
        {gemini_answer}

        Improved Answer:
        """
    )
    final = groq.invoke(groq_prompt.format(gemini_answer=state["gemini_answer"]))
    return {**state, "groq_answer": final.content}




graph = StateGraph(AlphaState)
graph.add_node("retrieve", retrieve)
graph.add_node("gemini", gemini_node)
graph.add_node("groq", groq_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "gemini")
graph.add_edge("gemini", "groq")
graph.add_edge("groq", END)
app = graph.compile()

def askGeminiAndGroq(query):
    result = app.invoke({"question": query})
    print("\nFinal Answer:\n", result["groq_answer"])
    return result["groq_answer"]


