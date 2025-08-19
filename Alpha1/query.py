from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

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
    llama_answer: str
    mistral_answer: str


def retrieve(state: AlphaState):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def llama_node(state: AlphaState):
    llama = Ollama(model="llama3")
    llama_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are LLaMA 3. Use the provided context to answer the question concisely.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
    answer = llama.invoke(llama_prompt.format(context=state["context"], question=state["question"]))

    return {**state, "llama_answer": answer}


#graph mapping
def mistral_node(state: AlphaState):
    mistral = Ollama(model="mistral")
    mistral_prompt = PromptTemplate(
        input_variables=["llama_answer"],
        template="""
        You are Mistral. Translate the following answer to Victorian English.

        Original Answer:
        {llama_answer}

        Improved Answer:
        """
    )
    final = mistral.invoke(mistral_prompt.format(llama_answer=state["llama_answer"]))
    return {**state, "mistral_answer": final}

graph = StateGraph(AlphaState)
graph.add_node("retrieve", retrieve)
graph.add_node("llama", llama_node)
graph.add_node("mistral", mistral_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "llama")
graph.add_edge("llama", "mistral")
graph.add_edge("mistral", END)
app = graph.compile()

def askLlamaAndMistral(query)  :
    result = app.invoke({"question": query})
    print("\n Final Answer:\n", result["mistral_answer"])
    return result["mistral_answer"]

