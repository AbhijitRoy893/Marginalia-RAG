# main.py
# RAG query engine: retriever + prompt + Mistral LLM.
# Importable by app.py, and still runnable as a terminal chatbot.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def load_vectorstore(persist_directory="chroma_db"):
    """Load an existing Chroma store from disk (used only in standalone/CLI mode)."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


def get_retriever(vectorstore, k=4, fetch_k=10, lambda_mult=0.5):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def get_llm(model="mistral-small-2506"):
    return ChatMistralAI(model=model)


def answer_question(retriever, llm, query):
    """
    Run one turn of RAG: retrieve context, fill the prompt, call the LLM.
    Returns: (answer_text, source_docs)
    """
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    return response.content, docs


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    print("RAG system created!")
    print("Press 0 to exit")

    while True:
        query = input("\nYou: ")

        if query == "0":
            break

        answer, docs = answer_question(retriever, llm, query)
        print(f"\nAI: {answer}")