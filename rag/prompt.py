import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run():
    print("Retrieving...")

    # OpenAIEmbeddings initializes an embeddings object that will convert text into vectors.
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "Can we add stations in mapbox? How?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})

    print(result.content)

    vectorstore = PineconeVectorStore(
        index_name="medium",
        embedding=embeddings,
    )

    # Retrieval
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    result = retrival_chain.invoke(input={"input": query})

    print(result)

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template=template)
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)

    print(res)
