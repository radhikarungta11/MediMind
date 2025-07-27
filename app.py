from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

app = Flask(__name__)
load_dotenv()

# Load environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load HF client
hf_client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HUGGINGFACEHUB_API_TOKEN
)

# Set up embeddings and retriever
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def build_prompt(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
System: You are a helpful medical assistant.
Always provide detailed explanations and recommend multiple options where possible.
Your answers should be comprehensive, informative, and easy to understand.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.

{context}

Human: {query}
Medical Assistant:
"""
    return prompt


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    query = request.form["msg"]
    print("User Input:", query)

    docs = retriever.invoke(query)
    prompt = build_prompt(query, docs)

    response = hf_client.text_generation(
        prompt=prompt,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        stop_sequences=["Human:"]
    )

    print("Response:", response)
    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# from langchain_community.llms import HuggingFaceHub
# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# # OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# embeddings = download_hugging_face_embeddings()


# index_name = "medicalbot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# # llm = OpenAI(temperature=0.4, max_tokens=500)
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     model_kwargs={"temperature": 0.4, "max_new_tokens": 1024}
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     print("User Input:", msg)
#     response = rag_chain.invoke({"input": msg})
#     print("Response:", response["answer"])
#     return str(response["answer"])    




# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8001, debug= True)
# # from flask import Flask, render_template, jsonify, request
# # from src.helper import download_hugging_face_embeddings
# # from langchain_pinecone import PineconeVectorStore
# # from langchain_huggingface import HuggingFaceEndpoint  # Changed this import
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# # from dotenv import load_dotenv
# # from src.prompt import *
# # import os

# # app = Flask(__name__)
# # load_dotenv()

# # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# # HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')  # New environment variable

# # os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# # embeddings = download_hugging_face_embeddings()
# # index_name = "medicalbot"

# # # Embed each chunk and upsert the embeddings into your Pinecone index.
# # docsearch = PineconeVectorStore.from_existing_index(
# #     index_name=index_name,
# #     embedding=embeddings
# # )

# # retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":1})

# # Replace OpenAI with HuggingFaceEndpoint
# # llm = HuggingFaceEndpoint(
# #     endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",  # You can try different models
# #     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
# #     max_length=200,
# #     temperature=0.4
# # )
# # llm = HuggingFaceEndpoint(
# #     endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
# #     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
# #     max_new_tokens=250,  # Changed parameter name and value
# #     temperature=0.4
# # )
# # llm = HuggingFaceEndpoint(
# #     endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
# #     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
# #     model_kwargs={
# #         "temperature": 0.4,
# #         "max_new_tokens": 250
# #     }
# # )

# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         ("system", system_prompt),
# #         ("human", "{input}"),
# #     ]
# # )

# # question_answer_chain = create_stuff_documents_chain(llm, prompt)
# # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # @app.route("/")
# # def index():
# #     return render_template('chat.html')

# # @app.route("/get", methods=["GET", "POST"])
# # def chat():
# #     msg = request.form["msg"]
# #     print(msg)
    
# #     try:
# #         response = rag_chain.invoke({"input": msg})
# #         print("Response:", response["answer"])
# #         return str(response["answer"])
# #     except Exception as e:
# #         print(f"Error: {e}")
# #         return "Sorry, I encountered an error processing your request. Please try again."

# # if __name__ == '__main__':
# #     app.run(host="0.0.0.0", port=8000, debug=True)

