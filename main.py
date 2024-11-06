import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.chat_models import ChatOpenAI

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st
import tempfile
from neo4j import GraphDatabase


def main():
    # st.set_page_config(layout="wide", page_title="Graphy v1", page_icon=":graph:")
    # st.sidebar.image("logo.png", use_column_width=True)
    # with st.sidebar.expander("Expand Me"):
    #     st.markdown(
    #         """
    # This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and perform queries using natural language.
    # It leverages LangChain and OpenAI's GPT models to generate Cypher queries that interact with the Neo4j database in real-time.
    # """
    #     )
    # st.title("Graphy: Realtime GraphRAG App")

    load_dotenv()

    # if "OPENAI_API_KEY" not in st.session_state:
    #     st.sidebar.subheader("OpenAI API Key")
    #     openai_api_key = st.sidebar.text_input(
    #         "Enter your OpenAI API Key:", type="password"
    #     )
    #     if openai_api_key:
    #         os.environ["OPENAI_API_KEY"] = openai_api_key
    #         st.session_state["OPENAI_API_KEY"] = openai_api_key
    #         st.sidebar.success("OpenAI API Key set successfully.")
    #         embeddings = OpenAIEmbeddings()
    #         llm = ChatOpenAI(
    #             model_name="gpt-4o"
    #         )  # Use model that supports function calling
    #         st.session_state["embeddings"] = embeddings
    #         st.session_state["llm"] = llm
    # else:
    #     embeddings = st.session_state["embeddings"]
    #     llm = st.session_state["llm"]

    api_key = os.getenv("OPENAI_API_KEY")
    llm_model_name = os.getenv("LLM_MODEL")
    llm_model = ChatOpenAI(model_name=llm_model_name)

    embedding_model_name = os.getenv("EMBEDDING_MODEL")

    if embedding_model_name == "OpenAI":
        embedding_model = OpenAIEmbeddings()
    else:
        raise ValueError("Unsupported embedding model specified in .env")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database,
    )

    # if "neo4j_connected" not in st.session_state:
    #     st.sidebar.subheader("Connect to Neo4j Database")
    #     neo4j_url = st.sidebar.text_input(
    #         "Neo4j URL:", value="neo4j+s://<your-neo4j-url>"
    #     )
    #     neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
    #     neo4j_password = st.sidebar.text_input("Neo4j Password:", type="password")
    #     connect_button = st.sidebar.button("Connect")
    #     if connect_button and neo4j_password:
    #         try:
    #             graph = Neo4jGraph(
    #                 url=neo4j_url, username=neo4j_username, password=neo4j_password
    #             )
    #             st.session_state["graph"] = graph
    #             st.session_state["neo4j_connected"] = True
    #             # Store connection parameters for later use
    #             st.session_state["neo4j_url"] = neo4j_url
    #             st.session_state["neo4j_username"] = neo4j_username
    #             st.session_state["neo4j_password"] = neo4j_password
    #             st.sidebar.success("Connected to Neo4j database.")
    #         except Exception as e:
    #             st.error(f"Failed to connect to Neo4j: {e}")
    # else:
    #     graph = st.session_state["graph"]
    #     neo4j_url = st.session_state["neo4j_url"]
    #     neo4j_username = st.session_state["neo4j_username"]
    #     neo4j_password = st.session_state["neo4j_password"]
    # if graph is not None:
    #     # File uploader
    #     uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

    #     if uploaded_file is not None and "qa" not in st.session_state:
    #         with st.spinner("Processing the PDF..."):
    #             # Save uploaded file to temporary file
    #             with tempfile.NamedTemporaryFile(
    #                 delete=False, suffix=".pdf"
    #             ) as tmp_file:
    #                 tmp_file.write(uploaded_file.read())
    #                 tmp_file_path = tmp_file.name
    # Load and split the PDF
    pdf_file_path = os.getenv("PDF_FILE_PATH")

    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    docs = text_splitter.split_documents(pages)

    lc_docs = []
    for doc in docs:
        lc_docs.append(
            Document(
                page_content=doc.page_content.replace("\n", ""),
                metadata={"source": pdf_file_path},
            )
        )

    # Clear the graph database
    # cypher = """
    #     MATCH (n)
    #     DETACH DELETE n;
    # """
    # graph.query(cypher)

    # Define allowed nodes and relationships
    # allowed_nodes = [
    #     "Patient",
    #     "Disease",
    #     "Medication",
    #     "Test",
    #     "Symptom",
    #     "Doctor",
    # ]
    allowed_nodes = ["*"]

    # allowed_relationships = [
    #     "HAS_DISEASE",
    #     "TAKES_MEDICATION",
    #     "UNDERWENT_TEST",
    #     "HAS_SYMPTOM",
    #     "TREATED_BY",
    # ]
    allowed_relationships = ["*"]

    # Transform documents into graph documents
    transformer = LLMGraphTransformer(
        llm=llm_model,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=False,
        relationship_properties=False,
    )

    graph_documents = transformer.convert_to_graph_documents(lc_docs)
    graph.add_graph_documents(graph_documents, include_source=True)

    # Use the stored connection parameters
    index = Neo4jVector.from_existing_graph(
        embedding=embedding_model,
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database="neo4j",
        node_label="*",  # Adjust node_label as needed
        text_node_properties=["*"],
        embedding_node_property="embedding",
        index_name="vector_index",
        keyword_index_name="entity_index",
        search_type="hybrid",
    )

    # st.success(f"{uploaded_file.name} preparation is complete.")

    # Retrieve the graph schema
    schema = graph.get_schema

    # Set up the QA chain
    template = """
    Task: Generate a Cypher statement to query the graph database.

    Instructions:
    Use only relationship types and properties provided in schema.
    Do not use other relationship types or properties that are not provided.

    schema:
    {schema}

    Note: Do not include explanations or apologies in your answers.
    Do not answer questions that ask anything other than creating Cypher statements.
    Do not include any text other than generated Cypher statements.

    Question: {question}"""

    question_prompt = PromptTemplate(
        template=template, input_variables=["schema", "question"]
    )

    qa = GraphCypherQAChain.from_llm(
        llm=llm_model,
        graph=graph,
        cypher_prompt=question_prompt,
        verbose=True,
        allow_dangerous_requests=True,
    )

    while True:
        question = input("질문을 입력해주세요 (종료하려면 '종료' 입력): ")

        # '종료' 입력 시 반복 종료
        if question.lower() == "종료":
            print("프로그램을 종료합니다.")
            break

        # 질문을 qa 객체에 전달하여 응답을 받습니다.
        response = qa({"query": question})

        # 응답의 필요한 부분을 출력합니다.
        print(response)  # 'result' 키에 응답이 있다고 가정합니다.


# st.session_state["qa"] = qa
# else:
#     st.warning("Please connect to the Neo4j database before you can upload a PDF.")

# if "qa" in st.session_state:
#     st.subheader("Ask a Question")
#     with st.form(key="question_form"):
#         question = st.text_input("Enter your question:")
#         submit_button = st.form_submit_button(label="Submit")

#     if submit_button and question:
#         with st.spinner("Generating answer..."):
#             res = st.session_state["qa"].invoke({"query": question})
#             st.write("\n**Answer:**\n" + res["result"])

if __name__ == "__main__":
    main()

    # Text file import
    # with open(
    #     "/Users/gimtaegi/graphrag/ragtest/input/merged_dataset.txt",
    #     "r",
    #     encoding="UTF-8",
    # ) as file:
    #     content = file.read()
    #     # print(content)

#     pdf_file_path = os.getenv("PDF_FILE_PATH")

#     # Load and split the PDF
#     loader = PyPDFLoader(pdf_file_path)
#     pages = loader.load_and_split()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
#     docs = text_splitter.split_documents(pages)

#     lc_docs = []
#     for doc in docs:
#         lc_docs.append(
#             Document(
#                 page_content=doc.page_content.replace("\n", ""),
#                 metadata={"source": pdf_file_path},
#             )
#         )

#     # Clear the graph database
#     cypher = """
#         MATCH (n)
#         DETACH DELETE n;
#     """
#     graph.query(cypher)

#     # Define allowed nodes and relationships
#     allowed_nodes = ["*"]

#     allowed_relationships = ["*"]

#     transformer = LLMGraphTransformer(
#         llm=llm_model,
#         allowed_nodes=allowed_nodes,
#         allowed_relationships=allowed_relationships,
#         node_properties=False,  # default : False
#         relationship_properties=False,  # default : False
#     )

#     graph_documents = transformer.convert_to_graph_documents(lc_docs)
#     graph.add_graph_documents(graph_documents, include_source=True)

#     # Use the stored connection parameters
#     index = Neo4jVector.from_existing_graph(
#         embedding=embedding_model,
#         url=neo4j_uri,
#         username=neo4j_username,
#         password=neo4j_password,
#         database=neo4j_database,
#         node_label="Patient",  # Adjust node_label as needed
#         text_node_properties=["id", "text"],
#         embedding_node_property="embedding",
#         index_name="vector_index",
#         keyword_index_name="entity_index",
#         search_type="hybrid",
#     )

#     schema = graph.get_schema

#     # Set up the QA chain
#     template = """
#     Task: Generate a Cypher statement to query the graph database.

#     Instructions:
#     Use only relationship types and properties provided in schema.
#     Do not use other relationship types or properties that are not provided.

#     schema:
#     {schema}

#     Note: Do not include explanations or apologies in your answers.
#     Do not answer questions that ask anything other than creating Cypher statements.
#     Do not include any text other than generated Cypher statements.

#     Question: {question}"""

#     question_prompt = PromptTemplate(
#         template=template, input_variables=["schema", "question"]
#     )

#     qa = GraphCypherQAChain.from_llm(
#         llm=llm_model,
#         graph=graph,
#         cypher_prompt=question_prompt,
#         verbose=True,
#         allow_dangerous_requests=True,
#     )

#     # 질문할 내용을 정의합니다.
#     question = "수강신청 일정 알려줘"

#     # 질문을 qa 객체에 전달하여 응답을 받습니다.
#     response = qa({"query": question})

#     # 응답을 출력합니다.
#     print(response)


# if __name__ == "__main__":
#     main()
