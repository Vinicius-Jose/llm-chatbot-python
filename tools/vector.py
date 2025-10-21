import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from prompts import retrieval_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="moviePlots",
    node_label="Movie",
    text_node_property="plot",
    embedding_node_property="plotEmbedding",  # (6)
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
""",
)

# Create the retriever
retriever = neo4jvector.as_retriever()


# Create the chain
question_answer_chain = create_stuff_documents_chain(llm, retrieval_prompt)
plot_retriever = create_retrieval_chain(retriever, question_answer_chain)


# Create a function to call the chain
def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})
