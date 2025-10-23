from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

chat_prompt = ChatPromptTemplate(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

agent_prompt = PromptTemplate.from_template(
    """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
)

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".

Example Cypher Statements:

1. To find who acted in a movie:
```
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {{title: "Movie Title"}})
RETURN p.name, r.role
```

2. To find who directed a movie:
```
MATCH (p:Person)-[r:DIRECTED]->(m:Movie {{title: "Movie Title"}})
RETURN p.name
```

3. How to find how many degrees of separation there are between two people::
```
MATCH path = shortestPath(
  (p1:Person {{name: "Actor 1"}})-[:ACTED_IN|DIRECTED*]-(p2:Person {{name: "Actor 2"}})
)
WITH path, p1, p2, relationships(path) AS rels
RETURN
  p1 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p1.tmdbId }} AS start,
  p2 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p2.tmdbId }} AS end,
  reduce(output = '', i in range(0, length(path)-1) |
    output + CASE
      WHEN i = 0 THEN
       startNode(rels[i]).name + CASE WHEN type(rels[i]) = 'ACTED_IN' THEN ' played '+ rels[i].role +' in 'ELSE ' directed ' END + endNode(rels[i]).title
       ELSE
         ' with '+ startNode(rels[i]).name + ', who '+ CASE WHEN type(rels[i]) = 'ACTED_IN' THEN 'played '+ rels[i].role +' in '
    ELSE 'directed '
      END + endNode(rels[i]).title
      END
  ) AS pathBetweenPeople
```

Schema:
{schema}

Question:
{question}
"""
cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
