import tiny_graphrag
from tiny_graphrag import store_document, QueryEngine, init_db

# Initialize the database first
engine = init_db("postgresql://admin:admin@localhost:5432/tiny-graphrag")

# Define custom entity types
geography_entity_types = [
    "ProjectName",
    "Organization",
    "Person",
    "ContractDate"
]

# # Define custom relation types with constraints
# geography_relation_types = {
#     "capital of": {
#         "allowed_head": ["City"],
#         "allowed_tail": ["Country"]
#     },
#     "flows through": {
#         "allowed_head": ["River"],
#         "allowed_tail": ["City", "Country"]
#     },
#     "located in": {
#         "allowed_head": ["City", "Landmark"],
#         "allowed_tail": ["Country"]
#     }
# }

# Store document with custom types
doc_id, graph_path = store_document(
    filepath="data/project_text.txt",
    title="AMENDMENT NUMBER 15 TO THE ARIZONA NUCLEAR POWER PROJECT PARTICIPATION AGREEMENT",
    engine=engine,
    entity_types=geography_entity_types
)
# Create query engine with the database connection
query_engine = QueryEngine(engine)

# Local search
result = query_engine.local_search(
    query="What is the name of the project in this agreement?",
    graph_path=graph_path
)

# Local search
result = query_engine.local_search(
    query="What are the main entities in this agreement?",
    graph_path=graph_path
)


# Global search
result = query_engine.global_search(
    query="What are the main themes of this document?",
    doc_id=doc_id
)

# Naive RAG
result = query_engine.naive_search(
    query="What is the name of the project in this agreement?"
)
