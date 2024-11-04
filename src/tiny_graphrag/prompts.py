LOCAL_SEARCH = """
Extract key entities from this query separated by newlines: {query}
"""

COMMUNITY_SUMMARY = """
Summarize the following entities and relations from a graph in a paragraph that describes their overall content. Include title and summary.
{community}
"""

COMMUNITY_COMBINE = """
Summarize these related points into a single coherent paragraph: {combined}
"""

LOCAL_SEARCH_RESPONSE = """
Answer the query using only the provided context. Be specific and concise.
"""

GLOBAL_SEARCH_COMMUNITY = """
Answer the query based only on the provided community summary. If the summary doesn't contain relevant information, say 'No relevant information found.'
"""

GLOBAL_SEARCH_COMBINE = """
Combine the provided answers into a single coherent response that fully addresses the query.
"""

NAIVE_SEARCH_RESPONSE = """
Answer the query using only the provided context. Be specific and concise. If the context doesn't contain relevant information, say so and do not make up an answer.
"""

LOCAL_SEARCH_CONTEXT = """
Relevant Entities: {entities}

Relationships:
{relationships}

Supporting Text:
{text_chunks}
"""
