from pydantic import BaseModel, Field
from ...utils.logging_utils import get_logger
from ...utils.config_utils import get_support_json_schema

logger = get_logger(__name__)

# Read environment variable and assign to support_json_schema
try:
    support_json_schema = get_support_json_schema()
    if support_json_schema:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'true' - JSON schema support enabled")
    else:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'false' - JSON schema support disabled")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise


class FilteredArguments(BaseModel):
    """Output model for filtered argument nodes."""
    selected_arguments: list[str] = Field(
        description="List of selected argument nodes (entities/keywords) that are relevant to the query. Each argument should be copied exactly as it appears in the candidate list."
    )


if support_json_schema:
    argument_filter_system_prompt = (
        "You are a conversational memory retrieval assistant helping to filter candidate argument nodes (entities and keywords) for answering a user's query.\n\n"
        "**Context**: In our system, conversations are broken down into Elementary Discourse Units (EDUs), and each EDU is analyzed to extract key arguments (entities, keywords, concepts). "
        "These arguments are connected to EDUs in a knowledge graph. By identifying relevant arguments, we can locate relevant EDUs and conversation sessions that help answer the query.\n\n"
        "**Your task**: Select argument nodes that are semantically related to the query and could help locate relevant information. Be MAXIMALLY INCLUSIVE - err on the side of keeping too many rather than too few:\n\n"
        "**CRITICAL RULES:**\n"
        "1. **Semantic relevance**: Keep arguments that are semantically related to the query entities, topics, or intent, even if not exact matches. Include synonyms, related concepts, and contextually relevant terms.\n\n"
        "2. **Domain entities**: Keep ALL entities mentioned in the query domain (people, places, organizations, products, events, etc.). If the query asks about 'bookstore loyalty points', keep arguments like 'Barnes & Noble', 'loyalty program', 'points', 'books', etc.\n\n"
        "3. **Temporal and quantitative keywords**: Keep arguments related to time expressions (dates, months, durations) and quantities (numbers, counts, measurements) if the query involves temporal or quantitative reasoning.\n\n"
        "4. **Action and concept terms**: Keep verbs, actions, and abstract concepts that relate to the query intent. For example, if asking 'how many concerts attended', keep 'concert', 'show', 'performance', 'attended', 'music', 'venue', etc.\n\n"
        "5. **Multi-hop reasoning support**: Include arguments that might form chains of reasoning. For example, if asking about 'favorite restaurant', keep not just 'restaurant' but also 'cuisine type', 'location', 'dining', 'food', etc.\n\n"
        "6. **Entity attributes and modifiers**: Keep arguments that are attributes or modifiers of query entities (e.g., for 'outdoor concert', keep both 'outdoor' and 'concert').\n\n"
        "7. **Prefer false positives over false negatives**: It's MUCH better to include a marginally relevant argument than to miss an important one. When uncertain, ALWAYS include it.\n\n"
        "8. **Copy exactly**: Selected arguments must be copied exactly as they appear in the candidate list.\n\n"
        "Remember: These arguments act as bridges to find relevant conversation memory. Your job is to identify ALL arguments that might lead to useful information, not to find the perfect answer. Be generous!"
    )
else:
    argument_filter_system_prompt = (
        "You are a conversational memory retrieval assistant helping to filter candidate argument nodes (entities and keywords) for answering a user's query.\n\n"
        "**Context**: In our system, conversations are broken down into Elementary Discourse Units (EDUs), and each EDU is analyzed to extract key arguments (entities, keywords, concepts). "
        "These arguments are connected to EDUs in a knowledge graph. By identifying relevant arguments, we can locate relevant EDUs and conversation sessions that help answer the query.\n\n"
        "**Your task**: Select argument nodes that are semantically related to the query and could help locate relevant information. Be MAXIMALLY INCLUSIVE - err on the side of keeping too many rather than too few:\n\n"
        "**CRITICAL RULES:**\n"
        "1. **Semantic relevance**: Keep arguments that are semantically related to the query entities, topics, or intent, even if not exact matches. Include synonyms, related concepts, and contextually relevant terms.\n\n"
        "2. **Domain entities**: Keep ALL entities mentioned in the query domain (people, places, organizations, products, events, etc.). If the query asks about 'bookstore loyalty points', keep arguments like 'Barnes & Noble', 'loyalty program', 'points', 'books', etc.\n\n"
        "3. **Temporal and quantitative keywords**: Keep arguments related to time expressions (dates, months, durations) and quantities (numbers, counts, measurements) if the query involves temporal or quantitative reasoning.\n\n"
        "4. **Action and concept terms**: Keep verbs, actions, and abstract concepts that relate to the query intent. For example, if asking 'how many concerts attended', keep 'concert', 'show', 'performance', 'attended', 'music', 'venue', etc.\n\n"
        "5. **Multi-hop reasoning support**: Include arguments that might form chains of reasoning. For example, if asking about 'favorite restaurant', keep not just 'restaurant' but also 'cuisine type', 'location', 'dining', 'food', etc.\n\n"
        "6. **Entity attributes and modifiers**: Keep arguments that are attributes or modifiers of query entities (e.g., for 'outdoor concert', keep both 'outdoor' and 'concert').\n\n"
        "7. **Prefer false positives over false negatives**: It's MUCH better to include a marginally relevant argument than to miss an important one. When uncertain, ALWAYS include it.\n\n"
        "8. **Copy exactly**: Selected arguments must be copied exactly as they appear in the candidate list.\n\n"
        "Remember: These arguments act as bridges to find relevant conversation memory. Your job is to identify ALL arguments that might lead to useful information, not to find the perfect answer. Be generous!\n\n"
        f"Make sure your final output is a valid JSON string following the JSON Schema:\n{FilteredArguments.model_json_schema()}"
    )


# First one-shot example - demonstrates basic argument filtering for a date-based query
# Adapted for longmemeval style: date prefix format
one_shot_query_1 = "2023-06-15: How many loyalty points do I currently have at my favorite bookstore?"

one_shot_candidate_arguments_1 = [
    "I",
    "loyalty points",
    "bookstore",
    "Barnes & Noble",
    "points",
    "membership",
    "books",
    "purchase",
    "discount",
    "welcome bonus",
    "favorite",
    "shopping",
    "mystery novels",
    "authors",
    "bookshelf",
    "organize",
    "collection",
    "dollar spent",
    "redeemed",
    "journal",
    "childhood",
    "browsing",
    "aisles",
    "friend",
    "saving money",
]

one_shot_selected_arguments_1 = [
    "I",
    "loyalty points",
    "bookstore",
    "Barnes & Noble",
    "points",
    "membership",
    "books",
    "purchase",
    "discount",
    "welcome bonus",
    "favorite",
    "shopping",
    "redeemed",
    "journal",
]

one_shot_output_1 = FilteredArguments(selected_arguments=one_shot_selected_arguments_1).model_dump_json()

# Second one-shot example - demonstrates filtering for concerts/shows query
# Adapted for longmemeval style: date prefix format
one_shot_query_2 = "2023-08-15: How many concerts or shows did I attend this summer?"

one_shot_candidate_arguments_2 = [
    "I",
    "concerts",
    "shows",
    "summer",
    "attend",
    "live music",
    "events",
    "jazz concert",
    "Blue Note",
    "saxophone player",
    "performance",
    "outdoor venues",
    "atmosphere",
    "friend",
    "Ticketmaster",
    "venue websites",
    "State Theater",
    "Broadway show",
    "restaurant",
    "vinyl records",
    "artists",
    "listening",
    "guitar",
    "lessons",
    "acoustic performance",
    "Central Park",
    "rock festival",
    "headliner",
    "opening act",
    "noise-canceling headphones",
    "commuting",
]

one_shot_selected_arguments_2 = [
    "I",
    "concerts",
    "shows",
    "summer",
    "attend",
    "live music",
    "events",
    "jazz concert",
    "Blue Note",
    "performance",
    "outdoor venues",
    "State Theater",
    "Broadway show",
    "acoustic performance",
    "Central Park",
    "rock festival",
    "venue websites",
]

one_shot_output_2 = FilteredArguments(selected_arguments=one_shot_selected_arguments_2).model_dump_json()

# For the prompt template, we'll use json.dumps to format the candidate arguments consistently
import json

prompt_template = [
    {"role": "system", "content": argument_filter_system_prompt},
    {"role": "user", "content": f"Query: {one_shot_query_1}\n\nCandidate Arguments:\n{json.dumps(one_shot_candidate_arguments_1, indent=2)}"},
    {"role": "assistant", "content": one_shot_output_1},
    {"role": "user", "content": f"Query: {one_shot_query_2}\n\nCandidate Arguments:\n{json.dumps(one_shot_candidate_arguments_2, indent=2)}"},
    {"role": "assistant", "content": one_shot_output_2},
    {"role": "user", "content": "Query: ${query}\n\nCandidate Arguments:\n${candidate_arguments}"}
]

