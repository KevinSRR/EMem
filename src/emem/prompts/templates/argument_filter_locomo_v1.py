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
        "**Your task**: Intelligently select argument nodes that could lead to information needed to answer the query. Prioritize PRECISION - focus on arguments that are truly relevant to the query intent.\n\n"
        "**REASONING PROCESS:**\n"
        "1. **Understand the query intent**: What information is needed to answer this query?\n"
        "2. **Identify key entities and concepts**: What are the core elements mentioned in the query?\n"
        "3. **Think about information dependencies**: What related entities or attributes would be needed to answer the query?\n"
        "4. **Select relevant arguments**: Choose arguments that could lead to the target information.\n\n"
        "**SELECTION RULES:**\n"
        "1. **Direct query entities**: Keep entities explicitly mentioned in the query (people, places, organizations, objects, events).\n\n"
        "2. **Query-relevant attributes**: Keep attributes, properties, or modifiers that are directly relevant to what the query asks about. For example:\n"
        "   - If asking about 'loyalty points', keep 'points', 'loyalty', 'membership', 'earned', 'redeemed'\n"
        "   - If asking about 'cost comparison', keep cost-related terms for the compared items\n\n"
        "3. **Quantitative and temporal terms**: Keep numbers, counts, dates, or time-related terms when the query involves 'how many', 'when', 'how much', or comparison.\n\n"
        "4. **Domain-specific terms**: Keep domain-specific terminology that relates to the query topic (e.g., for concert queries: 'venue', 'performance', 'show'; for restaurant queries: 'cuisine', 'dining', 'menu').\n\n"
        "5. **Causal/relational terms**: Keep arguments that represent relationships or interactions relevant to the query (e.g., 'purchase', 'attend', 'visit', 'received').\n\n"
        "6. **Arguments with partial relevance**: Keep arguments that contain relevant information EVEN IF they also include irrelevant information. Since argument extraction may not be perfect, an argument might be a list or phrase containing multiple entities where only some are relevant. If ANY part of the argument relates to the query, keep it. Examples:\n"
        "   - 'coffee shop and bookstore' → Keep if query asks about either location\n"
        "   - 'bus fare, taxi cost, and parking fee' → Keep if query asks about any transportation cost\n"
        "   - 'Italian restaurant on Main Street' → Keep even if only 'Italian restaurant' or 'Main Street' is relevant\n\n"
        "**WHAT TO EXCLUDE:**\n"
        "- Arguments that are entirely irrelevant to the query intent with no connection to the information needed\n"
        "- Generic background terms that don't help locate the target information (e.g., for a loyalty points query, exclude 'childhood memories', 'browsing aisles')\n"
        "- Context entities with no involvement in the query target (e.g., for 'Sarah's points', exclude other people who don't affect her point transactions)\n\n"
        "**IMPORTANT**: Do NOT exclude an argument just because it contains some irrelevant information. The argument extraction process is imperfect, so arguments may be compound phrases or lists. Focus on whether the argument CONTAINS relevant information, not whether it's PURELY relevant.\n\n"
        "**BALANCE**: Be precise but not overly restrictive. If an argument could plausibly lead to information needed for the answer, include it. When uncertain about an argument's relevance, consider: 'Would conversations containing this argument likely contain information to answer the query?'\n\n"
        "**Copy exactly**: Selected arguments must be copied exactly as they appear in the candidate list."
    )
else:
    argument_filter_system_prompt = (
        "You are a conversational memory retrieval assistant helping to filter candidate argument nodes (entities and keywords) for answering a user's query.\n\n"
        "**Context**: In our system, conversations are broken down into Elementary Discourse Units (EDUs), and each EDU is analyzed to extract key arguments (entities, keywords, concepts). "
        "These arguments are connected to EDUs in a knowledge graph. By identifying relevant arguments, we can locate relevant EDUs and conversation sessions that help answer the query.\n\n"
        "**Your task**: Intelligently select argument nodes that could lead to information needed to answer the query. Prioritize PRECISION - focus on arguments that are truly relevant to the query intent.\n\n"
        "**REASONING PROCESS:**\n"
        "1. **Understand the query intent**: What information is needed to answer this query?\n"
        "2. **Identify key entities and concepts**: What are the core elements mentioned in the query?\n"
        "3. **Think about information dependencies**: What related entities or attributes would be needed to answer the query?\n"
        "4. **Select relevant arguments**: Choose arguments that could lead to the target information.\n\n"
        "**SELECTION RULES:**\n"
        "1. **Direct query entities**: Keep entities explicitly mentioned in the query (people, places, organizations, objects, events).\n\n"
        "2. **Query-relevant attributes**: Keep attributes, properties, or modifiers that are directly relevant to what the query asks about. For example:\n"
        "   - If asking about 'loyalty points', keep 'points', 'loyalty', 'membership', 'earned', 'redeemed'\n"
        "   - If asking about 'cost comparison', keep cost-related terms for the compared items\n\n"
        "3. **Quantitative and temporal terms**: Keep numbers, counts, dates, or time-related terms when the query involves 'how many', 'when', 'how much', or comparison.\n\n"
        "4. **Domain-specific terms**: Keep domain-specific terminology that relates to the query topic (e.g., for concert queries: 'venue', 'performance', 'show'; for restaurant queries: 'cuisine', 'dining', 'menu').\n\n"
        "5. **Causal/relational terms**: Keep arguments that represent relationships or interactions relevant to the query (e.g., 'purchase', 'attend', 'visit', 'received').\n\n"
        "6. **Arguments with partial relevance**: Keep arguments that contain relevant information EVEN IF they also include irrelevant information. Since argument extraction may not be perfect, an argument might be a list or phrase containing multiple entities where only some are relevant. If ANY part of the argument relates to the query, keep it. Examples:\n"
        "   - 'coffee shop and bookstore' → Keep if query asks about either location\n"
        "   - 'bus fare, taxi cost, and parking fee' → Keep if query asks about any transportation cost\n"
        "   - 'Italian restaurant on Main Street' → Keep even if only 'Italian restaurant' or 'Main Street' is relevant\n\n"
        "**WHAT TO EXCLUDE:**\n"
        "- Arguments that are entirely irrelevant to the query intent with no connection to the information needed\n"
        "- Generic background terms that don't help locate the target information (e.g., for a loyalty points query, exclude 'childhood memories', 'browsing aisles')\n"
        "- Context entities with no involvement in the query target (e.g., for 'Sarah's points', exclude other people who don't affect her point transactions)\n\n"
        "**IMPORTANT**: Do NOT exclude an argument just because it contains some irrelevant information. The argument extraction process is imperfect, so arguments may be compound phrases or lists. Focus on whether the argument CONTAINS relevant information, not whether it's PURELY relevant.\n\n"
        "**BALANCE**: Be precise but not overly restrictive. If an argument could plausibly lead to information needed for the answer, include it. When uncertain about an argument's relevance, consider: 'Would conversations containing this argument likely contain information to answer the query?'\n\n"
        "**Copy exactly**: Selected arguments must be copied exactly as they appear in the candidate list.\n\n"
        f"Make sure your final output is a valid JSON string following the JSON Schema:\n{FilteredArguments.model_json_schema()}"
    )


# First one-shot example - demonstrates basic argument filtering for a bookstore query
# Adapted for LoCoMo style: third-person query without date prefix
one_shot_query_1 = "How many loyalty points does Sarah currently have at her favorite bookstore?"

one_shot_candidate_arguments_1 = [
    "Sarah",
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
    "Emma",
    "saving money",
    "books, journals, and bookmarks",  # Compound argument - contains 'journal' which relates to purchases
    "loyalty points and rewards program",  # Compound argument - both parts relevant
]

one_shot_selected_arguments_1 = [
    "Sarah",
    "loyalty points",
    "bookstore",
    "Barnes & Noble",
    "points",
    "membership",
    "purchase",
    "welcome bonus",
    "redeemed",
    "favorite",
    "books, journals, and bookmarks",  # Keep: contains purchase-related items even though 'bookmarks' may not be directly relevant
    "loyalty points and rewards program",  # Keep: compound argument with relevant parts
]

one_shot_output_1 = FilteredArguments(selected_arguments=one_shot_selected_arguments_1).model_dump_json()

# Second one-shot example - demonstrates filtering for concerts/shows query
# Adapted for LoCoMo style: third-person query without date prefix
one_shot_query_2 = "How many concerts or shows did Marcus attend this summer?"

one_shot_candidate_arguments_2 = [
    "Marcus",
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
    "Lisa",
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
    "concerts, festivals, and theater shows",  # Compound argument - all parts relevant to attending shows
    "tickets and venue information",  # Compound argument - relevant to attending events
]

one_shot_selected_arguments_2 = [
    "Marcus",
    "concerts",
    "shows",
    "summer",
    "attend",
    "live music",
    "events",
    "jazz concert",
    "Blue Note",
    "performance",
    "State Theater",
    "Broadway show",
    "acoustic performance",
    "Central Park",
    "rock festival",
    "concerts, festivals, and theater shows",  # Keep: compound argument with all relevant parts
    "tickets and venue information",  # Keep: compound argument relevant to event attendance
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

