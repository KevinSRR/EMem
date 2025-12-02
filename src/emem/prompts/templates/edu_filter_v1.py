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


class FilteredEDUs(BaseModel):
    """Output model for filtered EDUs."""
    selected_edus: list[str] = Field(
        description="List of selected EDUs that are relevant to answering the query. Each EDU should be copied exactly as it appears in the candidate list."
    )


if support_json_schema:
    edu_filter_system_prompt = (
        "You are a conversational memory retrieval assistant helping to filter candidate memory units (EDUs - Elementary Discourse Units) for answering a user's query.\n\n"
        "Your task is to select EDUs that are relevant to answering the query. Be MAXIMALLY INCLUSIVE - err on the side of keeping too many rather than too few:\n\n"
        "**CRITICAL RULES:**\n"
        "1. **Keep EDUs with embedded information**: Even if an EDU contains extra context or discusses multiple topics, KEEP IT if ANY part is relevant to the query. Don't discard EDUs just because they're long or contain additional information.\n\n"
        "2. **All temporal references**: Keep EVERY EDU that mentions dates, times, durations, or events within the query's timeframe, even if buried in longer descriptions.\n\n"
        "3. **All quantitative information**: Keep EVERY EDU containing numbers, counts, amounts, or measurements related to the query domain (e.g., points, items, events, days).\n\n"
        "4. **Direct matches**: Keep ANY EDU that directly mentions entities, actions, or topics from the query (e.g., if query asks about 'items to pick up', keep EDUs mentioning picking up anything).\n\n"
        "5. **Multi-hop reasoning**: Include EDUs that form chains of information. If asked 'how many X', keep ALL EDUs mentioning individual instances of X, even if they don't say 'how many'.\n\n"
        "6. **Historical context**: Keep EDUs about past activities, purchases, visits, or experiences that relate to the query domain.\n\n"
        "7. **Prefer false positives over false negatives**: It's MUCH better to include an irrelevant EDU than to miss a relevant one. When uncertain, ALWAYS include it.\n\n"
        "8. **Copy exactly**: Selected EDUs must be copied exactly as they appear in the candidate list.\n\n"
        "Remember: Your job is NOT to find the perfect answer, but to keep ALL EDUs that MIGHT help answer the query. Be generous!"
    )
else:
    edu_filter_system_prompt = (
        "You are a conversational memory retrieval assistant helping to filter candidate memory units (EDUs - Elementary Discourse Units) for answering a user's query.\n\n"
        "Your task is to select EDUs that are relevant to answering the query. Be MAXIMALLY INCLUSIVE - err on the side of keeping too many rather than too few:\n\n"
        "**CRITICAL RULES:**\n"
        "1. **Keep EDUs with embedded information**: Even if an EDU contains extra context or discusses multiple topics, KEEP IT if ANY part is relevant to the query. Don't discard EDUs just because they're long or contain additional information.\n\n"
        "2. **All temporal references**: Keep EVERY EDU that mentions dates, times, durations, or events within the query's timeframe, even if buried in longer descriptions.\n\n"
        "3. **All quantitative information**: Keep EVERY EDU containing numbers, counts, amounts, or measurements related to the query domain (e.g., points, items, events, days).\n\n"
        "4. **Direct matches**: Keep ANY EDU that directly mentions entities, actions, or topics from the query (e.g., if query asks about 'items to pick up', keep EDUs mentioning picking up anything).\n\n"
        "5. **Multi-hop reasoning**: Include EDUs that form chains of information. If asked 'how many X', keep ALL EDUs mentioning individual instances of X, even if they don't say 'how many'.\n\n"
        "6. **Historical context**: Keep EDUs about past activities, purchases, visits, or experiences that relate to the query domain.\n\n"
        "7. **Prefer false positives over false negatives**: It's MUCH better to include an irrelevant EDU than to miss a relevant one. When uncertain, ALWAYS include it.\n\n"
        "8. **Copy exactly**: Selected EDUs must be copied exactly as they appear in the candidate list.\n\n"
        "Remember: Your job is NOT to find the perfect answer, but to keep ALL EDUs that MIGHT help answer the query. Be generous!\n\n"
        f"Make sure your final output is a valid JSON string following the JSON Schema:\n{FilteredEDUs.model_json_schema()}"
    )


# First one-shot example - demonstrates basic filtering
one_shot_query_1 = "Date of user query: 2023/06/15 (Thu) 14:30\nUser: How many loyalty points do I currently have at my favorite bookstore?"

one_shot_candidate_edus_1 = [
    "On June 10, 2023, the User purchased three books at Barnes & Noble and earned 45 loyalty points, bringing the total to 320 points.",
    "The User enjoys reading mystery novels and has been collecting books from their favorite authors.",
    "On May 15, 2023, the User signed up for the Barnes & Noble membership program and received 50 welcome bonus points.",
    "The User is considering purchasing a new bookshelf to organize their growing book collection.",
    "Barnes & Noble offers 1 point for every dollar spent on books and other items.",
    "The User's favorite bookstore is Barnes & Noble, where they shop regularly for books and gifts.",
    "The User redeemed 75 points for a discount on June 12, 2023, leaving 245 points in their account.",
    "On June 14, 2023, the User purchased a journal at Barnes & Noble for $25 and earned 25 points, bringing the total to 270 points.",
    "The User has been visiting Barnes & Noble since childhood and has many fond memories of browsing the aisles.",
    "The Assistant suggests that loyalty programs are great for saving money on frequent purchases.",
]

one_shot_selected_edus_1 = [
    "On May 15, 2023, the User signed up for the Barnes & Noble membership program and received 50 welcome bonus points.",
    "On June 10, 2023, the User purchased three books at Barnes & Noble and earned 45 loyalty points, bringing the total to 320 points.",
    "Barnes & Noble offers 1 point for every dollar spent on books and other items.",
    "The User's favorite bookstore is Barnes & Noble, where they shop regularly for books and gifts.",
    "The User redeemed 75 points for a discount on June 12, 2023, leaving 245 points in their account.",
    "On June 14, 2023, the User purchased a journal at Barnes & Noble for $25 and earned 25 points, bringing the total to 270 points.",
]

one_shot_output_1 = FilteredEDUs(selected_edus=one_shot_selected_edus_1).model_dump_json()

# Second one-shot example - demonstrates keeping EDUs with embedded relevant information
one_shot_query_2 = "Date of user query: 2023/08/20 (Sun) 16:45\nUser: How many concerts or shows did I attend this summer?"

one_shot_candidate_edus_2 = [
    "The User is planning to attend more live music events and is looking for upcoming concerts in the area.",
    "On June 15, 2023, the User mentioned going to a jazz concert at the Blue Note last night and really enjoying the saxophone player's performance.",
    "The User prefers outdoor venues for concerts because of the better atmosphere and fresh air.",
    "The Assistant suggests checking Ticketmaster and local venue websites for upcoming concert schedules.",
    "On July 4, 2023, the User asked for restaurant recommendations near the State Theater before attending a Broadway show that evening.",
    "The User has been collecting vinyl records from their favorite artists and enjoys listening to them at home.",
    "The User is interested in learning to play the guitar and asked about beginner lessons on July 20, 2023, mentioning they were inspired after seeing a live acoustic performance at Central Park the previous weekend.",
    "The Assistant recommends following local music venues on social media to stay updated about upcoming shows.",
    "On August 10, 2023, the User shared that they went to an outdoor rock festival last Saturday and the headliner was amazing, though it rained during the opening act.",
    "The User is considering buying noise-canceling headphones for listening to music while commuting.",
]

one_shot_selected_edus_2 = [
    "On June 15, 2023, the User mentioned going to a jazz concert at the Blue Note last night and really enjoying the saxophone player's performance.",
    "On July 4, 2023, the User asked for restaurant recommendations near the State Theater before attending a Broadway show that evening.",
    "The User is interested in learning to play the guitar and asked about beginner lessons on July 20, 2023, mentioning they were inspired after seeing a live acoustic performance at Central Park the previous weekend.",
    "On August 10, 2023, the User shared that they went to an outdoor rock festival last Saturday and the headliner was amazing, though it rained during the opening act.",
    "The User prefers outdoor venues for concerts because of the better atmosphere and fresh air.",
    "The User is planning to attend more live music events and is looking for upcoming concerts in the area.",
]

one_shot_output_2 = FilteredEDUs(selected_edus=one_shot_selected_edus_2).model_dump_json()

# For the prompt template, we'll use json.dumps to format the candidate EDUs consistently
import json

prompt_template = [
    {"role": "system", "content": edu_filter_system_prompt},
    {"role": "user", "content": f"Query: {one_shot_query_1}\n\nCandidate EDUs:\n{json.dumps(one_shot_candidate_edus_1, indent=2)}"},
    {"role": "assistant", "content": one_shot_output_1},
    {"role": "user", "content": f"Query: {one_shot_query_2}\n\nCandidate EDUs:\n{json.dumps(one_shot_candidate_edus_2, indent=2)}"},
    {"role": "assistant", "content": one_shot_output_2},
    {"role": "user", "content": "Query: ${query}\n\nCandidate EDUs:\n${candidate_edus}"}
]
