from pydantic import BaseModel, Field
from typing import List

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


# New models for turn attribution (used by batch_conversation_edu_extraction_without_context_user_edus)
class ConversationEDUWithTurnAttribution(BaseModel):
    edu_text: str = Field(..., description="Content of the extracted EDU - should be self-contained and informative")
    source_turn_ids: List[int] = Field(..., description="List of turn IDs (integers) from which this EDU was extracted")


class ConversationEDUExtractionWithTurnAttribution(BaseModel):
    edus: List[ConversationEDUWithTurnAttribution]




# ===== Turn Attribution Version =====
# System prompt for turn attribution
if support_json_schema:
    conversation_edu_extraction_turn_attribution_system = (
        "Given a conversation session between speakers with numbered turns, your task is to extract Elementary Discourse Units (EDUs) from the User's utterances only and attribute each EDU to the specific turn IDs it was extracted from. "
        "Each EDU should be a short span of text that is minimal yet complete in meaning, expressing a single fact, event, or "
        "proposition and be atomic (not easily divisible further while still making sense). "
        "It is critical that you preserve all information from the User's utterances - no detail should be lost in the extraction process."
        "\n"
        "Requirements for User-side Conversation EDUs with Turn Attribution:\n"
        "1. Extract EDUs only from the User's utterances - do not extract information that is solely from the Assistant's responses without the User referencing or engaging with it\n"
        "2. When the User references or engages with information from the Assistant's utterances (e.g., 'those programs you mentioned', 'the salary ranges you provided'), you must resolve these references by incorporating the specific details from the Assistant's utterances to make the EDU self-contained\n"
        "3. Each EDU should be a self-contained unit of meaning that can be understood independently, without requiring context from other EDUs or the conversation\n"
        "4. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "5. The extracted EDUs must include all the information and all the details expressed by the User, regardless of how minor the detail may seem. The extracted EDUs should be able to completely replace all of the User's utterances.\n"
        "6. For each EDU, you must provide the source_turn_ids field containing a list of turn ID integers from which the EDU was extracted or referenced (e.g., [1], [3, 5], etc.)\n"
        "7. EDUs can span multiple User turns if they represent the same factual unit or evolving state - in such cases, include all relevant turn IDs\n"
        "8. Focus on extracting facts, events, preferences, decisions, concerns, and substantive information rather than conversational pleasantries\n"
        "9. Infer and add complete temporal context where needed for clarity\n"
        "10. Pay special attention to capturing all details, facts, preferences, background information, personal attributes, and states mentioned by the User - these are critical and must never be omitted"
    )
else:
    conversation_edu_extraction_turn_attribution_system = (
        "Given a conversation session between speakers with numbered turns, your task is to extract Elementary Discourse Units (EDUs) from the User's utterances only and attribute each EDU to the specific turn IDs it was extracted from. "
        "Each EDU should be a short span of text that is minimal yet complete in meaning, expressing a single fact, event, or "
        "proposition and be atomic (not easily divisible further while still making sense). "
        "It is critical that you preserve all information from the User's utterances - no detail should be lost in the extraction process."
        "\n"
        "Requirements for User-side Conversation EDUs with Turn Attribution:\n"
        "1. Extract EDUs only from the User's utterances - do not extract information that is solely from the Assistant's responses without the User referencing or engaging with it\n"
        "2. When the User references or engages with information from the Assistant's utterances (e.g., 'those programs you mentioned', 'the salary ranges you provided'), you must resolve these references by incorporating the specific details from the Assistant's utterances to make the EDU self-contained\n"
        "3. Each EDU should be a self-contained unit of meaning that can be understood independently, without requiring context from other EDUs or the conversation\n"
        "4. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "5. The extracted EDUs must include all the information and all the details expressed by the User, regardless of how minor the detail may seem. The extracted EDUs should be able to completely replace all of the User's utterances.\n"
        "6. For each EDU, you must provide the source_turn_ids field containing a list of turn ID integers from which the EDU was extracted or referenced (e.g., [1], [3, 5], etc.)\n"
        "7. EDUs can span multiple User turns if they represent the same factual unit or evolving state - in such cases, include all relevant turn IDs\n"
        "8. Focus on extracting facts, events, preferences, decisions, concerns, and substantive information rather than conversational pleasantries\n"
        "9. Infer and add complete temporal context where needed for clarity\n"
        "10. Pay special attention to capturing all details, facts, preferences, background information, personal attributes, and states mentioned by the User - these are critical and must never be omitted\n"
        f"11. Make sure your final output is a valid JSON string following the JSON Schema:\n"
        f"{ConversationEDUExtractionWithTurnAttribution.model_json_schema()}"
    )


# One-shot example for conversation EDU extraction with turn IDs (User-side only)
one_shot_conversation_session_turn_attribution = """Date: 2023/05/29 (Mon) 04:50

Turn 1:
User: I'm considering pursuing a Master's degree in Data Science and I've narrowed down my options to Stanford and Carnegie Mellon. Can you give me an overview of the programs and their strengths?

Turn 2:
Assistant: You've shortlisted two of the most prestigious institutions for Data Science programs. Here's a brief overview:

**Stanford University - Master of Science in Data Science (MS-DS)**
* Duration: 12-18 months
* Strengths: Located in Silicon Valley with access to top tech companies, renowned faculty in AI/ML, strong emphasis on practical applications and industry collaboration, entrepreneurial resources
* Considerations: Highly competitive admissions (5-6% acceptance rate), one of the most expensive universities in the US

**Carnegie Mellon University - Master of Science in Data Science (MS-DS)**
* Duration: 16-24 months  
* Strengths: Leader in AI/ML research, strong industry partnerships with Google and Uber, interdisciplinary approach integrating technical skills with policy/management, Pittsburgh's lower cost of living
* Considerations: Location may be less desirable for West Coast/Silicon Valley seekers, emphasis on policy aspects may not appeal to purely technical students

Both programs are excellent choices. Research thoroughly, reach out to current students and alumni, and weigh pros and cons based on your goals.

Turn 3:
User: I'm leaning towards Stanford, but I'm concerned about the cost. Can you tell me about the average salary of a software engineer in San Francisco? Also, I need to finalize which program to accept by this Friday since that's the admission offer deadline. By the way, I completed my undergraduate degree in Computer Science from UCLA in 2021, which I believe has a strong reputation in the tech industry.

Turn 4:
Assistant: Great that you have a CS background from UCLA - it's well-regarded! For SF software engineer salaries: the average is around $144,000-$164,000 per year, but ranges from $110,000 to over $200,000 depending on experience, company, and role. Entry-level (0-3 years): $110,000-$130,000; Mid-level (4-7 years): $140,000-$160,000; Senior (8-12 years): $160,000-$190,000; Lead/Manager (13+ years): $190,000-$220,000. These estimates vary by company and industry.

Turn 5:
User: Thanks! Those salary ranges you mentioned would definitely help justify the investment. I'm particularly interested in the Silicon Valley location and the industry connections you described for Stanford."""

one_shot_conversation_input_turn_attribution = f"Session conversation:\n{one_shot_conversation_session_turn_attribution}"

one_shot_conversation_edus_turn_attribution = [
    ConversationEDUWithTurnAttribution(
        edu_text="In a conversation on May 29, 2023, the User expressed that the User is considering pursuing a Master's degree in Data Science",
        source_turn_ids=[1]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User has narrowed down Master's degree options to Stanford University and Carnegie Mellon University, but is leaning towards Stanford University after initial consideration",
        source_turn_ids=[1, 3]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User is concerned about the cost of attending Stanford University's Master of Science in Data Science (MS-DS) program",
        source_turn_ids=[3]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User needs to decide between accepting the Stanford University Master of Science in Data Science (MS-DS) program admission offer or the Carnegie Mellon University Master of Science in Data Science (MS-DS) program admission offer by June 2, 2023 (Friday) as the deadline for accepting the admission offers",
        source_turn_ids=[3]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User completed an undergraduate degree in Computer Science from UCLA (University of California, Los Angeles) in 2021",
        source_turn_ids=[3]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User believes UCLA (University of California, Los Angeles) has a strong reputation in the tech industry",
        source_turn_ids=[3]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User believes that software engineer salary ranges in San Francisco (entry-level $110,000-$130,000, mid-level $140,000-$160,000, senior $160,000-$190,000, lead/manager $190,000-$220,000) would help justify the investment in Stanford University's Master of Science in Data Science (MS-DS) program",
        source_turn_ids=[5]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User is particularly interested in Stanford University's Silicon Valley location with access to top tech companies",
        source_turn_ids=[5]
    ),
    ConversationEDUWithTurnAttribution(
        edu_text="The User is particularly interested in Stanford University's industry connections and collaboration opportunities",
        source_turn_ids=[5]
    )
]

one_shot_conversation_output_turn_attribution = ConversationEDUExtractionWithTurnAttribution(edus=one_shot_conversation_edus_turn_attribution).model_dump_json()


prompt_template_turn_attribution = [
    {"role": "system", "content": conversation_edu_extraction_turn_attribution_system},
    {"role": "user", "content": one_shot_conversation_input_turn_attribution + "\n\nSpeaker names: User, Assistant"},
    {"role": "assistant", "content": one_shot_conversation_output_turn_attribution},
    {"role": "user", "content": "Session conversation:\n${session_text}\n\nSpeaker names: ${speaker_names}"}
]

# For this template file, we want to use the turn attribution version by default
# since the filename indicates it's specifically for turn attribution
prompt_template = prompt_template_turn_attribution
