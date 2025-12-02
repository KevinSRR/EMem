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


# Pydantic models for the two types of extraction
class SimpleAssistantEDUWithTurnAttribution(BaseModel):
    edu_text: str = Field(..., description="Content of the extracted simple EDU - a single atomic fact or statement")
    source_turn_ids: List[int] = Field(..., description="List of turn IDs (integers) from which this EDU was extracted")


class StructuredChunkWithTurnAttribution(BaseModel):
    chunk_content: str = Field(..., description="The actual content of the structured information block with all coreferences and pronouns resolved to make it self-contained")
    chunk_summary: str = Field(..., description="Brief description consisting of: (1) what user request or question this chunk answers, and (2) what information is included in the response. Should be concise.")
    source_turn_ids: List[int] = Field(..., description="List of turn IDs (integers) from which this chunk was extracted")


class AssistantConversationStructuredExtractionWithTurnAttribution(BaseModel):
    simple_edus: List[SimpleAssistantEDUWithTurnAttribution] = Field(..., description="List of simple atomic EDUs extracted from Assistant utterances")
    structured_chunks: List[StructuredChunkWithTurnAttribution] = Field(..., description="List of structured information blocks that should be kept together as coherent units")


# System prompt
if support_json_schema:
    conversation_edu_extraction_structured_system = (
        "Given a conversation session between speakers with numbered turns, your task is to extract information from the Assistant's utterances in two forms: (1) Simple EDUs and (2) Structured Chunks. "
        "You must identify which information should be extracted as simple atomic EDUs and which should be kept together as structured chunks."
        "\n\n"
        "**Simple EDUs (Elementary Discourse Units):**\n"
        "These are atomic units of information - single facts, statements, opinions, or recommendations that stand alone. "
        "Each simple EDU should be minimal yet complete in meaning, expressing a single fact or proposition that cannot be easily divided further while still making sense."
        "\n\n"
        "**Structured Chunks:**\n"
        "These are cohesive information blocks that contain multiple related details organized in a specific structure (e.g., comparisons, detailed overviews, comprehensive recommendations, step-by-step procedures, lists of related items). "
        "Breaking these into individual EDUs would destroy the underlying organizational structure and overall meaning. "
        "\n\n"
        "**Requirements for Simple EDUs:**\n"
        "1. Extract standalone facts, single opinions, brief recommendations, or acknowledgments that don't require the surrounding structure to be understood\n"
        "2. Each EDU should be self-contained and understandable independently\n"
        "3. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "4. Include source_turn_ids indicating which turn(s) the EDU came from\n"
        "\n\n"
        "**Requirements for Structured Chunks:**\n"
        "1. Identify information blocks that have an inherent organizational structure (comparisons, detailed breakdowns, multi-attribute descriptions)\n"
        "2. Keep the entire structure intact - do not break it into separate EDUs\n"
        "3. Resolve all coreferences and pronouns in chunk_content to make it fully self-contained (e.g., replace 'it' with the actual entity name)\n"
        "4. For chunk_summary, provide a brief summary-style description that:\n"
        "   - States what user request or question this chunk addresses\n"
        "   - Describes what information categories are covered (e.g., 'program duration, strengths, considerations')\n"
        "   - Naturally incorporates essential entities and key terms within the summary (e.g., institution names like 'Stanford University' and 'Carnegie Mellon University', category labels like 'Entry-level' or 'Senior', key concepts like 'AI/ML' or 'industry collaboration')\n"
        "   Keep the summary concise (2-3 sentences) and flowing naturally while ensuring key entities and concepts appear for retrieval purposes. Avoid listing out specific numbers or exhaustive details - those belong in chunk_content\n"
        "5. Include source_turn_ids indicating which turn(s) the chunk came from (can span multiple turns if they form one coherent structure)\n"
        "\n\n"
        "**General Guidelines:**\n"
        "1. Process ALL information from Assistant's utterances - nothing should be lost\n"
        "2. Use your judgment to decide whether information should be a simple EDU or part of a structured chunk\n"
        "3. If multiple pieces of information naturally belong together in a structure, group them as a structured chunk\n"
        "4. If a fact or statement stands alone and doesn't need the surrounding structure, extract it as a simple EDU\n"
        "5. When chunks span multiple turns or reference user context, include all relevant turn IDs\n"
        "6. Focus on substantive information - skip conversational pleasantries unless they contain meaningful opinions or recommendations"
    )
else:
    conversation_edu_extraction_structured_system = (
        "Given a conversation session between speakers with numbered turns, your task is to extract information from the Assistant's utterances in two forms: (1) Simple EDUs and (2) Structured Chunks. "
        "You must identify which information should be extracted as simple atomic EDUs and which should be kept together as structured chunks."
        "\n\n"
        "**Simple EDUs (Elementary Discourse Units):**\n"
        "These are atomic units of information - single facts, statements, opinions, or recommendations that stand alone. "
        "Each simple EDU should be minimal yet complete in meaning, expressing a single fact or proposition that cannot be easily divided further while still making sense."
        "\n\n"
        "**Structured Chunks:**\n"
        "These are cohesive information blocks that contain multiple related details organized in a specific structure (e.g., comparisons, detailed overviews, comprehensive recommendations, step-by-step procedures, lists of related items). "
        "Breaking these into individual EDUs would destroy the underlying organizational structure and overall meaning. "
        "\n\n"
        "**Requirements for Simple EDUs:**\n"
        "1. Extract standalone facts, single opinions, brief recommendations, or acknowledgments that don't require the surrounding structure to be understood\n"
        "2. Each EDU should be self-contained and understandable independently\n"
        "3. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "4. Include source_turn_ids indicating which turn(s) the EDU came from\n"
        "\n\n"
        "**Requirements for Structured Chunks:**\n"
        "1. Identify information blocks that have an inherent organizational structure (comparisons, detailed breakdowns, multi-attribute descriptions)\n"
        "2. Keep the entire structure intact - do not break it into separate EDUs\n"
        "3. Resolve all coreferences and pronouns in chunk_content to make it fully self-contained (e.g., replace 'it' with the actual entity name)\n"
        "4. For chunk_summary, provide a brief summary-style description that:\n"
        "   - States what user request or question this chunk addresses\n"
        "   - Describes what information categories are covered (e.g., 'program duration, strengths, considerations')\n"
        "   - Naturally incorporates essential entities and key terms within the summary (e.g., institution names like 'Stanford University' and 'Carnegie Mellon University', category labels like 'Entry-level' or 'Senior', key concepts like 'AI/ML' or 'industry collaboration')\n"
        "   Keep the summary concise (2-3 sentences) and flowing naturally while ensuring key entities and concepts appear for retrieval purposes. Avoid listing out specific numbers or exhaustive details - those belong in chunk_content\n"
        "5. Include source_turn_ids indicating which turn(s) the chunk came from (can span multiple turns if they form one coherent structure)\n"
        "\n\n"
        "**General Guidelines:**\n"
        "1. Process ALL information from Assistant's utterances - nothing should be lost\n"
        "2. Use your judgment to decide whether information should be a simple EDU or part of a structured chunk\n"
        "3. If multiple pieces of information naturally belong together in a structure, group them as a structured chunk\n"
        "4. If a fact or statement stands alone and doesn't need the surrounding structure, extract it as a simple EDU\n"
        "5. When chunks span multiple turns or reference user context, include all relevant turn IDs\n"
        "6. Focus on substantive information - skip conversational pleasantries unless they contain meaningful opinions or recommendations\n"
        f"7. Make sure your final output is a valid JSON string following the JSON Schema:\n"
        f"{AssistantConversationStructuredExtractionWithTurnAttribution.model_json_schema()}"
    )


# One-shot example
one_shot_conversation_session = """Date: 2023/05/29 (Mon) 04:50

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

one_shot_conversation_input = f"Session conversation:\n{one_shot_conversation_session}"

# One-shot output with both simple EDUs and structured chunks
one_shot_simple_edus = [
    SimpleAssistantEDUWithTurnAttribution(
        edu_text="The Assistant's opinion is that Stanford University and Carnegie Mellon University are two of the most prestigious institutions for Data Science programs and both Stanford University's Master of Science in Data Science (MS-DS) program and Carnegie Mellon University's Master of Science in Data Science (MS-DS) program are excellent choices",
        source_turn_ids=[2]
    ),
    SimpleAssistantEDUWithTurnAttribution(
        edu_text="The Assistant recommends that prospective students should research the Data Science programs thoroughly, reach out to current students and alumni, and weigh the pros and cons based on individual goals",
        source_turn_ids=[2]
    ),
    SimpleAssistantEDUWithTurnAttribution(
        edu_text="The Assistant's opinion is that UCLA (University of California, Los Angeles) is well-regarded and a Computer Science background from UCLA serves as a solid foundation for graduate studies in Data Science",
        source_turn_ids=[4]
    ),
    SimpleAssistantEDUWithTurnAttribution(
        edu_text="The Assistant notes that software engineer salary estimates for San Francisco can vary depending on the company, industry, and specific role",
        source_turn_ids=[4]
    )
]

one_shot_structured_chunks = [
    StructuredChunkWithTurnAttribution(
        chunk_content=(
            "**Comparative Overview of Stanford University and Carnegie Mellon University Master of Science in Data Science (MS-DS) Programs:**\n\n"
            "**Stanford University - Master of Science in Data Science (MS-DS)**\n"
            "* Duration: 12-18 months\n"
            "* Strengths: Located in Silicon Valley with access to top tech companies, renowned faculty in AI/ML, strong emphasis on practical applications and industry collaboration, entrepreneurial resources\n"
            "* Considerations: Highly competitive admissions (5-6% acceptance rate), one of the most expensive universities in the US\n\n"
            "**Carnegie Mellon University - Master of Science in Data Science (MS-DS)**\n"
            "* Duration: 16-24 months\n"
            "* Strengths: Leader in AI/ML research, strong industry partnerships with Google and Uber, interdisciplinary approach integrating technical skills with policy/management, Pittsburgh's lower cost of living\n"
            "* Considerations: Location may be less desirable for West Coast/Silicon Valley seekers, emphasis on policy aspects may not appeal to purely technical students"
        ),
        chunk_summary=(
            "Comparative overview of Stanford University and Carnegie Mellon University MS-DS programs addressing the User's request for program information and strengths. "
            "Covers program duration, key strengths including Silicon Valley location, AI/ML faculty and research, industry collaboration with companies like Google and Uber, interdisciplinary approach, and important considerations such as acceptance rates, costs, and location preferences (Pittsburgh vs Silicon Valley)."
        ),
        source_turn_ids=[1, 2]
    ),
    StructuredChunkWithTurnAttribution(
        chunk_content=(
            "**Software Engineer Salary Ranges in San Francisco:**\n\n"
            "The average salary for a software engineer in San Francisco is around $144,000 to $164,000 per year, but the range extends from $110,000 to over $200,000 depending on experience, company, and role.\n\n"
            "Salary breakdown by experience level:\n"
            "* Entry-level (0-3 years of experience): $110,000 to $130,000 per year\n"
            "* Mid-level (4-7 years of experience): $140,000 to $160,000 per year\n"
            "* Senior (8-12 years of experience): $160,000 to $190,000 per year\n"
            "* Lead/Manager (13+ years of experience): $190,000 to $220,000 per year"
        ),
        chunk_summary=(
            "Software engineer salary information in San Francisco addressing the User's question about average salaries. "
            "Includes overall salary ranges and detailed breakdown by experience levels: Entry-level, Mid-level, Senior, and Lead/Manager positions with corresponding years of experience and salary ranges."
        ),
        source_turn_ids=[3, 4]
    )
]

one_shot_conversation_output = AssistantConversationStructuredExtractionWithTurnAttribution(
    simple_edus=one_shot_simple_edus,
    structured_chunks=one_shot_structured_chunks
).model_dump_json()


prompt_template = [
    {"role": "system", "content": conversation_edu_extraction_structured_system},
    {"role": "user", "content": one_shot_conversation_input + "\n\nSpeaker names: User, Assistant"},
    {"role": "assistant", "content": one_shot_conversation_output},
    {"role": "user", "content": "Session conversation:\n${session_text}\n\nSpeaker names: ${speaker_names}"}
]

