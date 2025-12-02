import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI # the original OpenAI GPT inference class
from .openai_gpt_batch import CacheOpenAI as BatchCacheOpenAI
from .base import BaseLLM

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.openie_mode in ["edu_based_ee_online", "edu_based_contextual_ee_online"]: # override the original OpenAI GPT inference class to accommodate advanced memory structing
        return BatchCacheOpenAI.from_experiment_config(config)

    return CacheOpenAI.from_experiment_config(config)
    