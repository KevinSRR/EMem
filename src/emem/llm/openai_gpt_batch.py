import functools
import inspect
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple, Any, Callable
import re
import httpx
import aiohttp # for making API calls concurrently
import asyncio # for running API calls concurrently
import tiktoken # for counting tokens
import time # for sleeping after rate limit is hit
from dataclasses import dataclass, field # for storing API inputs, outputs, and metadata
import openai
from filelock import FileLock
from openai import OpenAI, AsyncOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed
import unicodedata as ud
import importlib
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import append_to_jsonl
from .base import BaseLLM, LLMConfig
import math

try:
    # v2 & v1 share the same helper
    from pydantic.json import pydantic_encoder
except ImportError:
    pydantic_encoder = lambda v: v   # fallback, shouldn't happen
    

logger = get_logger(__name__)



# ───────────────────────────── encoder ──────────────────────────────
def cache_json_encoder(obj):
    # 1️⃣  instances of BaseModel  → store as dict with a type tag
    if isinstance(obj, BaseModel):
        tag = f"{obj.__module__}.{obj.__class__.__qualname__}"
        return {"__pydantic_instance__": tag, "data": obj.model_dump()}
    
    # 2️⃣  class objects that *inherit* from BaseModel  → store dotted path
    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        return {"__pydantic_class__": f"{obj.__module__}.{obj.__qualname__}"}

    # 2b️⃣  any other class objects (e.g., ModelMetaclass) → store dotted path safely
    if inspect.isclass(obj):
        return {"__class__": f"{obj.__module__}.{obj.__qualname__}"}

    # 3️⃣  let Pydantic deal with datetime, UUID, Path, etc.
    try:
        return pydantic_encoder(obj)
    except TypeError:
        # 4️⃣  ultimate fallback → stringify
        return str(obj)

# ───────────────────────────── decoder ──────────────────────────────
def cache_json_decoder(d):
    # called for every dict that json.loads builds
    if "__pydantic_instance__" in d:
        mod_name, _, cls_name = d["__pydantic_instance__"].rpartition(".")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls.model_validate(d["data"])

    if "__pydantic_class__" in d:
        mod_name, _, cls_name = d["__pydantic_class__"].rpartition(".")
        return getattr(importlib.import_module(mod_name), cls_name)

    # nothing special → leave dict as‑is
    return d

def cache_response(func):
    """
    Same public behaviour:
        - returns (message, metadata, cache_hit)
    Works for both sync *and* async `func`.
    """
    async def _run_blocking(callable_, *args, **kw):
        """Run blocking I/O in the default executor."""
        return await asyncio.to_thread(callable_, *args, **kw)

    async def _wrapper_async(self, *args, **kwargs):
        # ── build cache key (unchanged) ──────────────────────────────────────────
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")
        
        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        
        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,
            "model": kwargs.get("model", gen_params.get("model")),
            "seed": kwargs.get("seed", gen_params.get("seed")),
            "temperature": kwargs.get("temperature", gen_params.get("temperature")),
            "record_in_n_out_w_metadata": kwargs.get("record_in_n_out_w_metadata", True), # be very careful with this variable, which now requires it to be entered in the format of kwargs
        }
        key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        lock_file = self.cache_file_name + ".lock" # the file name of lock, ensure mutual exclusion when accessing concurrently

        # ── try cache hit (blocking → thread) ───────────────────────────────────
        def _read():
            conn = sqlite3.connect(self.cache_file_name)
            try:
                conn.execute("""CREATE TABLE IF NOT EXISTS cache (
                                    key TEXT PRIMARY KEY,
                                    message TEXT,
                                    metadata TEXT
                                )""")
                cur = conn.execute("SELECT message, metadata FROM cache WHERE key = ?",
                                   (key_hash,))
                return cur.fetchone()
            finally:
                conn.close()
        row = await _run_blocking(_read)

        if row:
            message, metadata_str = row
            metadata = json.loads(metadata_str, object_hook=cache_json_decoder)
            # message, metadata = row[0], json.loads(row[1])
            return message, metadata, True          # ← cache hit

        # ── call the real function ──────────────────────────────────────────────
        # Fix: Check if func is async or sync to avoid awaiting a sync function result
        if inspect.iscoroutinefunction(func):
            result = await func(self, *args, **kwargs)
        else:
            # For sync functions, call directly without await
            result = func(self, *args, **kwargs)
        # Handle both 2-tuple and 3-tuple returns from the function
        if len(result) == 2:
            message, metadata = result
        elif len(result) == 3:
            message, metadata, _ = result  # ignore any cache_hit flag from the function
        else:
            raise ValueError(f"Function returned unexpected result length: {len(result)}")
        metadata_str = json.dumps(metadata, default=cache_json_encoder)
        
        
        # ── write cache (blocking → thread) ─────────────────────────────────────
        def _write():
            with FileLock(lock_file):
                with sqlite3.connect(self.cache_file_name) as conn:
                    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
                                        key TEXT PRIMARY KEY,
                                        message TEXT,
                                        metadata TEXT
                                    )""")
                    conn.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?,?,?)",
                                 (key_hash, message, metadata_str))
                    conn.commit()
        await _run_blocking(_write)

        return message, metadata, False             # ← cache miss

    def _wrapper_sync(self, *args, **kwargs):
        # identical to your original code, or if you prefer:
        return asyncio.run(_wrapper_async(self, *args, **kwargs))

    # choose the right one once, keep wraps()
    wrapper = _wrapper_async if inspect.iscoroutinefunction(func) else _wrapper_sync
    return functools.wraps(func)(wrapper)

    # @functools.wraps(func)
    # def wrapper(self, *args, **kwargs):
    #     # get messages from args or kwargs
    #     if args:
    #         messages = args[0]
    #     else:
    #         messages = kwargs.get("messages")
    #     if messages is None:
    #         raise ValueError("Missing required 'messages' parameter for caching.")

    #     # get model, seed and temperature from kwargs or self.llm_config.generate_params
    #     gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
    #     model = kwargs.get("model", gen_params.get("model"))
    #     seed = kwargs.get("seed", gen_params.get("seed"))
    #     temperature = kwargs.get("temperature", gen_params.get("temperature"))

    #     # build key data, convert to JSON string and hash to generate key_hash
    #     key_data = {
    #         "messages": messages,  # messages requires JSON serializable
    #         "model": model,
    #         "seed": seed,
    #         "temperature": temperature,
    #     }
    #     key_str = json.dumps(key_data, sort_keys=True, default=str)
    #     key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    #     # the file name of lock, ensure mutual exclusion when accessing concurrently
    #     lock_file = self.cache_file_name + ".lock"

    #     # Try to read from SQLite cache
    #     with FileLock(lock_file):
    #         conn = sqlite3.connect(self.cache_file_name)
    #         c = conn.cursor()
    #         # if the table does not exist, create it
    #         c.execute("""
    #             CREATE TABLE IF NOT EXISTS cache (
    #                 key TEXT PRIMARY KEY,
    #                 message TEXT,
    #                 metadata TEXT
    #             )
    #         """)
    #         conn.commit()  # commit to save the table creation
    #         c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
    #         row = c.fetchone()
    #         conn.close()
    #         if row is not None:
    #             message, metadata_str = row
    #             metadata = json.loads(metadata_str)
    #             # return cached result and mark as hit
    #             return message, metadata, True

    #     # if cache miss, call the original function to get the result
    #     result = func(self, *args, **kwargs)
    #     message, metadata = result

    #     # insert new result into cache
    #     with FileLock(lock_file):
    #         conn = sqlite3.connect(self.cache_file_name)
    #         c = conn.cursor()
    #         # make sure the table exists again (if it doesn't exist, it would be created)
    #         c.execute("""
    #             CREATE TABLE IF NOT EXISTS cache (
    #                 key TEXT PRIMARY KEY,
    #                 message TEXT,
    #                 metadata TEXT
    #             )
    #         """)
    #         metadata_str = json.dumps(metadata)
    #         c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
    #                   (key_hash, message, metadata_str))
    #         conn.commit()
    #         conn.close()

    #     return message, metadata, False

    # return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper



def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

        
        
@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_cache_hits: int = 0  # track requests completed via cache
    num_api_calls: int = 0   # track requests completed via actual API calls
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    infer_func: Callable
    result: list = field(default_factory=list)
    progress_bar: object = None  # Optional progress bar for real-time updates
    
    
    async def call_api(
        self,
        # session: aiohttp.ClientSession,
        # request_url: str,
        # request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logger.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            # async with session.post(
            #     url=request_url, headers=request_header, json=self.request_json
            # ) as response:
            #     response = await response.json()
                
            # The infer_func returns (message, metadata, cache_hit)
            response_tuple = await self.infer_func(**self.request_json)
            response, metadata, cache_hit = response_tuple
                
            # if "error" in response:
            #     logger.warning(
            #         f"Request {self.task_id} failed with error {response['error']}"
            #     )
            #     status_tracker.num_api_errors += 1
            #     error = response
            #     if "rate limit" in response["error"].get("message", "").lower():
            #         status_tracker.time_of_last_rate_limit_error = time.time()
            #         status_tracker.num_rate_limit_errors += 1
            #         status_tracker.num_api_errors -= (
            #             1  # rate limit errors are counted separately
            #         )
            
        except openai.RateLimitError as e:
            logger.warning(f"Request {self.task_id} failed with RateLimitError: {e}")
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            error = e
            
        except openai.OpenAIError as e:
            logger.warning(f"Request {self.task_id} failed with OpenAIError: {e}")
            status_tracker.num_api_errors += 1
            error = e
            
        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logger.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
            
            
        if error:
            self.result.append(error)
            # This attempt is over; decrement in_progress regardless of retry
            status_tracker.num_tasks_in_progress -= 1
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                status_tracker.num_tasks_failed += 1

                # Best-effort logging; don't block progress on I/O
                try:
                    data = (
                        [self.request_json, [str(e) for e in self.result], self.metadata]
                        if self.metadata
                        else [self.request_json, [str(e) for e in self.result]]
                    )
                    append_to_jsonl(data, save_filepath, default=cache_json_encoder)
                except Exception as log_err:
                    logger.warning(f"Failed to log failed request {self.task_id}: {log_err}")
                
                # Update progress bar immediately for failed requests
                if self.progress_bar is not None:
                    completed = status_tracker.num_tasks_succeeded + status_tracker.num_tasks_failed
                    self.progress_bar.n = completed
                    self.progress_bar.set_postfix({
                        'in_progress': status_tracker.num_tasks_in_progress,
                        'cache_hits': status_tracker.num_cache_hits,
                        'api_calls': status_tracker.num_api_calls,
                        'failed': status_tracker.num_tasks_failed,
                    })
                    self.progress_bar.refresh()
            
            return None
        else:
            # Update counters immediately
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            
            if cache_hit:
                status_tracker.num_cache_hits += 1
                logger.debug(f"Request {self.task_id} completed via cache hit")
            else:
                status_tracker.num_api_calls += 1
                # Log token consumption accuracy for debugging
                estimated_tokens = self.token_consumption
                actual_prompt_tokens = metadata.get('prompt_tokens', 0)
                actual_completion_tokens = metadata.get('completion_tokens', 0)
                actual_total = actual_prompt_tokens + actual_completion_tokens
                logger.debug(f"Request {self.task_id} completed via API call. Token estimation: {estimated_tokens} vs actual: {actual_total} (prompt: {actual_prompt_tokens}, completion: {actual_completion_tokens})")
                logger.debug(f"Request {self.task_id} saved to {save_filepath}")
                # Best-effort logging for API calls; don't block progress on I/O
                try:
                    response_data = {
                        "message": response,
                        "metadata": metadata,
                        "cache_hit": cache_hit
                    }
                    data = (
                        [self.request_json, response_data, self.metadata]
                        if self.metadata
                        else [self.request_json, response_data]
                    )
                    append_to_jsonl(data, save_filepath, default=cache_json_encoder)
                except Exception as log_err:
                    logger.warning(f"Failed to log successful request {self.task_id}: {log_err}")

            # Update progress bar immediately for real-time feedback
            if self.progress_bar is not None:
                completed = status_tracker.num_tasks_succeeded + status_tracker.num_tasks_failed
                self.progress_bar.n = completed
                self.progress_bar.set_postfix({
                    'in_progress': status_tracker.num_tasks_in_progress,
                    'cache_hits': status_tracker.num_cache_hits,
                    'api_calls': status_tracker.num_api_calls,
                    'failed': status_tracker.num_tasks_failed,
                })
                self.progress_bar.refresh()
            
            return response_tuple


class LLMTokenizer:
    """Naive tokenizer which count number of tokens from input string using naive rules."""
    def __init__(self):
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Rough token estimator.

        ──────────────────────────────  assumed tokens/char ──
        ASCII letters (A‑Z / a‑z)                  0.30
        Western‑style digits (0‑9)                 0.20
        Chinese / Japanese Kanji / CJK Ideos      0.60
        Hiragana / Katakana                       0.60
        Hangul (Korean)                           0.60
        Punctuation (any Unicode 'P*' category)   0.10
        Whitespace (isspace())                    0.05
        Emoji & pictographs (🌀–🧿, etc.)          1.00
        Everything else (Cyrillic, Greek, …)      0.30
        """
        t = 0.0
        for ch in text:
            code = ord(ch)

            # --- high‑confidence buckets first ------------------------------
            if ch.isascii() and ch.isalpha():                     # English A–Z
                t += 0.30

            elif 0x4E00 <= code <= 0x9FFF:                        # CJK Unified Ideographs
                t += 0.60
            elif 0x3040 <= code <= 0x30FF or 0xFF65 <= code <= 0xFF9F:   # Hiragana/Katakana
                t += 0.60
            elif 0xAC00 <= code <= 0xD7AF:                        # Hangul syllables
                t += 0.60

            elif ch.isdigit():                                    # 0‑9 (other numerals vary)
                t += 0.20
            elif ch.isspace():                                    # spaces, tabs, newlines
                t += 0.05

            # “fully‑qualified” emoji & pictographs live in these blocks
            elif 0x1F300 <= code <= 0x1FAFF:                      # Emoji & Symbols
                t += 1.00

            elif ud.category(ch).startswith('P'):                 # any punctuation
                t += 0.10

            # --- fallback for everything else ------------------------------
            else:                                                 # Latin‑1 accents, Cyrillic, Greek…
                t += 0.30

        return [0] * math.ceil(t)
        
        
class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        if global_config.date_format_type == "longmemeval" or global_config.date_format_type == "locomo":
            cache_dir = os.path.join(os.path.dirname(global_config.save_dir), "llm_cache")
        else:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, **config_dict)

    def __init__(self, cache_dir, cache_filename: str = None, llm_name: str = "gpt-4o-mini", api_key: str = None, llm_base_url: str = None, llm_calls_save_filepath: str = None, **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self._init_llm_config(**kwargs)
        
        # Remove per-instance asyncio cache lock; rely on FileLock at SQLite level
        self._cache_lock = None
        # if high_throughput:
        #     limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        #     client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        # else:
        #     client = None
        # self.max_retries = kwargs.get("max_retries", 2)
        # self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, http_client=client, max_retries=self.max_retries)

        self.api_key = api_key
        
        # Initialize sync OpenAI client for individual infer calls
        self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key)
        
        # Initialize async OpenAI client for async infer calls  
        self.async_openai_client = AsyncOpenAI(base_url=self.llm_base_url, api_key=api_key)
        
        self.max_retries = kwargs.get("max_retries", None)
        if self.max_retries is None: self.max_retries = kwargs.get("max_retry_attempts", 2)
        
        self.seconds_to_pause_after_rate_limit_error = kwargs.get("seconds_to_pause_after_rate_limit_error", 15)
        self.seconds_to_sleep_each_loop = kwargs.get("seconds_to_sleep_each_loop", 0.001) # 1 ms limits max throughput to 1,000 requests per second
        self.max_requests_per_minute = kwargs.get("max_requests_per_minute", 10000) # target number of requests to make per minute (will make less if limited by tokens), leave headroom by setting this to 50% or 75% of your limit
        self.max_tokens_per_minute = kwargs.get("max_tokens_per_minute", 1000000) # target number of tokens to use per minute (will use less if limited by requests), leave headroom by setting this to 50% or 75% of your limit
        self.max_attempts = kwargs.get("max_attempts", 5) # number of times to retry a failed request before giving up
        # Log the initial rate limit and timing parameters for debugging and transparency
        logger.info(
            f"Initialized CacheOpenAI with max_requests_per_minute={self.max_requests_per_minute}, "
            f"max_tokens_per_minute={self.max_tokens_per_minute}, "
            f"seconds_to_pause_after_rate_limit_error={self.seconds_to_pause_after_rate_limit_error}, "
            f"seconds_to_sleep_each_loop={self.seconds_to_sleep_each_loop}, "
            f"max_attempts={self.max_attempts}"
        )
        
        # file to save immediate llm api calling results (including errors)
        if llm_calls_save_filepath is None:
            self.llm_calls_save_filepath = os.path.join(self.cache_dir, f"{llm_name.replace('/', '_')}_llm_calls_save.jsonl")
        else:
            self.llm_calls_save_filepath = llm_calls_save_filepath
    
    
    async def _check_cache_hit(self, request_json: dict) -> bool:
        """
        Check if a request will result in a cache hit without actually making the call.
        Duplicates the cache key generation logic from cache_response decorator EXACTLY.
        """
        try:
            # EXACT replication of cache_response key generation logic
            
            # Extract messages from request_json (simulating args[0] or kwargs.get("messages"))
            messages = request_json.get("messages")
            if messages is None:
                return False
            
            # Get model, seed and temperature from request_json or self.llm_config.generate_params
            # This exactly matches lines 98-99 in cache_response
            gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
            
            # Build key data - EXACT match to lines 102-108 in cache_response
            key_data = {
                "messages": messages,
                "model": request_json.get("model", gen_params.get("model")),
                "seed": request_json.get("seed", gen_params.get("seed")),
                "temperature": request_json.get("temperature", gen_params.get("temperature")),
                "record_in_n_out_w_metadata": request_json.get("record_in_n_out_w_metadata", True), # be very careful with this variable, which now requires it to be entered in the format of kwargs
            }
            key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            lock_file = self.cache_file_name + ".lock" # the file name of lock, ensure mutual exclusion when accessing concurrently
            
            # Check cache - EXACT match to lines 113-125 in cache_response, without async lock
            def _read():
                conn = sqlite3.connect(self.cache_file_name)
                try:
                    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
                                        key TEXT PRIMARY KEY,
                                        message TEXT,
                                        metadata TEXT
                                    )""")
                    cur = conn.execute("SELECT message, metadata FROM cache WHERE key = ?",
                                       (key_hash,))
                    return cur.fetchone()
                finally:
                    conn.close()
            row = await asyncio.to_thread(_read)
            
            return row is not None
        except Exception:
            return False  # If any error, assume cache miss

    async def _get_cached_tuple(self, request_json: dict):
        """
        Fetch (message, metadata) from cache for the given request, or return None.
        Must mirror cache_response key generation.
        """
        try:
            messages = request_json.get("messages")
            if messages is None:
                return None
            gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
            key_data = {
                "messages": messages,
                "model": request_json.get("model", gen_params.get("model")),
                "seed": request_json.get("seed", gen_params.get("seed")),
                "temperature": request_json.get("temperature", gen_params.get("temperature")),
                "record_in_n_out_w_metadata": request_json.get("record_in_n_out_w_metadata", True),
            }
            key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            def _read():
                conn = sqlite3.connect(self.cache_file_name)
                try:
                    cur = conn.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                    return cur.fetchone()
                finally:
                    conn.close()
            row = await asyncio.to_thread(_read)
            if not row:
                return None
            message, metadata_str = row
            metadata = json.loads(metadata_str, object_hook=cache_json_decoder)
            return message, metadata
        except Exception:
            return None
    
    def num_tokens_consumed_from_request(self, request_json: dict, token_encoding_name: str,):
        """
        This function assumes we are calling OpenAI LLMs.
        Count the number of tokens in the request. Only supports completion and embedding requests.
        
        For output token estimation:
        - If OUTPUT_INPUT_TOKEN_RATIO env var is set, uses that ratio * input_tokens
        - Otherwise falls back to max_completion_tokens (conservative estimate)
        """
        try:
            # encoding = tiktoken.get_encoding(token_encoding_name)
            encoding = tiktoken.encoding_for_model(token_encoding_name)
        except Exception:
            # Middle branch: if model is gpt-4.1 family, try gpt-4o tokenizer as a fallback
            if isinstance(token_encoding_name, str) and "gpt-4.1" in token_encoding_name:
                try:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                    logger.info(
                        f"tokenizer for {token_encoding_name} not available through tiktoken. Going with gpt-4o encoding."
                    )
                except Exception:
                    logger.info(
                        f"tokenizer for {token_encoding_name} (and gpt-4o fallback) not available through tiktoken. Switching to a rule-based naive tokenizer."
                    )
                    encoding = LLMTokenizer()
            else:
                # logger.warning(f"tokenizer for {token_encoding_name} not available through tiktoken. Switching to a rule-based naive tokenizer.")
                logger.info(
                    f"tokenizer for {token_encoding_name} not available through tiktoken. Switching to a rule-based naive tokenizer."
                )
                encoding = LLMTokenizer()
        
        # Calculate input tokens for chat completion
        input_tokens = 0
        for message in request_json["messages"]:
            input_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                input_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    input_tokens -= 1  # role is always required and always 1 token
        input_tokens += 2  # every reply is primed with <im_start>assistant
        
        # Calculate output tokens based on environment variable or fallback to max_tokens
        output_token_ratio = os.getenv("OUTPUT_INPUT_TOKEN_RATIO")
        if output_token_ratio is not None:
            try:
                ratio = float(output_token_ratio)
                if "n" in request_json:            
                    n = request_json["n"]
                else:
                    n = self.llm_config.generate_params.get('n', 1)
                
                completion_tokens = int(n * ratio * input_tokens)
                logger.debug(f"Using OUTPUT_INPUT_TOKEN_RATIO={ratio}: input_tokens={input_tokens}, estimated_output_tokens={completion_tokens}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid OUTPUT_INPUT_TOKEN_RATIO value: {output_token_ratio}. Falling back to max_completion_tokens.")
                completion_tokens = self._get_max_completion_tokens_fallback(request_json)
        else:
            # Fallback to original logic (conservative max_completion_tokens estimate)
            completion_tokens = self._get_max_completion_tokens_fallback(request_json)
            
        return input_tokens + completion_tokens
    
    def _get_max_completion_tokens_fallback(self, request_json: dict) -> int:
        """
        Fallback method to get max completion tokens when ratio-based estimation is not available.
        """
        if 'max_completion_tokens' in request_json:
            max_tokens = request_json['max_completion_tokens']
        elif 'max_tokens' in request_json:
            max_tokens = request_json['max_tokens']
        else:
            max_tokens = self.llm_config.generate_params['max_completion_tokens']
        
        if "n" in request_json:            
            n = request_json["n"]
        else:
            n = self.llm_config.generate_params.get('n', 1)
        
        return n * max_tokens
    
      
    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "max_completion_tokens": kwargs.get("max_new_tokens", 400),
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.0),
            }
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.info(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        record_in_n_out_w_metadata: bool = True,
        **kwargs
    ) -> Tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')


        record_reasoning = False # whether to record the reasoning tokens in the metadata
        if "record_reasoning" in params:
            record_reasoning = params["record_reasoning"]
            params.pop("record_reasoning")
        
        if params.get("response_format", None) is not None and (not isinstance(params["response_format"], dict) or params["response_format"]["type"] == "json_schema"): # OpenAI has supported json schema
            response = self.openai_client.beta.chat.completions.parse(**params)
        else:    
            response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        if record_in_n_out_w_metadata:
            metadata['params'] = params
            metadata['response_content'] = response_message # does not include reason tokens if any
            
        if record_reasoning: 
            metadata["reason_tokens"] = response.usage.completion_tokens_details.reasoning_tokens # tailored for DeepSeek R1 calling with openai client
            metadata["reasoning_content"] = response.choices[0].message.reasoning_content
            
            
        return response_message, metadata



    @cache_response
    async def ainfer(
        self,
        messages: List[TextChatMessage],
        record_in_n_out_w_metadata: bool = True,
        **kwargs
    ) -> Tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')


        record_reasoning = False # whether to record the reasoning tokens in the metadata
        if "record_reasoning" in params:
            record_reasoning = params["record_reasoning"]
            params.pop("record_reasoning")
        
        if params.get("response_format", None) is not None and (not isinstance(params["response_format"], dict) or params["response_format"]["type"] == "json_schema"): # OpenAI has supported json schema
            response = await self.async_openai_client.beta.chat.completions.parse(**params)
        else:    
            response = await self.async_openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        if record_in_n_out_w_metadata:
            metadata['params'] = params
            metadata['response_content'] = response_message # does not include reason tokens if any
            
            
        if record_reasoning: 
            metadata["reason_tokens"] = response.usage.completion_tokens_details.reasoning_tokens # tailored for DeepSeek R1 calling with openai client
            metadata["reasoning_content"] = response.choices[0].message.reasoning_content
            
            
        return response_message, metadata
    
    
    async def abatch_infer(
        self,
        messages: List[List[TextChatMessage]],
        **kwargs
    ) -> List[Tuple[str, dict]]:
        # self.openai_client = AsyncOpenAI(api_key=self.api_key)
        results_future_list = []
        total_requests = len(messages)
        
        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        
        task_id_generator = task_id_generator_function() # generates integer IDs of 0, 1, 2, ...
        
        status_tracker = StatusTracker() # single instance to track a collection of variables
        
        next_request = None  # variable to hold the next request to call
        
        # initialize available capacity counts
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()
        
        
        # initialize flags
        # processed_messages_idx = 0 # indicating which messages in the batched messages input needs LLM inference
        messages_iter = iter(messages)
        messages_not_finished = True
        # idx_lock = self._cache_lock
        logger.info(f"Starting batch inference for {total_requests} requests")
        
        # Initialize progress bar
        pbar = tqdm(total=total_requests, desc="Batch LLM Inference", unit="req")
        last_progress_update = time.time()
        
        
        # Use the instance async client - no need to create a new one or reassign
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logger.debug(f"Retrying request {next_request.task_id}: {next_request}")
                # elif processed_messages_idx < len(messages):
                elif messages_not_finished:
                    try:
                        # get new request
                        # request_json = {"messages": messages[processed_messages_idx], **kwargs}
                        request_json = {"messages": next(messages_iter), **kwargs}
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=self.num_tokens_consumed_from_request(
                                request_json=request_json,
                                token_encoding_name=request_json["model"] if "model" in request_json else self.llm_config.generate_params["model"],
                            ),
                            attempts_left=self.max_attempts,
                            metadata=request_json.pop("metadata", None),
                            infer_func=self.ainfer,
                            progress_bar=pbar
                        )

                        logger.debug(
                            f"Reading request {next_request.task_id}: {next_request}"
                        )
                        # processed_messages_idx += 1
                        # logger.warning(processed_messages_idx)
                    except:
                        # if file runs out, set flag to stop reading it
                        logger.debug("Read input messages exhausted")
                        messages_not_finished = False
                    
                    
            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            
            available_request_capacity = min(
                    available_request_capacity
                    + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute,
                )
            available_token_capacity = min(
                    available_token_capacity
                    + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
            last_update_time = current_time
            
            # if enough capacity available, call API (always allow cache hits)
            if next_request:
                next_request_tokens = next_request.token_consumption
                # Pre-check cache hit first
                will_be_cache_hit = await self._check_cache_hit(next_request.request_json)

                if will_be_cache_hit:
                    # Fast-path: complete from cache without scheduling a task
                    cached = await self._get_cached_tuple(next_request.request_json)
                    if cached is not None:
                        message, metadata = cached
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_succeeded += 1
                        status_tracker.num_cache_hits += 1

                        # push a completed future to preserve ordering
                        fut = asyncio.get_running_loop().create_future()
                        fut.set_result((message, metadata, True))
                        results_future_list.append(fut)
                        next_request = None
                    else:
                        # Fallback to capacity-checked scheduling if cache read failed
                        will_be_cache_hit = False

                if next_request and (available_request_capacity >= 1 and available_token_capacity >= next_request_tokens):
                    # Only consume capacity if NOT a cache hit
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens

                    is_initial_attempt = (next_request.attempts_left == self.max_attempts)
                    next_request.attempts_left -= 1
                    if is_initial_attempt:
                        status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1

                    # call API (always create task, let cache handling happen in call_api)
                    next_request_task = asyncio.create_task(
                        next_request.call_api(
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=self.llm_calls_save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    results_future_list.append(next_request_task)
                    next_request = None  # reset next_request to empty
                
            # Update progress bar every second
            current_time = time.time()
            if current_time - last_progress_update >= 1.0:  # Update every second
                completed = status_tracker.num_tasks_succeeded + status_tracker.num_tasks_failed
                pbar.n = completed
                pbar.set_postfix({
                    'started': status_tracker.num_tasks_started,
                    'queued': max(0, total_requests - status_tracker.num_tasks_started),
                    'in_progress': status_tracker.num_tasks_in_progress,
                    'cache_hits': status_tracker.num_cache_hits,
                    'api_calls': status_tracker.num_api_calls,
                    'failed': status_tracker.num_tasks_failed,
                    'rate_limit_errors': status_tracker.num_rate_limit_errors,
                    'req_capacity': f"{available_request_capacity:.0f}",
                    'token_capacity': f"{available_token_capacity:.0f}"
                })
                pbar.refresh()
                last_progress_update = current_time
            
            # if all messages processed and no tasks in progress, break
            if not messages_not_finished and status_tracker.num_tasks_in_progress == 0:
                break
            
            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(self.seconds_to_sleep_each_loop)
            
            
            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
            if seconds_since_rate_limit_error < self.seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = self.seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logger.warning(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + self.seconds_to_pause_after_rate_limit_error)}")
            
        
        # Collect results maintaining order
        results = []
        if results_future_list:
            # Wait for all tasks to complete (like original gather)
            results = await asyncio.gather(*results_future_list, return_exceptions=True)
        else:
            # No non-cache tasks were created, all were cache hits
            results = []        
        
        # Final progress bar update
        completed = status_tracker.num_tasks_succeeded + status_tracker.num_tasks_failed
        pbar.n = completed
        pbar.set_postfix({
            'started': status_tracker.num_tasks_started,
            'queued': 0,
            'in_progress': 0,
            'cache_hits': status_tracker.num_cache_hits,
            'api_calls': status_tracker.num_api_calls,
            'failed': status_tracker.num_tasks_failed,
            'rate_limit_errors': status_tracker.num_rate_limit_errors,
            'req_capacity': f"{available_request_capacity:.0f}",
            'token_capacity': f"{available_token_capacity:.0f}"
        })
        pbar.close()
                
        # after finishing, log final status
        cache_hit_rate = (status_tracker.num_cache_hits / status_tracker.num_tasks_started * 100) if status_tracker.num_tasks_started > 0 else 0
        logger.info(f"Parallel processing complete. Results saved to {self.llm_calls_save_filepath}")
        logger.info(f"Cache performance: {status_tracker.num_cache_hits} cache hits, {status_tracker.num_api_calls} API calls ({cache_hit_rate:.1f}% cache hit rate)")
        
        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {self.llm_calls_save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")
            
        
        return results
        # await self.openai_client.close() # close the underlying httpx.AsyncClient
    
    
            
    def batch_infer(
        self,
        messages: List[List[TextChatMessage]],
        **kwargs
    ) -> List[Tuple[str, dict]]:
        """
        A normal (blocking) function that you can call from synchronous code. Under the hood it runs the async method to completion.
        """
        logger.info(f"Starting synchronous batch inference for {len(messages)} requests")
        # asyncio.run will create a fresh event loop,
        # run your coro, then close it.
        return asyncio.run(self.abatch_infer(messages, **kwargs))