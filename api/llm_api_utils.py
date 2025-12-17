import json
import os
import time
import threading

import httpx
import requests
from openai import OpenAI

# Import API logger for tracking LLM usage
from local_api_logger import log_completion, set_log_dir

# Global flag to control API logging (configured at runtime)
_api_logger_enabled = True
_api_logger_user = "default"

# Thread-local storage for httpx clients (one client per thread for thread-safety)
# CRITICAL: httpx.Client is NOT thread-safe. Using a single global client across
# multiple threads causes connection pool corruption and "Connection error" failures.
# Solution: Each thread gets its own httpx.Client instance via thread-local storage.
_thread_local = threading.local()

def _get_thread_local_httpx_client(timeout):
    """
    Get or create a thread-local httpx client for connection pooling.

    IMPORTANT: httpx.Client is NOT thread-safe. When multiple threads share the same
    client instance (as in batch processing with ThreadPoolExecutor), concurrent API
    calls corrupt the connection pool, leading to "Connection error" failures.

    This function ensures each thread has its own httpx.Client instance, preventing
    connection pool corruption while still benefiting from connection reuse WITHIN
    each thread.

    Args:
        timeout: httpx.Timeout object with timeout configuration

    Returns:
        Thread-local httpx.Client instance
    """
    # Check if this thread already has a client
    if not hasattr(_thread_local, 'httpx_client') or _thread_local.httpx_client is None:
        # Create a new client for this thread
        _thread_local.httpx_client = httpx.Client(
            verify=False,  # TODO: Enable certificate verification in production
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=10,  # Per-thread limit (reduced from 20)
                max_connections=25,            # Per-thread limit (reduced from 50)
                keepalive_expiry=300.0         # Keep connections alive for 5 minutes
            )
        )

    return _thread_local.httpx_client


def cleanup_thread_local_httpx_client():
    """
    Cleanup the thread-local httpx client for the current thread.

    This should be called after completing a batch of work in a thread
    to properly close connections and free resources. Each thread manages
    its own client lifecycle.
    """
    if hasattr(_thread_local, 'httpx_client') and _thread_local.httpx_client is not None:
        try:
            _thread_local.httpx_client.close()
        except Exception:
            pass  # Ignore errors during cleanup
        finally:
            _thread_local.httpx_client = None


def cleanup_global_httpx_client():
    """
    Legacy function for backward compatibility.
    Now delegates to thread-local cleanup.
    """
    cleanup_thread_local_httpx_client()


def initialize_api_logger(enable: bool = True, log_dir: str = "./api_logs", user: str = "default"):
    """
    Initialize the API logger with configuration from config.
    Should be called once at application startup.

    Args:
        enable: Whether to enable API logging
        log_dir: Directory to store API logs
        user: User identifier for logs
    """
    global _api_logger_enabled, _api_logger_user
    _api_logger_enabled = enable
    _api_logger_user = user

    if enable:
        set_log_dir(log_dir)


def _should_log_api_call(model_name: str) -> bool:
    """
    Determine if an API call should be logged.
    
    Rules:
    - API logger must be enabled globally
    - Pangu models are NEVER logged (per company policy)
    
    Args:
        model_name: Name of the model being called
        
    Returns:
        True if the call should be logged, False otherwise
    """
    if not _api_logger_enabled:
        return False
    
    # Never log pangu models
    if model_name.find('pangu') != -1:
        return False
    
    return True


def _log_api_call(
    model_name: str,
    request_data: dict,
    response_data: dict,
    duration_ms: float,
    api_key: str = None, # <--- [新增]
    success: bool = True,
    error: str = None
):
    """
    Log an API call using the local_api_logger.
    
    Args:
        model_name: Name of the model
        request_data: Request payload (messages, temperature, etc.)
        response_data: Response from API (must contain usage field for token tracking)
        duration_ms: Call duration in milliseconds
        success: Whether the call succeeded
        error: Error message if call failed
    """
    if not _should_log_api_call(model_name):
        return
    
    try:
        log_completion(
            model=model_name,
            request_data=request_data,
            response_data=response_data,
            user=_api_logger_user,
            duration_ms=duration_ms,
            api_key=api_key
        )
    except Exception as e:
        # Don't let logging errors break the main flow
        print(f"Warning: Failed to log API call: {e}")


def get_initial_messages(model_name, user_msg_content, system_msg_content=None):
    # AWS Bedrock requires content to be non-empty
    # Provide sensible defaults if content is None or empty
    if not user_msg_content or not user_msg_content.strip():
        user_content = "Please help me with this task."
    else:
        user_content = user_msg_content

    if system_msg_content and system_msg_content.strip():
        system_content = system_msg_content
    else:
        system_content = None

    if system_content is not None:
        if model_name.find('pangu') == -1:
            return [{"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}]
        else:
            return [{"role": "system", "content": system_content},
                    {"role": "user", "content": user_content + " /no_think"}]
    if model_name.find('pangu') == -1:

        return [{"role": "user", "content": user_content}]
    else:
        return [{"role": "user", "content": user_content + " /no_think"}]

def get_model_response(model_url, model_name, model_token, messages, tool_schemas, agent_logger, max_retry_num=10,
                       generation_configs=None, temperature=None, max_tokens=None, timeout=None):
    """
    Get model response from either OpenAI-compatible API or Pangu API.

    Args:
        model_url: API endpoint URL
        model_name: Name of the model
        model_token: API token
        messages: Conversation history
        tool_schemas: Tool schemas (only for OpenAI-compatible, ignored for Pangu)
        agent_logger: Logger instance
        max_retry_num: Maximum retry attempts
        generation_configs: Pangu-specific configs (chat_template, temperature, max_tokens, timeout). Only used for Pangu.
        temperature: Temperature for OpenAI client (ignored for Pangu)
        max_tokens: Max tokens for OpenAI client (ignored for Pangu)
        timeout: Unified timeout value (seconds) for all operations (connect, read, write, pool). If None, uses default from config.
    """
    # Track timing for API logging
    call_start_time = time.time()
    
    if model_name.find('pangu') == -1:
        # OpenAI-compatible API path - does NOT use generation_configs or /no_think
        for retry_num in range(max_retry_num):
            try:
                os.environ["OPENAI_API_KEY"] = model_token
                os.environ["OPENAI_BASE_URL"] = model_url

                # Configure unified timeout for all httpx operations
                # Uses MODEL_REQUEST_TIMEOUT for connect, read, write, and pool operations
                if timeout is None:
                    # Get default from config if not provided
                    from config.config import get_config
                    timeout = get_config().model_request_timeout

                # Apply better timeout values (shorter connect, longer read for LLM responses)
                httpx_timeout = httpx.Timeout(
                    connect=30.0,     # 30 seconds for SSL handshake (shorter is better)
                    read=timeout,     # Response read timeout (can be long for LLM)
                    write=60.0,       # 60 seconds for request write
                    pool=30.0         # 30 seconds for connection pool
                )

                # Use thread-local HTTP client for connection pooling (one client per thread)
                # CRITICAL: This prevents connection pool corruption when multiple threads
                # make concurrent API calls (e.g., batch processing with ThreadPoolExecutor)
                httpx_client = _get_thread_local_httpx_client(httpx_timeout)
                client = OpenAI(http_client=httpx_client)
                
                # Build request parameters for OpenAI client
                request_params = {
                    "model": model_name,
                    "messages": messages,
                }
                
                # Add optional parameters if provided
                if temperature is not None:
                    request_params["temperature"] = temperature
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
                if tool_schemas is not None:
                    request_params["tools"] = tool_schemas
                
                response = client.chat.completions.create(**request_params)
                if agent_logger is not None:
                    agent_logger.debug(f"API response received")
                
                # Log the API call (excluding pangu models)
                call_duration_ms = (time.time() - call_start_time) * 1000
                _log_api_call(
                    model_name=model_name,
                    request_data=request_params,
                    response_data={
                        "choices": [{
                            "message": {
                                "role": response.choices[0].message.role,
                                "content": response.choices[0].message.content,
                            },
                            "finish_reason": response.choices[0].finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            "total_tokens": response.usage.total_tokens if response.usage else 0
                        } if response.usage else None,
                        "model": response.model,
                        "id": response.id
                    },
                    duration_ms=call_duration_ms,
                    success=True,
                    api_key=model_token
                )
                break
            except Exception as e:
                # Build concise context for debugging
                agent_name = getattr(agent_logger, 'name', 'UnknownAgent') if agent_logger else 'UnknownAgent'
                last_user_msg = None
                try:
                    for m in reversed(messages or []):
                        if m.get("role") == "user":
                            content = m.get("content", "")
                            last_user_msg = (content[:300] + "…") if len(content) > 300 else content
                            break
                except Exception:
                    last_user_msg = None

                context = {
                    "agent": agent_name.split('.')[-1],
                    "model": model_name,
                    "url": model_url,
                    "last_user_message": last_user_msg,
                }

                if agent_logger is not None:
                    agent_logger.error(
                        f"API request failed (attempt {retry_num + 1}/{max_retry_num}): {str(e)} | context={json.dumps(context, ensure_ascii=False)}"
                    )
                    # Full messages only at debug level
                    agent_logger.debug(f"Failed messages: {json.dumps(messages, ensure_ascii=False)}")

                # Use exponential backoff instead of fixed delay
                # Start at 2 seconds, double each time, cap at 60 seconds
                if retry_num < max_retry_num - 1:
                    backoff_delay = min(2 ** retry_num, 60)
                    if agent_logger is not None:
                        agent_logger.info(f"Retrying in {backoff_delay} seconds...")
                    time.sleep(backoff_delay)
                else:
                    raise ValueError(
                        f"API request failed after {max_retry_num} attempts: {str(e)} | context={json.dumps(context, ensure_ascii=False)}"
                    )
                continue
        return response.choices[0].message

    else:
        # Pangu API path - uses generation_configs and expects /no_think in messages
        if generation_configs is None:
            raise ValueError("generation_configs is required for Pangu models (must include chat_template, temperature, max_tokens)")
        
        headers = {'Content-Type': 'application/json', 'csb-token': model_token}
        for retry_num in range(max_retry_num):
            try:
                request_payload = {
                    "model": model_name,
                    "chat_template": generation_configs.get('chat_template'),
                    "messages": messages,
                    "temperature": generation_configs.get('temperature'),
                    "spaces_between_special_tokens": False,
                    "max_tokens": generation_configs.get('max_tokens'),
                }
                
                response = requests.post(
                    url=model_url,
                    headers=headers,
                    json=request_payload,
                    timeout=generation_configs.get("timeout", 180)
                )
                response = response.json()
                if agent_logger is not None:
                    agent_logger.debug(f"API response received")
                
                # Note: Pangu models are NOT logged per company policy
                # The _should_log_api_call function will filter them out
                # This is just for completeness - in practice, pangu calls won't be logged
                call_duration_ms = (time.time() - call_start_time) * 1000
                _log_api_call(
                    model_name=model_name,
                    request_data=request_payload,
                    response_data=response,
                    duration_ms=call_duration_ms,
                    api_key=model_token,
                    success=True
                )
                break
            except Exception as e:
                # Build concise context for debugging
                agent_name = getattr(agent_logger, 'name', 'UnknownAgent') if agent_logger else 'UnknownAgent'
                last_user_msg = None
                try:
                    for m in reversed(messages or []):
                        if m.get("role") == "user":
                            content = m.get("content", "")
                            last_user_msg = (content[:300] + "…") if len(content) > 300 else content
                            break
                except Exception:
                    last_user_msg = None

                context = {
                    "agent": agent_name.split('.')[-1],
                    "model": model_name,
                    "url": model_url,
                    "last_user_message": last_user_msg,
                }
                if agent_logger is not None:
                    agent_logger.error(
                        f"API request failed (attempt {retry_num + 1}/{max_retry_num}): {str(e)} | context={json.dumps(context, ensure_ascii=False)}"
                    )
                    agent_logger.debug(f"Failed messages: {json.dumps(messages, ensure_ascii=False)}")

                # Use exponential backoff instead of fixed delay
                if retry_num < max_retry_num - 1:
                    backoff_delay = min(2 ** retry_num, 60)
                    if agent_logger is not None:
                        agent_logger.info(f"Retrying in {backoff_delay} seconds...")
                    time.sleep(backoff_delay)
                else:
                    raise ValueError(
                        f"API request failed after {max_retry_num} attempts: {str(e)} | context={json.dumps(context, ensure_ascii=False)}"
                    )
                continue

        return response["choices"][0]["message"]


def extract_tool_calls_for_pangu(content):
    import re
    if not content:
        return []
    tool_call_str = re.findall(r"\[unused11\]([\s\S]*?)\[unused12\]", content)
    if len(tool_call_str) > 0:
        try:
            tool_calls = json.loads(tool_call_str[0].strip())
        except:
            return []
    else:
        return []
    return tool_calls


def get_formatted_tool_calls_for_openai_api_response(assistant_message):
    format_tool_calls = []
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name
            format_tool_calls.append({"function": {
                "arguments": json.dumps(arguments, ensure_ascii=False), "name": tool_name},
                "id": tool_call.id, "type": "function"})
    return format_tool_calls


def get_pangu_tool_prompt_for_system():
    return """\n\nBelow, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
[unused11][{\"name\": <function name>, \"arguments\": <args json object>}][unused12]"""


class ProcessedModelResponse:
    """
    Standardized model response structure that works uniformly across different model types.
    """
    def __init__(self, model_name, raw_response):
        """
        Initialize processed response.

        Args:
            model_name: Name of the model (used to determine processing logic)
            raw_response: Raw response from get_model_response
        """
        self.model_name = model_name
        self.raw_response = raw_response
        self.is_pangu = model_name.find('pangu') != -1

        # Standardized fields
        self.reasoning_content = None
        self.tool_calls = []
        self.assistant_message_for_history = None

        # Token usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        # Process the response based on model type
        self._process_response()
    
    def _process_response(self):
        """Process the raw response based on model type."""
        if not self.is_pangu:
            # OpenAI-compatible model processing
            self.reasoning_content = self.raw_response.content if self.raw_response.content else None

            # Extract token usage
            if hasattr(self.raw_response, 'usage') and self.raw_response.usage:
                self.prompt_tokens = getattr(self.raw_response.usage, 'prompt_tokens', 0)
                self.completion_tokens = getattr(self.raw_response.usage, 'completion_tokens', 0)
                self.total_tokens = getattr(self.raw_response.usage, 'total_tokens', 0)

            # Format tool calls
            if hasattr(self.raw_response, 'tool_calls') and self.raw_response.tool_calls:
                self.tool_calls = [
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "id": tool_call.id,
                        "type": "function"
                    }
                    for tool_call in self.raw_response.tool_calls
                ]

            # Prepare message for conversation history
            # AWS Bedrock requires content to be non-empty if present
            # If there's no content but there are tool calls, omit the content field entirely
            self.assistant_message_for_history = {
                "role": "assistant",
            }

            # Only include content if it's non-empty
            if self.raw_response.content:
                self.assistant_message_for_history["content"] = self.raw_response.content

            # Include tool calls if present
            if hasattr(self.raw_response, 'tool_calls') and self.raw_response.tool_calls:
                self.assistant_message_for_history["tool_calls"] = get_formatted_tool_calls_for_openai_api_response(
                    self.raw_response
                )
        else:
            # Pangu model processing
            content = self.raw_response.get("content", "")

            # Extract token usage if available (Pangu may provide usage in response dict)
            if isinstance(self.raw_response, dict) and "usage" in self.raw_response:
                usage = self.raw_response["usage"]
                self.prompt_tokens = usage.get("prompt_tokens", 0)
                self.completion_tokens = usage.get("completion_tokens", 0)
                self.total_tokens = usage.get("total_tokens", 0)

            # Extract reasoning content (between [unused16] and [unused17])
            try:
                if content:
                    reasoning_parts = content.split("[unused16]")[-1].split("[unused17]")
                    if len(reasoning_parts) > 0:
                        self.reasoning_content = reasoning_parts[0]
            except Exception:
                self.reasoning_content = None

            # Extract tool calls
            self.tool_calls = extract_tool_calls_for_pangu(content)

            # Prepare message for conversation history
            self.assistant_message_for_history = {
                "role": "assistant",
                "content": "[unused16][unused17]" + content
            }
    
    def has_tool_calls(self):
        """Check if there are any tool calls."""
        return len(self.tool_calls) > 0
    
    def format_tool_result_for_history(self, tool_result, tool_call_id=None):
        """
        Format tool result for conversation history based on model type.
        
        Args:
            tool_result: The tool execution result
            tool_call_id: Tool call ID (only used for OpenAI-compatible models)
            
        Returns:
            Dictionary formatted for conversation history
        """
        if not self.is_pangu:
            # OpenAI format
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result, ensure_ascii=False)
            }
        else:
            # Pangu format (append /no_think)
            return {
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False) + " /no_think"
            }
    
    def format_followup_message(self, message_content):
        """
        Format a follow-up user message based on model type.

        Args:
            message_content: The message content

        Returns:
            Dictionary formatted for conversation history
        """
        # AWS Bedrock requires content to be non-empty
        # If message_content is None or empty, use a minimal placeholder
        if not message_content or not message_content.strip():
            content = "Continue."
        else:
            content = message_content

        if not self.is_pangu:
            return {"role": "user", "content": content}
        else:
            return {"role": "user", "content": content + " /no_think"}


def process_model_response(model_name, raw_response):
    """
    Process model response and return a standardized ProcessedModelResponse object.
    
    Args:
        model_name: Name of the model
        raw_response: Raw response from get_model_response
        
    Returns:
        ProcessedModelResponse object with standardized interface
    """
    return ProcessedModelResponse(model_name, raw_response)


def get_and_process_model_response(model_url, model_name, model_token, conversation_history,
                                     tool_schemas, logger, agent_config, max_retry=10):
    """
    Unified function to get and process model response without caring about model type.

    This function handles both Pangu and OpenAI-compatible models automatically,
    eliminating the need for if-else blocks in agent code.

    Args:
        model_url: API endpoint URL
        model_name: Name of the model
        model_token: API token
        conversation_history: Conversation history list
        tool_schemas: Tool schemas (for OpenAI models)
        logger: Logger instance
        agent_config: Agent config object with temperature, max_tokens, chat_template
        max_retry: Maximum retry attempts

    Returns:
        ProcessedModelResponse object with standardized interface
    """
    # Get timeout from config - uses MODEL_REQUEST_TIMEOUT for both Pangu and OpenAI-compatible models
    from config.config import get_config
    unified_timeout = get_config().model_request_timeout

    # Get raw response based on model type
    if model_name.find('pangu') == -1:
        # OpenAI-compatible models
        raw_response = get_model_response(
            model_url, model_name, model_token,
            conversation_history, tool_schemas, logger, max_retry,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            timeout=unified_timeout
        )
    else:
        # Pangu models
        generation_configs = {
            "chat_template": agent_config.chat_template,
            "temperature": agent_config.temperature,
            "max_tokens": agent_config.max_tokens,
            "timeout": unified_timeout  # Now actually uses MODEL_REQUEST_TIMEOUT!
        }
        raw_response = get_model_response(
            model_url, model_name, model_token,
            conversation_history, None, logger, max_retry,
            generation_configs=generation_configs
        )

    # Process and return standardized response
    return process_model_response(model_name, raw_response)


def build_system_prompt_with_tools(model_name, base_system_prompt, tool_schemas):
    """
    Build system prompt with tool schemas based on model type.
    
    For Pangu models, appends special tool invocation format.
    For OpenAI-compatible models, returns base prompt as-is (tools passed separately).
    
    Args:
        model_name: Name of the model
        base_system_prompt: Base system prompt without tool information
        tool_schemas: List of tool schemas
        
    Returns:
        Complete system prompt string
    """
    if model_name.find('pangu') == -1:
        # OpenAI-compatible: tools are passed separately, not in system prompt
        return base_system_prompt
    else:
        # Pangu: tools must be included in system prompt
        tool_schemas_str = json.dumps(tool_schemas, ensure_ascii=False)
        return base_system_prompt + get_pangu_tool_prompt_for_system().replace("$tool_schemas", tool_schemas_str)
