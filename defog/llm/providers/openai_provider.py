from defog import config as defog_config
import time
import json
import base64
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_openai_format
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or defog_config.get("OPENAI_API_KEY"),
            base_url or "https://api.openai.com/v1/",
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create OpenAI provider from config."""
        return cls(
            api_key=config.get_api_key("openai"),
            base_url=config.get_base_url("openai") or "https://api.openai.com/v1/",
            config=config,
        )

    def get_provider_name(self) -> str:
        return "openai"

    def convert_content_to_openai(self, content: Any) -> Any:
        """Convert message content to OpenAI format."""
        return convert_to_openai_format(content)

    def preprocess_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """Preprocess messages for OpenAI-specific requirements."""
        messages = deepcopy(messages)

        # Convert multimodal content
        for msg in messages:
            msg["content"] = self.convert_content_to_openai(msg["content"])

            # Handle system/developer role conversion
            if model not in ["gpt-4o", "gpt-4o-mini"]:
                if msg.get("role") == "system":
                    msg["role"] = "developer"

        return messages

    def _get_media_type(self, img_data: str) -> str:
        """Detect media type from base64 image data."""
        try:
            decoded = base64.b64decode(img_data[:100])
            if decoded.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            elif decoded.startswith(b"GIF8"):
                return "image/gif"
            elif decoded.startswith(b"RIFF"):
                return "image/webp"
            else:
                return "image/png"  # Default
        except Exception:
            return "image/png"

    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Dict[str, Any]:
        """
        Create a message with image content in OpenAI's format with validation.

        Args:
            image_base64: Base64-encoded image data - can be single string or list of strings
            description: Description of the image(s)
            image_detail: Level of detail for image analysis - "low" or "high" (default: "low")

        Returns:
            Message dict in OpenAI's format

        Raises:
            ValueError: If no valid images are provided or validation fails
        """
        from ..utils_image_support import (
            validate_and_process_image_data,
            safe_extract_media_type_and_data,
        )

        # Validate image_detail parameter
        if image_detail not in ["low", "high"]:
            raise ValueError(
                f"Invalid image_detail value: {image_detail}. Must be 'low' or 'high'"
            )

        # Validate and process image data
        valid_images, errors = validate_and_process_image_data(image_base64)

        if not valid_images:
            error_summary = "; ".join(errors) if errors else "No valid images provided"
            raise ValueError(f"Cannot create image message: {error_summary}")

        if errors:
            # Log warnings for any invalid images but continue with valid ones
            for error in errors:
                logger.warning(f"Skipping invalid image: {error}")

        content = [{"type": "text", "text": description}]

        # Handle validated images
        for img_data in valid_images:
            media_type, clean_data = safe_extract_media_type_and_data(img_data)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{clean_data}",
                        "detail": image_detail,
                    },
                }
            )

        return {"role": "user", "content": content}

    def supports_tools(self, model: str) -> bool:
        return True

    def supports_response_format(self, model: str) -> bool:
        return True

    def build_params(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        seed: int = 0,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build the parameter dictionary for OpenAI's chat.completions.create().
        Also handles special logic for o1-mini, o1-preview, deepseek-chat, etc.
        """
        # Preprocess messages using the base class method
        messages = self.preprocess_messages(messages, model)

        request_params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "seed": seed,
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }

        # Tools are only supported for certain models
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

            # Set parallel_tool_calls based on configuration
            # this parameter is not available on the o-series models, though
            if not model.startswith("o"):
                request_params["parallel_tool_calls"] = (
                    self.config.enable_parallel_tool_calls
                )

        # Some models do not allow temperature or response_format:
        if model.startswith("o") or model == "deepseek-reasoner":
            request_params.pop("temperature", None)

        # Reasoning effort
        if model.startswith("o") and reasoning_effort is not None:
            request_params["reasoning_effort"] = reasoning_effort

        # Special case: model in ["gpt-4o", "gpt-4o-mini"] with `prediction`
        if model in ["gpt-4o", "gpt-4o-mini"] and prediction is not None:
            request_params["prediction"] = prediction
            request_params.pop("max_completion_tokens", None)
            request_params.pop("response_format", None)

        # Finally, set response_format if still relevant:
        if response_format:
            request_params["response_format"] = response_format

        # Add MCP servers if provided
        if mcp_servers:
            mcp_tools = []
            for server in mcp_servers:
                # Support both Anthropic-style and OpenAI-style server configs
                # Anthropic format: {"name": "server", "url": "https://...", "type": "url"}
                # OpenAI format: {"server_label": "server", "server_url": "https://..."}

                server_name = server.get("name") or server.get(
                    "server_label", "unknown"
                )
                server_url = server.get("url") or server.get("server_url")

                if not server_url:
                    raise ValueError(f"MCP server missing URL: {server}")

                mcp_tool = {
                    "type": "mcp",
                    "server_label": server_name,
                    "server_url": server_url,
                    "require_approval": server.get(
                        "require_approval", "never"
                    ),  # Default to "never"
                }

                # Add headers if provided
                if "headers" in server:
                    mcp_tool["headers"] = server["headers"]

                # Add allowed_tools if provided
                if "allowed_tools" in server:
                    mcp_tool["allowed_tools"] = server["allowed_tools"]

                mcp_tools.append(mcp_tool)

            # Add MCP tools to existing tools array or create new one
            if "tools" in request_params:
                request_params["tools"].extend(mcp_tools)
            else:
                request_params["tools"] = mcp_tools

        return request_params, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_handler: Optional[ToolHandler] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from OpenAI response.
        Handles chaining of tool calls.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        if len(response.choices) == 0:
            raise ProviderError(self.get_provider_name(), "No response from OpenAI")
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                # Use base class method for token calculation
                input_tokens, output_tokens, cached_tokens, _ = (
                    self.calculate_token_usage(response)
                )
                total_input_tokens += input_tokens
                total_cached_input_tokens += cached_tokens
                total_output_tokens += output_tokens
                message = response.choices[0].message
                if message.tool_calls:
                    try:
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call in message.tool_calls:
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            tool_calls_batch.append(
                                {
                                    "id": tool_call.id,
                                    "function": {"name": func_name, "arguments": args},
                                }
                            )

                        # Use base class method for tool execution with retry
                        (
                            results,
                            consecutive_exceptions,
                        ) = await self.execute_tool_calls_with_retry(
                            tool_calls_batch,
                            tool_dict,
                            request_params["messages"],
                            post_tool_function,
                            consecutive_exceptions,
                            tool_handler,
                        )

                        # Append the tool calls as an assistant response
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "tool_calls": message.tool_calls,
                            }
                        )

                        # Process results and handle images with provider-specific format
                        tool_call_blocks = message.tool_calls

                        # Store tool outputs for tracking
                        for tool_call, result in zip(message.tool_calls, results):
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "text": (
                                        message.content if message.content else None
                                    ),
                                }
                            )

                        # Use provider-specific image processing
                        from ..utils_image_support import (
                            process_tool_results_with_images,
                        )

                        tool_data_list = process_tool_results_with_images(
                            tool_call_blocks, results, image_result_keys
                        )

                        # Create OpenAI-specific messages (separate tool and image messages)
                        for tool_data in tool_data_list:
                            # Add tool message
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_data.tool_id,
                                    "content": tool_data.tool_result_text,
                                }
                            )

                            # Add image message immediately after if present
                            if tool_data.image_data:
                                image_message = self.create_image_message(
                                    tool_data.image_data,
                                    f"Image generated by {tool_data.tool_name} tool",
                                )
                                request_params["messages"].append(image_message)

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params, model
                        )

                        # Set tool_choice to "auto" so that the next message will be generated normally
                        request_params["tool_choice"] = (
                            "auto" if request_params["tool_choice"] != "auto" else None
                        )
                    except ProviderError:
                        # Re-raise provider errors from base class
                        raise
                    except Exception as e:
                        # For other exceptions, use the same retry logic
                        consecutive_exceptions += 1
                        if (
                            consecutive_exceptions
                            >= tool_handler.max_consecutive_errors
                        ):
                            raise ProviderError(
                                self.get_provider_name(),
                                f"Consecutive errors during tool chaining: {e}",
                                e,
                            )
                        print(
                            f"{e}. Retries left: {tool_handler.max_consecutive_errors - consecutive_exceptions}"
                        )
                        request_params["messages"].append(
                            {"role": "assistant", "content": str(e)}
                        )

                    # Make next call
                    response = await client.chat.completions.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = message.content
                    break
        else:
            # No tools provided
            if response_format:
                try:
                    parsed_content = response.choices[0].message.parsed
                    if parsed_content is not None:
                        content = parsed_content
                    else:
                        # Use base class method for structured response parsing
                        content = self.parse_structured_response(
                            response.choices[0].message.content, response_format
                        )
                except Exception:
                    # Use base class method for structured response parsing
                    content = self.parse_structured_response(
                        response.choices[0].message.content, response_format
                    )
            else:
                content = response.choices[0].message.content

        # Final token calculation
        input_tokens, output_tokens, cached_tokens, output_tokens_details = (
            self.calculate_token_usage(response)
        )
        total_input_tokens += input_tokens
        total_cached_input_tokens += cached_tokens
        total_output_tokens += output_tokens
        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            output_tokens_details,
        )

    async def execute_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        seed: int = 0,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with OpenAI."""
        from openai import AsyncOpenAI

        # Create a ToolHandler instance with tool_budget if provided
        tool_handler = self.create_tool_handler_with_budget(tool_budget)

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_openai = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        # Filter tools based on budget before building params
        tools = self.filter_tools_by_budget(tools, tool_handler)

        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            prediction=prediction,
            reasoning_effort=reasoning_effort,
            store=store,
            metadata=metadata,
            timeout=timeout,
            mcp_servers=mcp_servers,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            # If MCP servers are provided, use Responses API
            if mcp_servers and len(mcp_servers) > 0:
                # Check if responses API is available
                if not hasattr(client_openai, "responses"):
                    raise ProviderError(
                        self.get_provider_name(),
                        "MCP support requires OpenAI Python library with Responses API support. "
                        "Please update to the latest version or use Anthropic provider for MCP functionality.",
                    )

                # Prepare parameters for Responses API
                responses_params = {
                    "model": request_params["model"],
                    "input": messages,
                    "tools": request_params.get("tools", []),
                }

                # Add other relevant parameters
                if request_params.get("max_completion_tokens"):
                    responses_params["max_completion_tokens"] = request_params[
                        "max_completion_tokens"
                    ]
                if request_params.get("temperature") is not None:
                    responses_params["temperature"] = request_params["temperature"]
                if request_params.get("store") is not None:
                    responses_params["store"] = request_params["store"]

                response = await client_openai.responses.create(**responses_params)

                # Extract content and tool outputs from response
                content = response.output_text or ""
                tool_outputs = []

                # Extract MCP tool calls from response
                for output_item in response.output:
                    if output_item.type == "mcp_call":
                        tool_outputs.append(
                            {
                                "tool_call_id": output_item.id,
                                "name": output_item.name,
                                "args": (
                                    json.loads(output_item.arguments)
                                    if output_item.arguments
                                    else {}
                                ),
                                "result": output_item.output,
                                "server_label": output_item.server_label,
                            }
                        )

                # Get usage information
                input_tokens = (
                    response.usage.input_tokens
                    if hasattr(response, "usage") and response.usage
                    else 0
                )
                output_tokens = (
                    response.usage.output_tokens
                    if hasattr(response, "usage") and response.usage
                    else 0
                )
                cached_input_tokens = 0
                completion_token_details = None

            else:
                # Use regular Chat Completions API
                if request_params.get("response_format"):
                    response = await client_openai.beta.chat.completions.parse(
                        **request_params
                    )
                else:
                    response = await client_openai.chat.completions.create(
                        **request_params
                    )

                (
                    content,
                    tool_outputs,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    completion_token_details,
                ) = await self.process_response(
                    client=client_openai,
                    response=response,
                    request_params=request_params,
                    tools=tools,
                    tool_dict=tool_dict,
                    response_format=response_format,
                    model=model,
                    post_tool_function=post_tool_function,
                    image_result_keys=image_result_keys,
                    tool_handler=tool_handler,
                )
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_tokens, output_tokens, cached_input_tokens
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=completion_token_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
        )
