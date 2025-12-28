"""
Unit tests for the Grok-4 functionality in llm.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import time
from typing import List
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from llm import stream_grok_response, Llm, Completion


class TestStreamGrokResponse:
    """Test suite for the stream_grok_response function"""

    @pytest.fixture
    def mock_messages(self) -> List[ChatCompletionMessageParam]:
        """Sample messages for testing"""
        return [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, Grok!"}
        ]

    @pytest.fixture
    def mock_callback(self) -> AsyncMock:
        """Mock callback function"""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_should_use_default_xai_base_url_when_none_provided(
        self, mock_messages, mock_callback
    ):
        """Should use default xAI base URL when none is provided"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create a mock stream that yields no chunks (empty response)
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = []
            mock_client.chat.completions.create.return_value = mock_stream
            
            await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify AsyncOpenAI was initialized with default xAI URL
            mock_openai_class.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://api.x.ai/v1"
            )

    @pytest.mark.asyncio
    async def test_should_use_custom_base_url_when_provided(
        self, mock_messages, mock_callback
    ):
        """Should use custom base URL when provided"""
        custom_url = "https://custom.proxy.com/v1"
        
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create a mock stream that yields no chunks
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = []
            mock_client.chat.completions.create.return_value = mock_stream
            
            await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=custom_url,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify AsyncOpenAI was initialized with custom URL
            mock_openai_class.assert_called_once_with(
                api_key="test-api-key",
                base_url=custom_url
            )

    @pytest.mark.asyncio
    async def test_should_handle_streaming_responses_correctly(
        self, mock_messages, mock_callback
    ):
        """Should handle streaming responses correctly"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create mock chunks with content
            chunk1 = ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="grok-4",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content="Hello "),
                        finish_reason=None
                    )
                ]
            )
            
            chunk2 = ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567891,
                model="grok-4",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content="world!"),
                        finish_reason=None
                    )
                ]
            )
            
            # Create a mock stream that yields chunks
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = [chunk1, chunk2]
            mock_client.chat.completions.create.return_value = mock_stream
            
            result = await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify the full response was assembled correctly
            assert result["code"] == "Hello world!"
            
            # Verify callback was called for each chunk
            assert mock_callback.call_count == 2
            mock_callback.assert_has_calls([
                call("Hello "),
                call("world!")
            ])

    @pytest.mark.asyncio
    async def test_should_properly_handle_api_errors_and_reraise(
        self, mock_messages, mock_callback
    ):
        """Should properly handle API errors and re-raise them"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Make the API call raise an exception
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            with pytest.raises(Exception) as exc_info:
                await stream_grok_response(
                    messages=mock_messages,
                    api_key="test-api-key",
                    base_url=None,
                    callback=mock_callback,
                    model=Llm.GROK_4
                )
            
            assert str(exc_info.value) == "API Error"
            
            # Verify client was still closed
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_close_client_connection_even_when_errors_occur(
        self, mock_messages, mock_callback
    ):
        """Should close the client connection even when errors occur"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create a mock stream that raises an error during iteration
            mock_stream = AsyncMock()
            async def error_iterator():
                raise Exception("Stream error")
                yield  # This won't be reached
            
            mock_stream.__aiter__ = error_iterator
            mock_client.chat.completions.create.return_value = mock_stream
            
            with pytest.raises(Exception):
                await stream_grok_response(
                    messages=mock_messages,
                    api_key="test-api-key",
                    base_url=None,
                    callback=mock_callback,
                    model=Llm.GROK_4
                )
            
            # Verify client was closed despite the error
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_calculate_completion_time_correctly(
        self, mock_messages, mock_callback
    ):
        """Should calculate completion time correctly"""
        with patch('llm.AsyncOpenAI') as mock_openai_class, \
             patch('llm.time.time') as mock_time:
            
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock time to return specific values
            mock_time.side_effect = [100.0, 105.5]  # Start and end times
            
            # Create a mock stream with no chunks
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = []
            mock_client.chat.completions.create.return_value = mock_stream
            
            result = await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify duration was calculated correctly
            assert result["duration"] == 5.5

    @pytest.mark.asyncio
    async def test_should_handle_empty_response_chunks_gracefully(
        self, mock_messages, mock_callback
    ):
        """Should handle empty response chunks gracefully"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create chunks with various empty/None conditions
            chunks = [
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="grok-4",
                    choices=[]  # Empty choices
                ),
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567891,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content=None),  # None content
                            finish_reason=None
                        )
                    ]
                ),
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567892,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="Valid"),  # Valid content
                            finish_reason=None
                        )
                    ]
                )
            ]
            
            # Create a mock stream that yields chunks
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = chunks
            mock_client.chat.completions.create.return_value = mock_stream
            
            result = await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Only "Valid" should be in the response
            assert result["code"] == "Valid"
            
            # Callback should only be called once for valid content
            mock_callback.assert_called_once_with("Valid")

    @pytest.mark.asyncio
    async def test_should_pass_correct_parameters_to_openai_client(
        self, mock_messages, mock_callback
    ):
        """Should pass correct parameters to OpenAI client"""
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create a mock stream
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = []
            mock_client.chat.completions.create.return_value = mock_stream
            
            await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify the correct parameters were passed to create()
            mock_client.chat.completions.create.assert_called_once_with(
                model="grok-4",
                messages=mock_messages,
                temperature=0,
                stream=True,
                max_tokens=8192,
                timeout=600
            )

    @pytest.mark.asyncio
    async def test_should_invoke_callback_for_each_content_chunk(
        self, mock_messages
    ):
        """Should invoke callback for each content chunk"""
        callback_results = []
        
        async def test_callback(content: str):
            callback_results.append(content)
        
        with patch('llm.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create multiple chunks
            chunks = [
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="First "),
                            finish_reason=None
                        )
                    ]
                ),
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567891,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="Second "),
                            finish_reason=None
                        )
                    ]
                ),
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567892,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="Third"),
                            finish_reason=None
                        )
                    ]
                )
            ]
            
            # Create a mock stream
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = chunks
            mock_client.chat.completions.create.return_value = mock_stream
            
            await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=test_callback,
                model=Llm.GROK_4
            )
            
            # Verify callback was called with each chunk's content
            assert callback_results == ["First ", "Second ", "Third"]

    @pytest.mark.asyncio
    async def test_should_return_proper_completion_structure(
        self, mock_messages, mock_callback
    ):
        """Should return proper Completion structure with duration and code"""
        with patch('llm.AsyncOpenAI') as mock_openai_class, \
             patch('llm.time.time') as mock_time:
            
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock time
            mock_time.side_effect = [100.0, 102.5]
            
            # Create chunks
            chunks = [
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="Test "),
                            finish_reason=None
                        )
                    ]
                ),
                ChatCompletionChunk(
                    id="test-id",
                    object="chat.completion.chunk",
                    created=1234567891,
                    model="grok-4",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content="response"),
                            finish_reason=None
                        )
                    ]
                )
            ]
            
            # Create a mock stream
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = chunks
            mock_client.chat.completions.create.return_value = mock_stream
            
            result = await stream_grok_response(
                messages=mock_messages,
                api_key="test-api-key",
                base_url=None,
                callback=mock_callback,
                model=Llm.GROK_4
            )
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert "duration" in result
            assert "code" in result
            assert result["duration"] == 2.5
            assert result["code"] == "Test response"