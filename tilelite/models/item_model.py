from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, RootModel
from typing import Dict, Optional, List, Union, Any

class OllamaModel(BaseModel):
    name: str
    url: Optional[str] = Field(default_factory=lambda: "")
    dimension: Optional[int] = 1024 #qwel2-deepseek 3584, llama3.2 3072


class ChatEntry(BaseModel):
    question: str
    answer: str

class ChatHistory(BaseModel):
    chat_history: Dict[str, ChatEntry]

    @classmethod
    def from_dict(cls, data: dict) -> "ChatHistory":
        """Custom constructor to handle potential issues during initialization."""
        chat_history = {}
        for key, entry_data in data.items():
            try:
                if not isinstance(key, str):
                    raise ValueError(f"Invalid key type '{type(key)}'. Expected string.")
                chat_history[key] = ChatEntry(**entry_data)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"Error processing entry '{key}': {str(e)}")
        return cls(chat_history=chat_history)
    

class AWSAuthentication(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str


class QuestionToLLM(BaseModel):
    question: str
    llm_key: Union[str, AWSAuthentication]
    llm: str
    model: Union[str, OllamaModel] = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128),
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    debug: bool = Field(default_factory=lambda: False)
    thinking: Optional[Dict[str, Any]] = Field(default=None)
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)

    @field_validator("temperature")
    def temperature_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v

    @field_validator("n_messages")
    def n_messages_range(cls, v):
        """Ensures n_messages is within greater than 0"""
        if not v > 0:
            raise ValueError("n_messages must be greater than 0")
        return v

    @field_validator("max_tokens")
    def max_tokens_range(cls, v):
        """Ensures max_tokens is a positive integer."""
        if not 50 <= v <= 132000:
            raise ValueError("top_k must be a positive integer.")
        return v

    @field_validator("top_p")
    def top_p_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        return v


class ToolOptions(RootModel[Dict[str, Any]]):
    #__root__: Dict[str, Any] = Field(default_factory=dict)
    pass



class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoningn answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]






