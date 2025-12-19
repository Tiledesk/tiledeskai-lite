from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, RootModel, SecretStr
from typing import Dict, Optional, List, Union, Any, Literal


class PromptTokenInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class LocalModel(BaseModel):
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

class ReasoningConfig(BaseModel):
    """
    Configurazione unificata per reasoning models.
    Supporta parametri specifici per ogni provider:
    - OpenAI (GPT-5): reasoning_effort, reasoning_summary
    - Anthropic (Claude): type, budget_tokens
    - Google (Gemini 2.5): thinkingBudget
    - Google (Gemini 3.0): thinkingLevel
    - DeepSeek: nessun parametro specifico
    """
    # Controllo visibilità thinking nello stream
    show_thinking_stream: bool = Field(
        default=True,
        description="Se True, mostra il thinking content nello stream. Se False, lo nasconde ma lo include nella risposta finale"
    )

    # OpenAI GPT-5 specific
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="OpenAI GPT-5: Effort level for reasoning (low, medium, high)"
    )
    reasoning_summary: Optional[Literal["auto", "always", "never"]] = Field(
        default=None,
        description="OpenAI GPT-5: When to include reasoning summary (auto, always, never)"
    )

    # Anthropic Claude specific
    type: Optional[Literal["enabled", "disabled"]] = Field(
        default=None,
        description="Anthropic Claude: Enable/disable thinking mode"
    )
    budget_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Anthropic Claude: Token budget for thinking (0-100000)"
    )

    # Google Gemini 2.5 specific
    thinkingBudget: Optional[int] = Field(
        default=None,
        description="Gemini 2.5: Thinking token budget. -1=dynamic, 0=disabled, positive=specific budget (max 32000)"
    )

    # Google Gemini 3.0 specific
    thinkingLevel: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Gemini 3.0: Thinking level (low, medium, high)"
    )

    @field_validator("thinkingBudget")
    @classmethod
    def validate_thinking_budget(cls, v):
        """Valida il thinkingBudget per Gemini 2.5"""
        if v is not None:
            if v < -1:
                raise ValueError("thinkingBudget must be >= -1")
            if v > 32000:
                raise ValueError("thinkingBudget cannot exceed 32000")
        return v

class QuestionToLLM(BaseModel):
    question: str
    llm_key: Union[str, AWSAuthentication]
    llm: str
    model: Union[str, LocalModel] = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128),
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    debug: bool = Field(default_factory=lambda: False)
    thinking: Optional[ReasoningConfig] = Field(
        default=None,
        description="Reasoning configuration for advanced models (GPT-5, Claude 4/4.5, Gemini 2.5/3.0, DeepSeek)"
    )
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    structured_output: Optional[bool] = Field(default=False)
    output_schema: Optional[Any] = Field(default=None)
    chat_history_dict: Optional[Dict[str, ChatEntry]] = None
    n_messages: int = Field(default_factory=lambda: None)

    # Modalità di gestione history
    contextualize_prompt: Optional[bool] = Field(
        default=False, description="Se True, inietta la history come testo nel system prompt. Se False, passa la "
                                   "history come messaggi strutturati (consigliato per LLM moderni)"
    )

    # Limitazione history
    max_history_messages: Optional[int] = Field(
        default=None,
        description="Numero massimo di turni (coppie domanda/risposta) da mantenere. "
                    "None = illimitato. Es: 10 = ultimi 10 turni (20 messaggi)"
    )

    # Summarization
    summarize_old_history: bool = Field(
        default=False,
        description="Se True e max_history_messages è impostato, riassume automaticamente "
                    "la history più vecchia invece di scartarla. Richiede una chiamata LLM extra."
    )

    @model_validator(mode="after")
    def adjust_temperature_and_validate(self):
        # Ricava il nome del modello come stringa
        model_name: Optional[str] = None
        if isinstance(self.model, str):
            model_name = self.model
        elif isinstance(self.model, LocalModel):
            model_name = self.model.name

        # Se è gpt-5 o gpt-5-*, forza temperature a 1.0
        if model_name and model_name.startswith("gpt-5"):
            self.temperature = 1.0
            self.top_p = None
            return self

        # Se è claude-4 o claude-sonnet-4-*, rimuovi top_p se presente
        if model_name and ("claude-4" in model_name or "claude-sonnet-4" in model_name):
            if self.temperature is not None and self.top_p is not None:
                self.top_p = None

        # Se entrambi sono None, imposta default temperature
        if self.temperature is None and self.top_p is None:
            self.temperature = 0.0

        # Se entrambi sono specificati, gestisci in base al provider
        elif self.temperature is not None and self.top_p is not None:
            # Provider che richiedono temperature (non può essere None)
            if self.llm in ["google"]:
                # Mantieni temperature, rimuovi top_p
                self.top_p = None
            # Provider che richiedono top_p
            elif self.llm in []:  # Aggiungi qui provider che richiedono top_p
                self.temperature = None
            # Provider che accettano entrambi
            elif self.llm in ["openai", "vllm", "deepseek", "ollama"]:
                # Mantieni entrambi
                pass
            # Anthropic: gestione speciale per claude-4
            elif self.llm in ["anthropic","groq"]:
                # claude-4 non supporta entrambi, mantieni solo temperature
                self.top_p = None
            # Provider che non supportano top_p
            elif self.llm in ["cohere"]:
                self.top_p = None
            else:
                # Default: priorità a temperature
                self.top_p = None

        # Valida i range
        if self.temperature is not None and not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")

        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0.")

        return self

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


class ServerConfig(BaseModel):
    """Modello per la configurazione di un server MCP"""
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    api_key: Optional[str] = None
    parameters: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_transport_specific_fields(self):
        # Validazione per trasporto SSE
        if self.transport == "sse":
            if not self.url:
                raise ValueError("URL è obbligatorio per il trasporto SSE")

        # Validazione per trasporto stdio
        elif self.transport == "stdio":
            if not self.command or not self.args:
                raise ValueError("Command e args sono obbligatori per il trasporto stdio")

        return self

class QuestionToMCPAgent(BaseModel):
    question: str
    llm_key: Union[str, AWSAuthentication]
    llm: str
    model: Union[str, LocalModel] = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=128),
    top_p: Optional[float] = Field(default=1.0)
    stream: Optional[bool] = Field(default_factory=lambda: False)
    debug: bool = Field(default_factory=lambda: False)
    system_context: str = Field(default="You are a helpful AI bot. Always reply in the same language of the question.")
    servers: Dict[str, ServerConfig] = Field(default_factory=dict)

    @field_validator("temperature")
    def temperature_range(cls, v):
        """Ensures temperature is within valid range (usually 0.0 to 1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
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

    def create_mcp_client(self):
        """Crea un'istanza di MultiServerMCPClient dalla configurazione"""
        config_dict = {
            name: server_config.model_dump(exclude_unset=True)
            for name, server_config in self.servers.items()
        }
        return MultiServerMCPClient(config_dict)


class ToolOptions(RootModel[Dict[str, Any]]):
    #__root__: Dict[str, Any] = Field(default_factory=dict)
    pass



class SimpleAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    chat_history_dict: Optional[Dict[str, ChatEntry]]
    prompt_token_info: Optional[PromptTokenInfo] = None

class ReasoningAnswer(BaseModel):
    answer: Union[str, Dict[str, Any], list] = Field(default="No answer")
    reasoning_content: Union[str, Dict[str, Any], list] = Field(default="No reasoning answer")
    chat_history_dict: Optional[Dict[str, "ChatEntry"]] = None
    prompt_token_info: Optional[PromptTokenInfo] = None






