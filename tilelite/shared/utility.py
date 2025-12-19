from functools import wraps

import logging
from typing import Callable

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from tilelite.models.item_model import LocalModel
from tilelite.shared import const

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq

from tilelite.shared.llm_config import get_llm_params

logger = logging.getLogger(__name__)

class LLMInjectionError(Exception):
    """Eccezione personalizzata per errori di injection LLM"""
    pass

def inject_llm(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        try:
            if question.llm == "openai":
                chat_model = ChatOpenAI(api_key=question.llm_key,
                                        model=question.model,
                                        temperature=question.temperature,
                                        max_tokens=question.max_tokens,
                                        top_p=question.top_p)

            elif question.llm == "anthropic":
                chat_model = ChatAnthropic(anthropic_api_key=question.llm_key,
                                           model=question.model,
                                           temperature=question.temperature,
                                           max_tokens=question.max_tokens,
                                           top_p=question.top_p)

            elif question.llm == "cohere":
                chat_model = ChatCohere(cohere_api_key=question.llm_key,
                                        model=question.model,
                                        temperature=question.temperature,
                                        max_tokens=question.max_tokens
                                        )
                #p è compreso tra 0.00 e 0.99

            elif question.llm == "google":
                chat_model = ChatGoogleGenerativeAI(google_api_key=question.llm_key,
                                                    model=question.model,
                                                    temperature=question.temperature,
                                                    max_tokens=question.max_tokens,
                                                    top_p=question.top_p,
                                                    convert_system_message_to_human=True)

                #ai_msg.usage_metadata per controllare i token

            elif question.llm == "ollama":
                chat_model = ChatOllama(model = question.model.name,
                                        temperature=question.temperature,
                                        num_predict=question.max_tokens,
                                        top_p=question.max_tokens,
                                        base_url=question.model.url)

            elif question.llm == "groq":
                chat_model = ChatGroq(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      max_tokens=question.max_tokens,
                                      top_p=question.top_p
                                      )

            elif question.llm == "deepseek":
                chat_model = ChatDeepSeek(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      max_tokens=question.max_tokens,
                                      top_p=question.top_p
                                      )

            elif question.llm == "vllm":
                chat_model = ChatOpenAI(model = question.model.name,
                                        api_key=question.llm_key,
                                        temperature=question.temperature,
                                        max_tokens=question.max_tokens,
                                        top_p=question.top_p,
                                        base_url=question.model.url)


            else:
                chat_model = ChatOpenAI(api_key=question.llm_key,
                                        model=question.model,
                                        temperature=question.temperature,
                                        max_tokens=question.max_tokens,
                                        top_p=question.top_p
                                        )

            # Add chat_model agli kwargs
            kwargs['chat_model'] = chat_model

            # Chiama la funzione originale con i nuovi kwargs
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore nel decorator inject_llm per {func.__name__}: {e}", exc_info=True)
            raise LLMInjectionError(f"LLM injection failed: {e}") from e

    return wrapper

def inject_reason_llm_old(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        if question.llm == "openai":

            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_completion_tokens=question.max_tokens)
        elif question.llm == "anthropic":
            chat_model = ChatAnthropic(anthropic_api_key=question.llm_key,
                                       model=question.model,
                                       temperature=question.temperature,
                                       thinking=question.thinking,
                                       max_tokens=question.max_tokens)
        elif question.llm == "deepseek":
            chat_model = ChatDeepSeek(api_key=question.llm_key,
                                      model=question.model,
                                      temperature=question.temperature,
                                      #max_tokens=question.max_tokens
                                      )

        else:
            chat_model = ChatOpenAI(api_key=question.llm_key,
                                    model=question.model,
                                    temperature=question.temperature,
                                    max_completion_tokens=question.max_tokens)



        # Add chat_model agli kwargs
        kwargs['chat_model'] = chat_model

        # Chiama la funzione originale con i nuovi kwargs
        return await func(question, *args, **kwargs)

    return wrapper


def inject_reason_llm(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
        model_name_param = None
        base_url_param = None
        try:
            llm_params = get_llm_params(
                provider=question.llm,
                temperature=question.temperature,
                top_p=question.top_p,
                max_tokens=question.max_tokens
            )
            if isinstance(question.model, LocalModel):
                model_name_param = question.model.name
                base_url_param = question.model.url
            else:
                model_name_param = question.model if isinstance(question.model, str) else (
                    question.model.name if hasattr(question.model, 'name') else None)
                base_url_param = question.model.url if hasattr(question.model, 'url') else None

            client_base_config = {"model": model_name_param,
                                  "base_url": base_url_param,
                                  **llm_params # Include generic LLM parameters
                                  }



            client_config = {**llm_params, "api_key": question.llm_key}

            if client_base_config.get("model"):
                client_config["model"] = client_base_config["model"]
            if client_base_config.get("base_url"):
                client_config["base_url"] = client_base_config["base_url"]

            # Estrai la configurazione reasoning se presente
            thinking_config = question.thinking if hasattr(question, 'thinking') and question.thinking else None

            if question.llm == "openai":
                from langchain_openai import ChatOpenAI
                client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)

                # Aggiungi parametri specifici per GPT-5 reasoning
                if thinking_config:
                    # Costruisci il dizionario reasoning se ci sono parametri
                    reasoning_dict = {}
                    if thinking_config.reasoning_effort:
                        reasoning_dict["effort"] = thinking_config.reasoning_effort
                    if thinking_config.reasoning_summary:
                        reasoning_dict["summary"] = thinking_config.reasoning_summary

                    # Passa il dizionario reasoning solo se ha contenuti
                    if reasoning_dict:
                        client_config["reasoning"] = reasoning_dict
                        # Usa il nuovo formato response per reasoning models
                        client_config["output_version"] = "responses/v1"

                chat_model = ChatOpenAI(**client_config)

            elif question.llm == "anthropic":
                from langchain_anthropic import ChatAnthropic
                client_config["anthropic_api_key"] = client_config.pop("api_key", None)

                # Aggiungi parametri specifici per Claude thinking
                if thinking_config:
                    thinking_dict = {}
                    if thinking_config.type:
                        thinking_dict["type"] = thinking_config.type
                    if thinking_config.budget_tokens:
                        thinking_dict["budget_tokens"] = thinking_config.budget_tokens
                    if thinking_dict:
                        client_config["thinking"] = thinking_dict

                chat_model = ChatAnthropic(**client_config)

            elif question.llm == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                client_config["google_api_key"] = client_config.pop("api_key", None)

                # Aggiungi parametri specifici per Gemini thinking
                if thinking_config:
                    if thinking_config.thinkingBudget is not None:
                        # Gemini 2.5 Pro usa thinkingBudget
                        client_config["thinking_budget"] = thinking_config.thinkingBudget
                    elif thinking_config.thinkingLevel:
                        # Gemini 3.0 Pro usa thinkingLevel
                        client_config["thinking_level"] = thinking_config.thinkingLevel

                chat_model = ChatGoogleGenerativeAI(**client_config)

            elif question.llm == "deepseek":
                from langchain_deepseek import ChatDeepSeek
                # DeepSeek non ha parametri specifici per reasoning, è automatico
                chat_model = ChatDeepSeek(**client_config)

            elif question.llm == "vllm":
                from langchain_openai import ChatOpenAI  # vLLM uses OpenAI compatible API
                client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
                chat_model = ChatOpenAI(**client_config)

            else:
                logger.warning(f"LLM provider '{question.llm}' not optimized for reasoning, falling back to OpenAI")
                from langchain_openai import ChatOpenAI
                client_config["max_completion_tokens"] = client_config.pop("max_tokens", None)
                chat_model = ChatOpenAI(**client_config)

            kwargs['chat_model'] = chat_model

            # Chiama la funzione originale con i nuovi kwargs
            return await func(question, *args, **kwargs)

        except Exception as e:
            logger.error(f"Errore nel decorator inject_reason_llm per {func.__name__}: {e}", exc_info=True)
            raise LLMInjectionError(f"Reasoning LLM injection failed: {e}") from e


    return wrapper



def decode_jwt(token:str):
    import jwt
    jwt_secret_key = const.JWT_SECRET_KEY
    return jwt.decode(jwt=token, key=jwt_secret_key, algorithms=['HS256'])


