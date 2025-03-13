from functools import wraps

import logging

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from tilelite.shared import const

from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def inject_llm(func):
    @wraps(func)
    async def wrapper(question, *args, **kwargs):
        logger.debug(question)
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

    return wrapper

def inject_reason_llm(func):
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

def decode_jwt(token:str):
    import jwt
    jwt_secret_key = const.JWT_SECRET_KEY
    return jwt.decode(jwt=token, key=jwt_secret_key, algorithms=['HS256'])


