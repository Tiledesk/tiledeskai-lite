import json
import uuid
from datetime import datetime
from typing import List, AsyncGenerator, Dict, Union

import fastapi
import asyncio

import mcp
from fastapi.responses import JSONResponse
from langchain.agents import create_agent

from langchain_classic.chains import ConversationalRetrievalChain, LLMChain  # Deprecata

from langchain_core.documents import Document

from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient, StdioConnection, SSEConnection, WebsocketConnection, \
    StreamableHttpConnection
from langgraph.prebuilt import create_react_agent
from langgraph_sdk.schema import Command

from tilelite.controller.controller_utils import get_or_create_session_history, _create_event
from tilelite.models.item_model import (SimpleAnswer,
                                        QuestionToLLM,
                                        ReasoningAnswer, QuestionToMCPAgent, PromptTokenInfo, ChatEntry)
from tilelite.shared import const

from tilelite.shared.utility import inject_llm, inject_reason_llm

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


from langchain_classic.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage

)

import logging

logger = logging.getLogger(__name__)



@inject_reason_llm
async def ask_reason_llm(question, chat_model=None):
    """
    Gestisce richieste a LLM con supporto per reasoning content (es. DeepSeek, o1).
    Usa i metodi privati centralizzati per stream e history.

    :param question: QuestionToLLM
    :param chat_model: Il modello LLM
    :return: ReasoningAnswer in streaming o JSON
    """
    try:
        logger.info(question)

        # Costruisce il prompt template con history
        qa_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history", n_messages=question.n_messages),
            ("human", "{input}")
        ])

        # Setup session history
        store = {}
        #get_session_history = lambda session_id: get_or_create_session_history(
        #    store, session_id, question.chat_history_dict
        #)
        def get_session_history(session_id):
            return get_or_create_session_history(store,
                                                 session_id,
                                                 question.chat_history_dict
                                                 )

        # Crea il runnable con history
        runnable = qa_prompt | chat_model
        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        # Configurazione per il runnable
        config = {"configurable": {"session_id": uuid.uuid4().hex}}
        input_data = {"input": question.question}

        # --- STREAMING ---
        if question.stream:
            return StreamingResponse(
                _stream_generic_response(
                    runnable_with_history,
                    input_data,
                    question,
                    config=config,
                    chunk_processor=_reasoning_chunk_processor,
                    response_class=ReasoningAnswer
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        # --- RISPOSTA SINCRONA ---
        if question.structured_output and question.output_schema:
            structured_llm = chat_model.with_structured_output(question.output_schema)
            runnable = qa_prompt | structured_llm
            runnable_with_history = RunnableWithMessageHistory(
                runnable,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            result = await runnable_with_history.ainvoke(input_data, config=config)

            if hasattr(result, 'model_dump'):
                answer_content = result.model_dump()
            else:
                answer_content = result

            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                json.dumps(answer_content)
            )

            return JSONResponse(
                content=ReasoningAnswer(
                    answer=answer_content,
                    reasoning_content="Structured output does not have reasoning content.",
                    chat_history_dict=updated_history
                ).model_dump()
            )
        else:
            result = await runnable_with_history.ainvoke(input_data, config=config)

            # Estrai content e reasoning content
            _, content, reasoning_content = get_reasoning_content(result, question.llm)

            # Converti content in stringa se è una lista (formato responses/v1 di OpenAI)
            if isinstance(content, list):
                # Estrai il testo dalle risposte
                content_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            content_parts.append(item['text'])
                        elif 'content' in item:
                            content_parts.append(item['content'])
                content = ''.join(content_parts) if content_parts else str(content)
            elif not isinstance(content, str):
                content = str(content)

            # Aggiorna history usando il metodo centralizzato
            updated_history = _update_history(
                question.chat_history_dict,
                question.question,
                content
            )

            return JSONResponse(
                content=ReasoningAnswer(
                    answer=content,
                    reasoning_content=reasoning_content,
                    chat_history_dict=updated_history
                ).model_dump()
            )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in ask_reason_llm: {str(e)}\n{error_traceback}")

        # Determina il tipo di errore e il messaggio appropriato
        error_message = str(e)
        error_type = type(e).__name__

        # Gestisci errori specifici con messaggi più chiari
        if "unexpected keyword argument" in error_message:
            error_message = f"Configuration error: {error_message}. Please check the 'thinking' parameters for your chosen model."
        elif "API key" in error_message or "authentication" in error_message.lower():
            error_message = f"Authentication error: {error_message}"
        elif "rate limit" in error_message.lower():
            error_message = f"Rate limit exceeded: {error_message}"
        elif "timeout" in error_message.lower():
            error_message = f"Request timeout: {error_message}"

        # Restituisci errore come ReasoningAnswer con status 400
        error_response = ReasoningAnswer(
            answer=f"Error ({error_type}): {error_message}",
            reasoning_content="",
            chat_history_dict=question.chat_history_dict if hasattr(question, 'chat_history_dict') else {}
        )

        return JSONResponse(status_code=400,content=error_response.model_dump())


@inject_llm
async def ask_to_llm(question: QuestionToLLM , chat_model=None) :
    try:
        logger.info(question)
        chat_history_list = []

        if question.chat_history_dict is not None:
            for key, entry in question.chat_history_dict.items():
                chat_history_list.append(HumanMessage(content=entry.question))  # ('human', entry.question))
                chat_history_list.append(AIMessage(content=entry.answer))



        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question.system_context),
                MessagesPlaceholder("chat_history", n_messages=question.n_messages),
                ("human", "{input}"),
            ]
        )

        store = {}
        get_session_history = lambda session_id: get_or_create_session_history(store, session_id,
                                                                               question.chat_history_dict)



        runnable = qa_prompt | chat_model

        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"

        )

        if question.stream:

            #return runnable_with_history
           async def get_stream_llm():
               full_response = ""
               message_id = str(uuid.uuid4())
               start_time = datetime.now()

               yield _create_event("metadata", {
                   "message_id": message_id,
                   "status": "started",
                   "timestamp": start_time.isoformat()
               })

               async for chunk in runnable_with_history.astream({"input": question.question},
                                                                config={
                                                                    "configurable": {"session_id": uuid.uuid4().hex}}):
                   if hasattr(chunk, 'content'):
                       full_response += chunk.content
                       yield _create_event("chunk", {"content": chunk.content, "message_id": message_id})
                       await asyncio.sleep(0.02)  # Per un flusso più regolare

               end_time = datetime.now()

               if not question.chat_history_dict:
                   question.chat_history_dict = {}

               num_question = len(question.chat_history_dict.keys())
               question.chat_history_dict[str(num_question)] = {"question": question.question, "answer": full_response}

               simple_answer = SimpleAnswer(answer=full_response, chat_history_dict=question.chat_history_dict)
               yield _create_event("metadata", {
                   "message_id": message_id,
                   "status": "completed",
                   "timestamp": end_time.isoformat(),
                   "duration": (end_time - start_time).total_seconds(),
                   "full_response": full_response,
                   "model_used": simple_answer.model_dump() # Sostituire con calcolo reale dei token
               })

           return StreamingResponse(
               get_stream_llm(),
               media_type="text/event-stream",
               headers={"Cache-Control": "no-cache"}
           )


        else:
            result = await runnable_with_history.ainvoke(
                {"input": question.question}, # 'chat_history_a': chat_history_list,
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },
            )
            # logger.info(result)

            if not question.chat_history_dict:
                question.chat_history_dict = {}

            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] = {"question": question.question, "answer": result.content}


            return JSONResponse(content=SimpleAnswer(answer=result.content, chat_history_dict=question.chat_history_dict).model_dump())


    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={})
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())

@inject_llm
async def ask_mcp_agent_llm(question: QuestionToMCPAgent, chat_model=None):
    try:
        mcp_client = question.create_mcp_client()
        all_tools = await mcp_client.get_tools()

        base_agent = create_agent(
            model=chat_model,
            tools=all_tools,
            system_prompt=question.system_context if question.system_context else const.react_prompt_template,
            #middleware=[pre_model_cleaning_middleware]  # ← CHIAVE! Pulisce ad ogni step
        )

        # Invoca l'agent
        logger.info("Invoking agent...")
        agent_input =  {"messages":question.question}
        response = await base_agent.ainvoke(agent_input)
        result = extract_conversation_flow(response['messages'])

        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for msg in response.get('messages', []):
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                total_input_tokens += msg.usage_metadata.get('input_tokens', 0)
                total_output_tokens += msg.usage_metadata.get('output_tokens', 0)
                total_tokens += msg.usage_metadata.get('total_tokens', 0)

        # Crea l'oggetto PromptTokenInfo
        prompt_token_info = PromptTokenInfo(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens
        )

        logger.info(f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")

        response_data = SimpleAnswer(
            answer=result.get("ai_message", "No answer"),  # Assegna 'ai_message' ad 'answer'
            tools_log=result.get("tools"),  # Assegna 'tools' a 'tools_log'
            chat_history_dict={},
            prompt_token_info=prompt_token_info
        )

        return JSONResponse(
            content=response_data.model_dump()
        )

    except Exception as e:
        return handle_agent_exception(e, "ask_mcp_agent_llm_simple")

def verify_answer(s):
    if s.endswith("<NOANS>"):
        s = s[:-7]  # Rimuove <NOANS> dalla fine della stringa
        success = False
    else:
        success = True
    return s, success



def extract_conversation_flow(messages: list) -> dict:
    """
    Estrae tutti i ToolMessage e AIMessage dalla conversazione.

    Formato output:
    {
        "tools": ["contenuto tool 1", "contenuto tool 2", ...],
        "ai_message": "risposta dell'AI"
    }
    """
    tools = []
    ai_message = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Estrae il contenuto testuale (potrebbe avere multiple parti)
            main_text = ""
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        main_text = part.get('text', '')
                        break
            else:
                main_text = str(msg.content)

            # Preserva la formattazione originale (NON rimuovere newline!)
            cleaned_text = main_text.strip()
            if cleaned_text:
                ai_message = cleaned_text

        elif isinstance(msg, ToolMessage):
            # Preserva il contenuto del tool completo
            tool_content = str(msg.content).strip()
            if tool_content:
                tools.append(tool_content)

    return {
        "tools": tools,
        "ai_message": ai_message
    }


def handle_agent_exception(e: Exception, context: str = "Agent execution") -> JSONResponse:
    """
    Gestisce le eccezioni nelle funzioni agent restituendo messaggi user-friendly.

    Args:
        e: L'eccezione catturata
        context: Contesto dell'errore per il logging

    Returns:
        JSONResponse con messaggio di errore chiaro per l'utente
    """
    import traceback
    traceback.print_exc()
    logger.error(f"Error in {context}: {str(e)}")

    # Determina il tipo di errore e crea un messaggio user-friendly
    error_message = str(e)
    user_message = ""
    status_code = 500

    # Errori di file non trovato
    if "not found" in error_message.lower() or "does not exist" in error_message.lower():
        user_message = "Il file richiesto non è stato trovato. Verifica che il file sia stato caricato correttamente."
        status_code = 404

    # Errori di tool/MCP
    elif "tool" in error_message.lower() and ("error" in error_message.lower() or "failed" in error_message.lower()):
        user_message = f"Errore nell'esecuzione del tool: {error_message}"
        status_code = 400

    # Errori di formato base64
    elif "base64" in error_message.lower() or "invalid" in error_message.lower():
        user_message = "Errore nel processare il contenuto multimediale. Verifica che i file siano nel formato corretto."
        status_code = 400

    # Errori di modello/API
    elif "model" in error_message.lower() or "api" in error_message.lower() or "rate" in error_message.lower():
        user_message = f"Errore nella comunicazione con il modello AI: {error_message}"
        status_code = 503

    # Errori di timeout
    elif "timeout" in error_message.lower():
        user_message = "L'operazione ha impiegato troppo tempo. Riprova con una richiesta più semplice."
        status_code = 504

    # Errore generico
    else:
        user_message = f"Si è verificato un errore: {error_message}"
        status_code = 500

    error_response = SimpleAnswer(
        answer=user_message,
        tools_log=[],
        chat_history_dict={},
        prompt_token_info=PromptTokenInfo(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


def load_session_history(history) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    if history is not None:
        for key, entry in history.items():
            chat_history.add_message(HumanMessage(content=entry.question))  # ('human', entry.question))
            chat_history.add_message(AIMessage(content=entry.answer))
    return chat_history


def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Source: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


def _update_history(current_history: dict, new_question, new_answer) -> dict:
    """Aggiorna la history con la nuova domanda/risposta"""
    if not current_history:
        current_history = {}

    next_key = str(len(current_history))
    current_history[next_key] = ChatEntry(
        question=new_question,
        answer=new_answer
    )

    return current_history


def get_reasoning_content(chunk, llm):
    """
    Verifica se la chiave 'reasoning_content' esiste nel dizionario annidato
    'additional_kwargs' di un chunk e restituisce il suo valore.

    Args:
        chunk (dict): Il dizionario contenente i dati.
        llm: llm usato

    Returns:
        str or None: Il valore di 'reasoning_content' se esiste, altrimenti None.
    """
    if llm=="openai":
        return False, chunk.content, ''
    elif llm=="deepseek":
        if 'reasoning_content' in chunk.additional_kwargs:
            return True, chunk.content, chunk.additional_kwargs['reasoning_content']
        else:
            return False, chunk.content, ''
    elif llm=="anthropic":
        full_thinking=""
        full_text=""
        is_thinking = False
        if not chunk.content:  # Controlla se la lista è vuota
            return False, full_text, full_thinking
        for text_element in chunk.content:
            if 'thinking' in text_element:
                full_thinking += text_element['thinking']
                is_thinking = True
            if 'text' in text_element:
                full_text += text_element['text']
                is_thinking = False

        return is_thinking, full_text, full_thinking

    else:
        return False, '', ''


def _reasoning_chunk_processor(chunk, question):
    """
    Processor per chunk con reasoning content (es. DeepSeek, Claude, Gemini).

    Args:
        chunk: Il chunk ricevuto dallo stream
        question: QuestionToLLM object con la configurazione

    Returns:
        dict con chiavi: 'content', 'reasoning_content', 'events'
    """
    is_reasoning, content_text, reasoning_text = get_reasoning_content(chunk, question.llm)

    # Controlla se mostrare il thinking nello stream
    show_thinking = True  # Default
    if hasattr(question, 'thinking') and question.thinking is not None:
        show_thinking = question.thinking.show_thinking_stream

    events = []
    if is_reasoning:
        # Se show_thinking_stream è True, invia il reasoning nello stream
        if show_thinking:
            events.append({"reasoning_content": reasoning_text})
        # Altrimenti non inviare eventi, ma accumula comunque il reasoning_content
    else:
        # Invia sempre il content normale
        events.append({"content": content_text})

    return {
        'content': content_text,
        'reasoning_content': reasoning_text,
        'events': events
    }


async def _stream_generic_response(
        runnable,
        input_data,
        question: QuestionToLLM,
        config: dict = None,
        chunk_processor=None,
        response_class=None
):
    """
    Metodo generico per gestire streaming di qualsiasi runnable.

    Args:
        runnable: Il runnable da eseguire (chat_model, RunnableWithMessageHistory, agent, etc.)
        input_data: I dati di input da passare al runnable (dict o list di messaggi)
        question: L'oggetto QuestionToLLM con i parametri della richiesta
        config: Configurazione opzionale per il runnable (default: None)
        chunk_processor: Funzione opzionale che prende (chunk, question) e ritorna dict con chiavi:
                        - 'content': contenuto principale da accumulare
                        - 'events': lista di eventi SSE da emettere
                        Se None, usa il processing standard (solo 'content')
        response_class: Classe per la risposta finale (default: SimpleAnswer)

    Yields:
        Eventi SSE formattati
    """
    if response_class is None:
        response_class = SimpleAnswer

    full_response = ""
    additional_data = {}  # Per dati extra come reasoning_content
    message_id = str(uuid.uuid4())
    start_time = datetime.now()

    # Metadati iniziali
    yield _create_event("metadata", {
        "message_id": message_id,
        "status": "started",
        "timestamp": start_time.isoformat()
    })

    # Stream dei chunk (con o senza config)
    if config is not None:
        stream = runnable.astream(input_data, config=config)
    else:
        stream = runnable.astream(input_data)

    async for chunk in stream:
        if hasattr(chunk, 'content'):
            if chunk_processor:
                # Usa il processor custom per casi speciali (es. reasoning)
                result = chunk_processor(chunk, question)
                full_response += result.get('content', '')

                # Accumula dati extra
                for key, value in result.items():
                    if key not in ['content', 'events']:
                        if key not in additional_data:
                            additional_data[key] = ''
                        additional_data[key] += value

                # Emetti gli eventi custom
                for event_data in result.get('events', []):
                    yield _create_event("chunk", {**event_data, "message_id": message_id})
            else:
                # Processing standard: solo content
                full_response += chunk.content
                yield _create_event("chunk", {
                    "content": chunk.content,
                    "message_id": message_id
                })

            await asyncio.sleep(0.01)

    # Aggiorna history usando il metodo centralizzato
    updated_history = _update_history(
        question.chat_history_dict,
        question.question,
        full_response
    )

    # Token info (può essere 0 nello streaming)
    prompt_token_info = PromptTokenInfo(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0
    )

    end_time = datetime.now()

    # Costruisci la risposta finale
    response_data = {
        'answer': full_response,
        'chat_history_dict': updated_history,
        'prompt_token_info': prompt_token_info
    }
    response_data.update(additional_data)

    # Metadati finali
    yield _create_event("metadata", {
        "message_id": message_id,
        "status": "completed",
        "timestamp": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "full_response": full_response,
        "model_used": response_class(**response_data).model_dump()
    })
