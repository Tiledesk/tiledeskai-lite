import uuid
from datetime import datetime
from typing import List, AsyncGenerator

import fastapi
import asyncio
from fastapi.responses import JSONResponse

from langchain.chains import ConversationalRetrievalChain, LLMChain  # Deprecata

from langchain_core.documents import Document

from fastapi.responses import StreamingResponse

from tilelite.controller.controller_utils import get_or_create_session_history, _create_event
from tilelite.models.item_model import (SimpleAnswer,
                                        QuestionToLLM,
                                        ReasoningAnswer)


from tilelite.shared.utility import inject_llm, inject_reason_llm

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage

)

import logging

logger = logging.getLogger(__name__)

@inject_reason_llm
async def ask_reason_llm(question, chat_model=None):
    try:
        logger.info(question)
        chat_history_list = []

        if question.chat_history_dict is not None:
            for key, entry in question.chat_history_dict.items():
                chat_history_list.append(HumanMessage(content=entry.question))  # ('human', entry.question))
                chat_history_list.append(AIMessage(content=entry.answer))

        qa_prompt = ChatPromptTemplate.from_messages(
            [   MessagesPlaceholder("chat_history", n_messages=question.n_messages),
                ("human", "{input}")
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
            # return runnable_with_history
            async def get_stream_llm():
                full_response = ""
                full_response_reasoning= ""
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

                        is_reasoning_content, content_text, re_content= get_reasoning_content(chunk, question.llm)
                        full_response += content_text
                        if is_reasoning_content:
                           full_response_reasoning += re_content  # chunk.additional_kwargs.reasoning_content
                           yield _create_event("chunk", {"reasoning_content": re_content,
                                                          "message_id": message_id})
                        else:
                           yield _create_event("chunk", {"content": content_text, "message_id": message_id})

                        await asyncio.sleep(0.02)  # Per un flusso più regolare


                end_time = datetime.now()

                if not question.chat_history_dict:
                    question.chat_history_dict = {}

                num_question = len(question.chat_history_dict.keys())
                question.chat_history_dict[str(num_question)] = {"question": question.question, "answer": full_response}

                simple_reasoning_answer = ReasoningAnswer(answer=full_response,
                                                reasoning_content= full_response_reasoning,
                                                chat_history_dict=question.chat_history_dict)
                yield _create_event("metadata", {
                    "message_id": message_id,
                    "status": "completed",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "model_used": simple_reasoning_answer.model_dump()  # Sostituire con calcolo reale dei token
                })

            return StreamingResponse(
                get_stream_llm(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            logger.info(question)

            result = await runnable_with_history.ainvoke(
                {"input": question.question},  # 'chat_history_a': chat_history_list,
                config={"configurable": {"session_id": uuid.uuid4().hex}
                        },
            )
            # logger.info(result)

            if not question.chat_history_dict:
                question.chat_history_dict = {}


            _, content, reasoning_content=get_reasoning_content(result, question.llm)
            num = len(question.chat_history_dict.keys())
            question.chat_history_dict[str(num)] = {"question": question.question, "answer": content}
            return JSONResponse(
                content=ReasoningAnswer(answer=content,
                                        reasoning_content=reasoning_content,
                                        chat_history_dict=question.chat_history_dict).model_dump())


    except Exception as e:
        import traceback
        traceback.print_exc()

        result_to_return = SimpleAnswer(answer=repr(e),
                                        chat_history_dict={})
        raise fastapi.exceptions.HTTPException(status_code=400, detail=result_to_return.model_dump())


@inject_llm
async def ask_to_llm(question: QuestionToLLM, chat_model=None) :
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


def verify_answer(s):
    if s.endswith("<NOANS>"):
        s = s[:-7]  # Rimuove <NOANS> dalla fine della stringa
        success = False
    else:
        success = True
    return s, success


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




