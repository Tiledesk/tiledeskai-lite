from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage

)

import logging

logger = logging.getLogger(__name__)
# Function to get or create session history
def get_or_create_session_history(store, session_id, chat_history_dict):
    if session_id not in store:
        store[session_id] = load_session_history(chat_history_dict)
    return store[session_id]



def load_session_history(history) -> BaseChatMessageHistory:
    chat_history = ChatMessageHistory()
    if history is not None:
        for key, entry in history.items():
            chat_history.add_message(HumanMessage(content=entry.question))  # ('human', entry.question))
            chat_history.add_message(AIMessage(content=entry.answer))
    return chat_history


def _create_event(event_type: str, data: dict) -> str:
    import json
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"