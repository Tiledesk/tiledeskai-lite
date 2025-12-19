from fastapi import (FastAPI)
from tilelite.models.item_model import (QuestionToLLM,
                                        SimpleAnswer, QuestionToMCPAgent
                                        )


from tilelite.controller.controller import (ask_to_llm,
                                            ask_reason_llm,
                                            ask_mcp_agent_llm)

import logging


logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/api/ask", response_model=SimpleAnswer)
async def post_ask_to_llm_main(question: QuestionToLLM):
    """
    Query and Answer with a LLM
    :param question:
    :return: RetrievalResult
    """
    logger.info(question)

    return await ask_to_llm(question=question)


@app.post("/api/thinking", response_model=SimpleAnswer)
async def post_ask_to_llm_reason_main(question: QuestionToLLM):
    """
    Query and Answer with a LLM
    :param question:
    :return: RetrievalResult
    """
    logger.info(question)

    return await ask_reason_llm(question=question)


@app.post("/api/mcp-agent", response_model=SimpleAnswer)
async def post_ask_to_mcp_agent_main(question: QuestionToMCPAgent):
    """
    Query and Answer with a LLM
    :param question:
    :return: RetrievalResult
    """
    logger.info(question)

    return await ask_mcp_agent_llm(question=question)


@app.get("/")
async def get_root_endpoint():
    return "Hello from Tiledesk lite python server!!"


def main():
    import uvicorn
    uvicorn.run("tilelite.__main__:app", host="0.0.0.0", port=8001, reload=True, log_level="info")#, log_config=args.log_path


if __name__ == "__main__":
    main()
