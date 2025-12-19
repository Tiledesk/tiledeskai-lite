# STAGE 1: LLM
FROM python:3.13

WORKDIR /tiledeskai-lite

COPY log_conf.json /tiledeskai-lite/log_conf.json
COPY pyproject.toml /tiledeskai-lite/pyproject.toml
COPY ./tilelite /tiledeskai-lite/tilelite


RUN pip install .
RUN pip install "uvicorn[standard]" gunicorn

COPY entrypoint.sh /tiledeskai-lite/entrypoint.sh
RUN chmod +x /tiledeskai-lite/entrypoint.sh


ENTRYPOINT ["/tiledeskai-lite/entrypoint.sh"]

EXPOSE 8000

