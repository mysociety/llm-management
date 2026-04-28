FROM python:3.11-trixie

ENV IN_DOCKER=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /workspaces/llm-management

COPY pyproject.toml poetry.lock ./
RUN curl -sSL https://install.python-poetry.org | python - && \
    poetry config virtualenvs.create false && \
    poetry self add poetry-bumpversion && \
    poetry install --no-root

COPY . .
ENV PORT=8080
RUN poetry install
CMD ["llm-management", "serve"]