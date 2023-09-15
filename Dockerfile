FROM python:3.10-buster as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.10-slim-buster as runtime
LABEL maintainer="Nikolay Yurin <yurinnick@meta.com>"

RUN apt update && \
    apt install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY kernel_patches_daemon /app/kernel_patches_daemon
COPY configs /app/configs

VOLUME /app/configs

WORKDIR /app
ENTRYPOINT [ \
    "python", "-m", "kernel_patches_daemon", \
    "--label-colors=/app/configs/labels.json", \
    "--config=/app/configs/kpd.json" \
]
