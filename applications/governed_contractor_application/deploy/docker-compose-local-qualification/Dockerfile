# © Artur Czarnecki. All rights reserved.
# syntax=docker/dockerfile:1
#
# LOCAL QUALIFICATION overlay — not a production deployment profile.
# Requires a materialized GCA runtime image (see RUNBOOK.md):
#   uv run python scripts/build/build_application_image.py \
#     --application governed_contractor_application \
#     --tag governed_contractor-application:latest \
#     --context-dir applications/governed_contractor_application/docker/runtime-context \
#     --keep-context

ARG BASE_IMAGE=governed_contractor-application:latest
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="intergrax-gca-docker-local-qualification" \
      org.opencontainers.image.description="OD-R2-03 Docker Compose local qualification overlay (not production-qualified)"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sqlite3 \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10002 intergrax-gca \
    && useradd --create-home --uid 10002 --gid 10002 --shell /bin/sh intergrax-gca \
    && install -d -o intergrax-gca -g intergrax-gca -m 0750 /opt/intergrax/bin

COPY intergrax-gca-docker-local-qualification-entrypoint.sh /opt/intergrax/bin/intergrax-gca-docker-local-qualification-entrypoint.sh
COPY target-storage-probe.sh /opt/intergrax/bin/target-storage-probe.sh

RUN sed -i 's/\r$//' /opt/intergrax/bin/intergrax-gca-docker-local-qualification-entrypoint.sh \
    && sed -i 's/\r$//' /opt/intergrax/bin/target-storage-probe.sh \
    && chmod 750 /opt/intergrax/bin/intergrax-gca-docker-local-qualification-entrypoint.sh \
    && chmod 750 /opt/intergrax/bin/target-storage-probe.sh \
    && chown -R intergrax-gca:intergrax-gca /app

USER intergrax-gca
WORKDIR /app/applications/governed_contractor_application

ENV INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR=/var/lib/intergrax/production-v1/data \
    INTERGRAX_SINGLE_NODE_PRODUCTION_V1_SERVICE_RUNTIME_USER=intergrax-gca \
    INTERGRAX_SINGLE_NODE_PRODUCTION_V1_ENFORCE_SINGLE_ACTIVE_WRITER=1 \
    GOVERNED_CONTRACTOR_BACKEND_ENV=prod

ENTRYPOINT ["/opt/intergrax/bin/intergrax-gca-docker-local-qualification-entrypoint.sh"]
CMD ["serve"]
