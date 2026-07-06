#!/bin/sh
# © Artur Czarnecki. All rights reserved.
# Local Sentry proof only: load bootstrap DSN at container process start time.
set -eu

if [ ! -f /proof/generated.env ]; then
  echo "Missing /proof/generated.env; sentry-bootstrap must complete before local_workspace starts" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. /proof/generated.env
set +a

exec uvicorn local_workspace_application.host.main:app --host 0.0.0.0 --port 8020