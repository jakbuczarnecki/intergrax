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

export LOCAL_WORKSPACE_REFERENCE_PRODUCTION=1

exec python -m local_workspace_application.host.main
