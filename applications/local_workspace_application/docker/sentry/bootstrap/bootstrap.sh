#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Bootstrap local Sentry proof org/project and write LKW DSN env for Compose.
# Migrations run in sentry-upgrade before sentry-web starts; this script only
# seeds proof account/org/project and writes generated.env (local proof state).

set -euo pipefail

MARKER="/proof/.bootstrapped"
GENERATED_ENV="/proof/generated.env"
GENERATED_ENV_TMP="/proof/generated.env.tmp"
PROOF_ADMIN_EMAIL="${SENTRY_PROOF_ADMIN_EMAIL:-admin@intergrax.local}"
PROOF_ADMIN_PASSWORD="${SENTRY_PROOF_ADMIN_PASSWORD:-proof-local-only}"
ORG_SLUG="${SENTRY_PROOF_ORG_SLUG:-intergrax-local}"
PROJECT_SLUG="${SENTRY_PROOF_PROJECT_SLUG:-lkw-proof}"
RELAY_HOST="${SENTRY_PROOF_RELAY_HOST:-sentry-relay}"
RELAY_PORT="${SENTRY_PROOF_RELAY_PORT:-3000}"

if [[ -f "${MARKER}" && -f "${GENERATED_ENV}" ]]; then
  echo "sentry-bootstrap: already initialized"
  exit 0
fi

echo "sentry-bootstrap: waiting for sentry-web readiness"
for _ in $(seq 1 120); do
  if sentry django shell -c "import django; django.setup(); print('ok')" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

echo "sentry-bootstrap: ensuring proof superuser"
if ! sentry django shell -c "from sentry.models import User; import sys; sys.exit(0 if User.objects.filter(email='${PROOF_ADMIN_EMAIL}').exists() else 1)"; then
  sentry createuser \
    --email "${PROOF_ADMIN_EMAIL}" \
    --password "${PROOF_ADMIN_PASSWORD}" \
    --superuser \
    --no-input
fi

echo "sentry-bootstrap: ensuring proof org/project/key"
sentry django shell -c "
import os
from sentry.models import Organization, OrganizationMember, Project, ProjectKey, Team, User

email = '${PROOF_ADMIN_EMAIL}'
org_slug = '${ORG_SLUG}'
project_slug = '${PROJECT_SLUG}'
relay_host = '${RELAY_HOST}'
relay_port = '${RELAY_PORT}'
tmp_path = '${GENERATED_ENV_TMP}'
final_path = '${GENERATED_ENV}'

user = User.objects.get(email=email)
org, created = Organization.objects.get_or_create(slug=org_slug, defaults={'name': 'Intergrax Local Proof'})
if created:
    org.create_default_team()
team = Team.objects.get(organization=org, slug=org_slug)
OrganizationMember.objects.get_or_create(
    organization=org,
    user=user,
    defaults={'role': 'owner'},
)
project, _ = Project.objects.get_or_create(
    organization=org,
    slug=project_slug,
    defaults={'name': 'LKW Sentry Proof'},
)
if not project.teams.exists():
    project.add_team(team)
key = ProjectKey.objects.filter(project=project).order_by('id').first()
if key is None:
    key = ProjectKey.objects.create(project=project, label='LKW Proof')
dsn = f'http://{key.public_key}@{relay_host}:{relay_port}/{project.id}'
with open(tmp_path, 'w', encoding='utf-8') as handle:
    handle.write(f'LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN={dsn}\n')
os.replace(tmp_path, final_path)
print('sentry-bootstrap: wrote local proof DSN env')
"

touch "${MARKER}"
echo "sentry-bootstrap: complete"
