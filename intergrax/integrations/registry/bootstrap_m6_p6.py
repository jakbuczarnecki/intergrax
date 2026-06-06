# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Phase M.6 P6 integration slugs (32 harness providers)."""

from __future__ import annotations


def register_m6_p6_integrations(*, override: bool = False) -> None:
    from intergrax.integrations.providers.security_scanner.trivy.register import register_trivy_integration
    from intergrax.integrations.providers.security_scanner.snyk.register import register_snyk_integration
    from intergrax.integrations.providers.security_scanner.semgrep.register import register_semgrep_integration
    from intergrax.integrations.providers.secrets_store.infisical.register import register_infisical_integration
    from intergrax.integrations.providers.sandbox_host.e2b.register import register_e2b_integration
    from intergrax.integrations.providers.sandbox_host.modal.register import register_modal_integration
    from intergrax.integrations.providers.sandbox_host.daytona.register import register_daytona_integration
    from intergrax.integrations.providers.identity_provider.auth0.register import register_auth0_integration
    from intergrax.integrations.providers.identity_provider.keycloak.register import register_keycloak_integration
    from intergrax.integrations.providers.identity_provider.workos.register import register_workos_integration
    from intergrax.integrations.providers.ci_cd.argocd.register import register_argocd_integration
    from intergrax.integrations.providers.ci_cd.buildkite.register import register_buildkite_integration
    from intergrax.integrations.providers.ci_cd.jenkins.register import register_jenkins_integration
    from intergrax.integrations.providers.speech_provider.elevenlabs.register import register_elevenlabs_integration
    from intergrax.integrations.providers.speech_provider.deepgram.register import register_deepgram_integration
    from intergrax.integrations.providers.observability_backend.newrelic.register import register_newrelic_integration
    from intergrax.integrations.providers.observability_backend.splunk.register import register_splunk_integration
    from intergrax.integrations.providers.issue_tracker.zendesk.register import register_zendesk_integration
    from intergrax.integrations.providers.feature_flag.statsig.register import register_statsig_integration
    from intergrax.integrations.providers.workflow_orchestrator.prefect.register import register_prefect_integration
    from intergrax.integrations.providers.workflow_orchestrator.airflow.register import register_airflow_integration
    from intergrax.integrations.providers.vector_store.typesense.register import register_typesense_integration
    from intergrax.integrations.providers.relational_store.neon.register import register_neon_integration
    from intergrax.integrations.providers.message_bus.pulsar.register import register_pulsar_integration
    from intergrax.integrations.providers.search_provider.algolia.register import register_algolia_integration
    from intergrax.integrations.providers.message_bus.confluent.register import register_confluent_integration
    from intergrax.integrations.providers.object_storage.backblaze_b2.register import register_backblaze_b2_integration
    from intergrax.integrations.providers.vision_serving.triton.register import register_triton_integration
    from intergrax.integrations.providers.ml_inference_host.replicate.register import register_replicate_integration
    from intergrax.integrations.providers.billing_meter.stripe.register import register_stripe_integration
    from intergrax.integrations.providers.crm.salesforce.register import register_salesforce_integration
    from intergrax.integrations.providers.crm.hubspot.register import register_hubspot_integration

    register_trivy_integration(override=override)
    register_snyk_integration(override=override)
    register_semgrep_integration(override=override)
    register_infisical_integration(override=override)
    register_e2b_integration(override=override)
    register_modal_integration(override=override)
    register_daytona_integration(override=override)
    register_auth0_integration(override=override)
    register_keycloak_integration(override=override)
    register_workos_integration(override=override)
    register_argocd_integration(override=override)
    register_buildkite_integration(override=override)
    register_jenkins_integration(override=override)
    register_elevenlabs_integration(override=override)
    register_deepgram_integration(override=override)
    register_newrelic_integration(override=override)
    register_splunk_integration(override=override)
    register_zendesk_integration(override=override)
    register_statsig_integration(override=override)
    register_prefect_integration(override=override)
    register_airflow_integration(override=override)
    register_typesense_integration(override=override)
    register_neon_integration(override=override)
    register_pulsar_integration(override=override)
    register_algolia_integration(override=override)
    register_confluent_integration(override=override)
    register_backblaze_b2_integration(override=override)
    register_triton_integration(override=override)
    register_replicate_integration(override=override)
    register_stripe_integration(override=override)
    register_salesforce_integration(override=override)
    register_hubspot_integration(override=override)
