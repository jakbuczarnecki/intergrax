# © Artur Czarnecki. All rights reserved.

import json

import httpx
import pytest

from intergrax.integrations._shared.p5.clients import KubernetesCloudPlatform
from intergrax.integrations.providers.cloud_platform.kubernetes.rest_client import (
    KubernetesDeploymentScaleClient,
)
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind, ScalingTarget
from intergrax.runtime.capacity.provisioner import ScalingProvisioner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_kubernetes_rest_client_get_and_scale() -> None:
    replicas = {"value": 2}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"status": {"replicas": replicas["value"]}},
            )
        if request.method == "PATCH":
            body = json.loads(request.content.decode())
            replicas["value"] = int(body["spec"]["replicas"])
            return httpx.Response(200, json={"status": {"replicas": replicas["value"]}})
        return httpx.Response(404)

    client = KubernetesDeploymentScaleClient(
        base_url="https://k8s.example",
        namespace="lab",
        transport=httpx.MockTransport(handler),
    )
    platform = KubernetesCloudPlatform(client, namespace="lab")
    assert platform.get_replicas(deployment="nexus-host") == 2
    platform.scale_workload(deployment="nexus-host", replicas=4)
    assert platform.get_replicas(deployment="nexus-host") == 4


def test_provisioner_scales_via_kubernetes_rest_client() -> None:
    replicas = {"value": 1}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": {"replicas": replicas["value"]}})
        if request.method == "PATCH":
            replicas["value"] = int(json.loads(request.content.decode())["spec"]["replicas"])
            return httpx.Response(200, json={"status": {"replicas": replicas["value"]}})
        return httpx.Response(404)

    client = KubernetesDeploymentScaleClient(
        base_url="https://k8s.example",
        namespace="lab",
        transport=httpx.MockTransport(handler),
    )
    platform = KubernetesCloudPlatform(client, namespace="lab")
    provisioner = ScalingProvisioner(kubernetes=platform)
    ok = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            target=ScalingTarget.NEXUS_HOST,
            delta=2,
        ),
        deployment="nexus-host",
    )
    assert ok is True
    assert replicas["value"] == 3
