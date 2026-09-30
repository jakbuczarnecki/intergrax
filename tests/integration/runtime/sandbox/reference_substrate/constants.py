# © Artur Czarnecki. All rights reserved.

"""Qualification-only reference sandbox physical egress substrate."""

REFERENCE_PROVIDER_ID = "reference-substrate-qualification"

ALLOWED_HOSTNAME = "allowed.test"
DENIED_HOSTNAME = "denied.test"
ALLOWED_PORT = 18080
DENIED_PORT = 18081

ALLOWED_ADDR = "10.200.42.1"
DENIED_ADDR = "10.200.42.3"
SANDBOX_ADDR = "10.200.42.2"
VETH_HOST_ADDR = "10.200.42.1"
# Host listener bind (all interfaces); sandbox egress target remains ALLOWED_ADDR:ALLOWED_PORT.
ALLOWED_LISTEN_BIND = ""
SUBNET = "10.200.42.0/24"

REDIRECT_PATH = "/redirect-to-denied"
