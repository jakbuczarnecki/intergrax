# Signoff probe agent — architecture

## Purpose

Gate and lab **sign-off probe** for Appendix A human-approval flows (`signoff.probe` capability).

## Capabilities

- `signoff.probe`

## Runtime

- `HarnessReferenceAgent` + UAEP pipeline
- `LabHarnessContext` injected by Tier-3 host builders (no `applications` imports)

## Registration

- `applications/lab_application/manifest.py` when enabled in lab settings
