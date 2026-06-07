# Incident Postmortem: INC-2026-05

Severity: SEV2

## Customer Impact

The customer portal showed stale entitlement data for a subset of enterprise accounts.

## Root Cause

A stale edge cache served outdated entitlement data because the billing sync worker did not emit a cache invalidation event after plan changes.

## Remediation

The remediation owner is Platform Reliability.

Corrective actions:

- Add idempotent invalidation to the billing sync worker.
- Add dashboard alerting for stale entitlement reads.
- Add a replay test before release train 2026.06.

## Retrieval Notes

This fixture tests hybrid retrieval over incident IDs, component names, and remediation ownership.
