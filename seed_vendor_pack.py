from pathlib import Path

DOCS = {
    "01_vendor_overview.md": """
# Synthetic Demo Artifact: Vendor Overview

## Company profile
Northstar Analytics provides a multi-tenant reporting platform for enterprise customers.

## Service ownership
The platform team operates production services. The security team owns incident coordination and customer security communications.

## Hosting scope
Production workloads run in two regions. Customer data remains in the contracted primary region unless disaster recovery is activated.
""",
    "02_security_policy.md": """
# Synthetic Demo Artifact: Security Policy

## Encryption
Customer data is encrypted at rest with AES-256 and protected in transit with TLS 1.3.

## Incident notification
Customers are notified within 24 hours after Northstar confirms a critical security incident.

## Security log retention
Authentication and administrative audit logs are retained for 90 days.
""",
    "03_service_level_agreement.md": """
# Synthetic Demo Artifact: Service Level Agreement

## Availability
The monthly production availability target is 99.9 percent.

## Priority one response
Northstar acknowledges a Priority 1 production incident within 2 hours.

## Outage communication
Affected customers receive an initial service notification within 4 hours after a production outage is validated.
""",
    "04_incident_response_plan.md": """
# Synthetic Demo Artifact: Incident Response Plan

## Customer notification
The incident commander notifies affected customers within 72 hours after a suspected security incident enters formal investigation.

## Priority one response
The on-call security lead acknowledges a Priority 1 security incident within 6 hours.

## Evidence handling
Investigation evidence is preserved in a restricted case workspace until legal and security teams approve disposal.
""",
    "05_data_processing_addendum.md": """
# Synthetic Demo Artifact: Data Processing Addendum

## Subprocessor notice
Customers receive 30 days notice before a new material subprocessor handles customer personal data.

## Customer data deletion
Following contract termination, active customer data is deleted within 45 days unless law requires continued retention.

## Data requests
Northstar assists with verified data-subject requests within the time stated in the customer contract.
""",

    "06_data_retention_standard.md": """
# Synthetic Demo Artifact: Data Retention Standard

## Operational logs
Production operational and security logs are retained for 365 days to support investigations and capacity analysis.

## Backup deletion
Expired customer backups are removed within 90 days after contract termination.

## Exceptions
Legal holds override normal deletion schedules until the hold is released.
""",
    "07_business_continuity_plan.md": """
# Synthetic Demo Artifact: Business Continuity Plan

## Recovery objectives
The production recovery time objective is 8 hours. The recovery point objective is 24 hours.

## Exercises
Northstar performs one tabletop continuity exercise each calendar year.

## Customer updates
During a declared disaster, customers receive status updates every 6 hours.
""",
    "08_access_control_policy.md": """
# Synthetic Demo Artifact: Access Control Policy

## Workforce access
Production access requires single sign-on, multi-factor authentication, and manager approval.

## Privileged reviews
Privileged access is reviewed every quarter by Security and Engineering leadership.

## Termination
Workforce access is disabled within 4 hours after Human Resources records a termination.
""",
}


def seed_documents(target: str = "vendor_docs") -> int:
    folder = Path(target)
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in DOCS.items():
        (folder / name).write_text(content.strip() + "\n", encoding="utf-8")
    return len(DOCS)


if __name__ == "__main__":
    count = seed_documents()
    print(f"Created {count} synthetic vendor documents in vendor_docs")
