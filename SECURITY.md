# Security Policy

## Supported versions

Security and privacy corrections are applied to the latest version on `main`.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| Earlier development snapshots | No |

## Reporting a vulnerability

Do not report vulnerabilities, credentials, or sensitive deployment details
through a public GitHub issue.

Use GitHub private vulnerability reporting when available. Otherwise, contact
the repository owner through the public GitHub profile and request a private
communication channel.

Include the affected component, reproduction conditions, potential impact,
and suggested mitigation. Do not include real credentials, private network
addresses, production databases, or confidential operational data.

## Deployment security

Before deployment:

- replace every example credential;
- store `configs/device.env` outside Git;
- restrict network access;
- use TLS and a reverse proxy when appropriate;
- restrict database and output-directory permissions;
- apply relevant dependency and operating-system updates;
- validate external input; and
- rotate credentials after suspected exposure.

Wood-Chip Monitor is research software and is not presented as a hardened
production security platform.
