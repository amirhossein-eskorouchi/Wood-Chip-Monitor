# Wood-Chip Monitor User Guide

Wood-Chip Monitor provides a browser-based interface for real-time wood-chip quality evaluation using the local edge-AI inspection system.

The dashboard combines the live annotated camera feed with chip-size analytics, oversize monitoring, moisture assessment, event history, quality rules, audit information, and device status.

## Accessing the Dashboard

After the backend service has been started on the edge device, open the Wood-Chip Monitor interface in a web browser.

Sign in using credentials provided by the system administrator.

On first use, the interface may display a short guided tour introducing the main controls and dashboard components.

## User Roles

Wood-Chip Monitor implements role-based access control so that users receive access appropriate to their responsibilities.

| Role | Primary Access |
|---|---|
| Staff / Operator | View live monitoring and inspection results |
| Quality Engineer | Review/export events and adjust quality rules |
| Manager | Monitor operations, thresholds, and performance information |
| Administrator | Full system and configuration access |

Exact permissions are enforced by the backend application.

## Live Monitoring

The Live Monitoring page is the primary workspace for real-time inspection.

### Live Video

The camera panel displays the current inspection view with detected wood chips annotated by the AI system.

Chips exceeding the configured size threshold can be visually distinguished to provide an immediate oversize warning.

### Real-Time Size Statistics

The dashboard summarizes the current chip population using statistics such as:

- mean chip size,
- minimum and maximum chip size,
- standard deviation,
- number of detected chips.

### Size Distribution

A rolling histogram summarizes the distribution of measured chip sizes.

This view helps operators identify changes in the chip population, including shifts toward larger or more variable material.

### Moisture Assessment

The moisture component summarizes the visual moisture condition using three categories:

- Dry
- Medium
- Wet

The deployed pipeline evaluates selected high-confidence chip regions rather than necessarily processing every detection.

## Monitoring Controls

Authorized users can configure selected runtime parameters through the dashboard.

### Oversize Controls

Available settings include:

- oversize alarm threshold,
- oversize alarm enable/disable,
- reference-object diameter for physical calibration.

### Detection Controls

Detection parameters include:

- confidence threshold,
- non-maximum suppression IoU threshold.

### Moisture Controls

Moisture-inference parameters include:

- enable/disable moisture inference,
- number of Top-K chip crops,
- moisture inference frequency.

### Display

The interface supports light and dark display modes.

Configuration changes require the appropriate user permission.

## Events

The Events page preserves inspection records and alarm conditions over time.

Event information may include:

- timestamp,
- alarm state,
- maximum chip diameter,
- mean chip diameter,
- size variability,
- predicted moisture condition.

Authorized users can review historical events and export data for further analysis.

## Quality Rules

Quality rules define the criteria used to evaluate material during inspection.

Depending on user permissions, configurable rules can include:

- maximum acceptable chip diameter,
- alarm thresholds,
- detection thresholds.

The backend maintains versioned rule configurations so changes can be traced over time.

## Audit Logs

Wood-Chip Monitor maintains an audit trail for important system actions.

Examples include:

- login activity,
- quality-rule modifications,
- data export,
- administrative configuration changes.

Audit records support operational traceability by associating actions with timestamps and authenticated users.

## Device Status

The Device Status page provides information about the health of the inspection system.

Monitoring can include:

- device operational status,
- inference health,
- camera connectivity,
- processing activity.

This allows users to distinguish material-quality conditions from device or inference problems.

## Deployment Configuration

Machine-specific deployment settings should be configured using the variables documented in:

[`../configs/device.env.example`](../configs/device.env.example)

Jetson-specific setup information is available in:

[`jetson-setup.md`](jetson-setup.md)

## Security Notes

Production deployments should:

- replace all bootstrap/default credentials,
- restrict administrator access,
- protect the device network,
- maintain appropriate backups of operational data,
- avoid committing credentials or runtime databases to source control.