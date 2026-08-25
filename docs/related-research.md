# Related Research

Wood-Chip Monitor is part of a broader research program on vision-based measurement and quality assessment for wood-chip and biomass manufacturing.

## Edge-AI System

**An Edge Artificial Intelligence System for Wood Chip Quality Evaluation**

Amirhossein Eskorouchi, Prashant Bhattarai, Abdur Rahman, Mohammad Marufuzzaman, Jason T. Street, and Haifeng Wang.

Proceedings of the IISE Annual Conference & Expo 2026.

This work describes the integrated edge-AI prototype combining wood-chip geometry assessment, size-distribution monitoring, and vision-based moisture evaluation.

## UOT-DETR

**UOT-DETR** develops the distribution-aware detection methodology underlying the broader wood-chip visual-measurement research.

Repository:

`amirhossein-eskorouchi/UOT-DETR`

The method focuses on dense small-object detection and reliable recovery of wood-chip size distributions.

## MoistNet

**MoistNet: Machine vision-based deep learning models for wood chip moisture content measurement**

Abdur Rahman, Jason Street, James Wooten, Mohammad Marufuzzaman, Veera G. Gude, Randy Buchanan, and Haifeng Wang.

*Expert Systems with Applications*, Volume 259, Article 125363, 2025.

DOI:

`10.1016/j.eswa.2024.125363`

Wood-Chip Monitor uses the lightweight MoistNetLite architecture as the moisture-assessment component of the deployed inspection pipeline.

## WoodChip-Detection Dataset

**WoodChip-Detection: A Public Dataset for Dense Wood Chip Detection and Instance Segmentation**

Amirhossein Eskorouchi, Jason Street, James Wooten, Mohammad Marufuzzaman, and Haifeng Wang.

Zenodo, 2026.

DOI:

`10.5281/zenodo.18392693`

The dataset supports research on dense wood-chip detection, instance-level analysis, and visual measurement.

## Research Scope

The resources serve complementary roles:

    WoodChip-Detection Dataset
              |
              v
    Detection and Visual-Measurement Research
              |
              +---- UOT-DETR
              |
              v
    Wood-Chip Monitor
              |
              +---- Physical size estimation
              +---- Size-distribution monitoring
              +---- Oversize detection
              +---- MoistNetLite moisture assessment
              |
              v
    Industrial Edge-AI Decision Support

Resources:

- Wood-Chip Monitor:
  <https://github.com/amirhossein-eskorouchi/Wood-Chip-Monitor>
- UOT-DETR:
  <https://github.com/amirhossein-eskorouchi/UOT-DETR>
- MoistNet:
  <https://doi.org/10.1016/j.eswa.2024.125363>
- WoodChip-Detection dataset:
  <https://doi.org/10.5281/zenodo.18392693>

See [`../CITATION.cff`](../CITATION.cff),
[`../CITATION.bib`](../CITATION.bib), and
[`CITATION.md`](CITATION.md).
