# Reproducibility Notes

This repository has been reorganized for clearer execution boundaries and manuscript-ready reporting.

Recommended practice:

1. Keep raw imaging data outside Git and record only paths or accession identifiers.
2. Record software versions in a frozen environment file before training or analysis.
3. Save all derived tables, figures, and checkpoints to `results/` or an external storage path with date-stamped subdirectories.
4. Document cohort inclusion criteria, segmentation QC, and failed-case handling alongside each generated dataset.
5. Archive the exact commands used for the paper in the Methods supplement or a workflow log.

For Nature-style reporting, include:

- code availability statement
- data availability statement
- software version numbers
- hardware summary for model training
- random seeds where applicable
- explicit notes on restricted-access data and privacy constraints
