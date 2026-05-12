# `data/splits/`

Operational state directory. Holds the train/dev/holdout assignment
file (`split_v1.json`) regenerable from seed `0xBADA55` in notebook 02.

The split file is gitignored. Public release does not include it
because it can be reproduced from the seed and the RoSafetyBench
prompts (Paper 2 Zenodo deposit).

This file exists only so the directory survives Drive sync (which
silently drops empty `.gitkeep` files).
