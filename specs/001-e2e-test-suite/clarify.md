# Clarification Questions: End-to-End Test Suite

**Status**: Complete — all answers recorded in spec.md

| # | Question | Answer |
|---|----------|--------|
| Q1 | Max wall-clock time for full pipeline in CI? | **A** — 5 minutes |
| Q2 | How to isolate Firestore test data? | Same GCP project, named database `rio-test`, same collection names |
| Q3 | Minimum DeepEval GEval passing score? | **B** — 0.7 |
| Q4 | Test result report format? | **A** — Structured JSON file per run |
| Q5 | Is retry count always exactly 4? | **B** — 4 is default, configurable per workflow |
| Q6 | Where to store test images and audio? | Separate folders: `test/images/` and `test/audio/`, cleaned up after each run |
