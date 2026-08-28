# Durable TRAIN findings

## Evaluation preprocessing can manufacture evaluator errors

Sentence-v1 detached list markers from the content they introduced. Span overlap could label both
the marker and following content unsupported, creating an artificial standalone unit and apparent
local false-negative judge errors. Sentence-v2 removed this demonstrated preprocessing defect
before TEST; it was not introduced to improve a score.

## Judge/reference disagreement is not automatically judge error

After repair, human reviewers found all seven current TRAIN judge-only detections defensible from
the supplied evidence even though official RAGTruth did not annotate them. This supports careful
error analysis, not a claim that the judge is perfect or the benchmark is wrong.

## Human review should remain explanatory

Human adjudication diagnoses segmentation, mapping, rubric, judge, and benchmark ambiguity while
official benchmark labels remain intact for primary metrics.

## Scope

The six-record canary validated methodology and contracts. It is too small to establish general
model behavior, and its results must not be presented as final benchmark performance.
