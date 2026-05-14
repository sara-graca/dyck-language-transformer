# Transformers and Dyck Languages

A small Transformer encoder trained to detect and correct errors in Dyck language sequences, with analysis of out-of-distribution generalisation, attention patterns, and probing classifiers.

## Project Structure

```
src/
    data.py                     # Data generation (clean sequences + 4 error types)
    model.py                    # Shared Transformer encoder with detection and correction heads
notebooks/
    training.ipynb              # Detection training
    cor_training.ipynb          # Correction training
    finetuning.ipynb            # Fine-tuning on n=5 sequences
    curriculum_learning.ipynb   # Curriculum learning experiment
    evaluation.ipynb            # In-distribution and OOD evaluation, PDA comparison
    cor_evaluation.ipynb        # Correction evaluation by error type
    attention_analysis.ipynb    # Attention weights, bracket matching, rollout
    probing.ipynb               # Probing classifiers for nesting depth
results/
    models/                     # Saved model checkpoints
    figures/                    # Generated plots
```


## Tasks

- **Detection**: binary classification of valid vs corrupted Dyck sequences
- **Correction**: token-level edit label prediction (OK, DELETE, REPLACE, INSERT)

## Error Types

- E1: missing closer
- E2: spurious opener
- E3: type mismatch
- E4: premature close

## Key Results

- Near-perfect in-distribution detection accuracy (100%)
- OOD accuracy degrades from 73% at n=5 to 58% at n=7 for the standard model
- Curriculum learning recovers near-perfect OOD accuracy without any OOD training data
- No attention head implements bracket matching
- Nesting depth is linearly encoded in the [CLS] representation
