# Results

Experimental results organized by test type.

## Structure

```
results/
├── 7b_experiments/     # Results from run_experiments_7b.py
├── 9b_experiments/     # Results from run_experiments_9b.py
├── stack_attacks/      # Results from STACK attack scripts
└── README.md
```

## 7B Experiments

Classifier tests and simple attacks using 7B models.

Files:
- results_7b_classifier_test.json
- results_7b_perturbation_attack.json
- results_7b_suffix_attack.json
- results_7b_summary.json

## 9B Experiments

Classifier tests and attacks using 9B models (better performance).

Files:
- results_9b_classifier_test.json
- results_9b_perturbation_attack.json
- results_9b_jailbreak_attack.json
- results_9b_summary.json
- run_9b_corrected.log

## STACK Attacks

STACK attack results with different iteration counts.

Files:
- results_stack_gpu_optimized_20260606_2054.json (200 iterations, 53.3% ASR)
- stack_attack_gpu_optimized_20260606_2054.log
- gpu_run_output_20260606_2054.txt

## Key Result

GPU-Optimized STACK Attack (200 iterations, 19.2 hours):
- ASR: 53.3% (8/15 queries bypassed)
- Input blocks: 7/15 (46.7%)
- Output blocks: 0/15 (0%)
- Paper target: 71.0%
- Gap: -17.7 percentage points

See parent README.md for detailed analysis.
