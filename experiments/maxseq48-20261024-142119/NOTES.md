# Max-seqs 48 experiment

Before: {"requests": 72, "ok": 72, "elapsed_s": 5.181373765997705, "p95_ttft_s": 0.44903975899796933, "p95_tpot_s": 0.018798787611754676}

After: {"requests": 72, "ok": 72, "elapsed_s": 6.717006384998967, "p95_ttft_s": 4.059422944999824, "p95_tpot_s": 0.01475008763138926}

TPOT improved but TTFT regressed sharply. Restored original container as vllm. Candidate retained stopped as vllm-test-maxseq48-20261024-142119. No grader submission made. Synthetic short prompts do not establish Final score or accuracy.
