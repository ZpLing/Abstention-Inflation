"""The method itself, shared by every setting.

    prompts          the prompt builders, one per setting, strings frozen
    label_scheme     per-dataset verbs and framing
    evaluator        the deterministic tiered parser
    metrics          Acc, Abs Rate, macro-F1
    result_schema    the summary schema, and `paired_keep_ids` -- the rule that
                     decides which items a paired contrast is scored on
    llm_handler      the async OpenAI-compatible client
    paired_pass      the one pass that collects S1, S2, S3 and the S5 rerun,
                     so the paper's per-item contrasts see the same samples

Reading from disk lives in `loader/`; everything specific to one setting lives
in that setting's folder under `experiments/`.
"""
