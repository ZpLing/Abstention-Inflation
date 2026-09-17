"""Everything that reads from disk.

    config_loader    the experiment YAML, merged with API_Config.yaml
    dataset_loader   dataset/<name>.json, the only reader of the benchmarks
    data_handler     the sample-fetch interface the runners are written against

Keeping these apart from `infra/` marks the boundary the paper cares about:
nothing here decides anything about a prompt, a label or a metric.
"""
