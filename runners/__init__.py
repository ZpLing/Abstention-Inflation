"""The runners `main.py --config` dispatches.

What lives here is the machinery a YAML can point at, not one file per setting.
`ab_runner` alone collects S1, S2, S3 and the S5 rerun, because the paper
compares them per item and scoring them from separate runs would compare
different samples of the dataset.

The settings that are not driven from a YAML -- S4, S7, S8, S9's re-draws,
S10's temperature and local sweeps, S11 -- own their entry point under
`experiments/Cx_*/Sy_*/` and take their arguments on the command line.
"""
