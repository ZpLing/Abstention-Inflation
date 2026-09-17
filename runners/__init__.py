"""One runner per experimental setting.

Everything here drives a setting end to end; everything in the parent package
is shared infrastructure it calls into (prompts, parser, metrics, loaders).
Keeping the two apart is what makes ``infra`` readable: a file here answers
"how was S6 collected", a file above answers "how is any answer parsed".
"""
