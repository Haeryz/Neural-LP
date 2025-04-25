#!/usr/bin/env python

# Import parser and other components from skibidi
from skibidi import parser, Data, Learner, Option

print("Successfully imported from skibidi.py:")
print(f"- parser: {parser}")
print(f"- Data class: {Data}")
print(f"- Learner class: {Learner}")
print(f"- Option class: {Option}")

# Test parsing arguments
args = parser.parse_args(['--seed', '42', '--datadir', '../datasets/family'])
print(f"Parsed args: seed={args.seed}, datadir={args.datadir}") 