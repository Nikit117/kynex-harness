"""ExtractBench benchmark adapter (thin translation only).

This package contains NO extraction intelligence: no answer rules, no
document-name rules, no field hacks, no GT-derived aliases, no retrieval
tricks. It translates ExtractBench (document + JSON Schema) into generic
HARNESS ExtractionContracts, runs the REAL production path
(DeepAgentInvestigator + GPT-5.4 + native parsers + Vision + admission +
evidence + verification + recovery + completeness), and translates claims
back into the official prediction format.

Ground truth is evaluator-only and never enters this package.
"""

from .adapter import ExtractBenchAdapter

__all__ = ["ExtractBenchAdapter"]
