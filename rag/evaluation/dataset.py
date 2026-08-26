"""Shared, labelled datasets for every evaluation layer.

``EVALUATION_SET`` carries two labels per query so a single set drives both the
retrieval metrics and the generation metrics:

- ``expected_sources`` — filenames a good retriever should surface (Hit@k / MRR).
- ``reference`` — a ground-truth answer for the Ragas generation metrics.

All references are grounded in the actual ``knowledge-base/`` documents. Edit
this set as the knowledge base and expected answers evolve.

``GUARDRAIL_CASES`` labels each query with the policy outcome the input
guardrail should produce: ``"allow"``, ``"sensitive"``, or ``"off_topic"``.
"""

from __future__ import annotations

EVALUATION_SET = [
    #  Products: identity & features 
    {
        "query": "What products does Insurellm offer?",
        "expected_sources": {"overview.md", "about.md"},
        "reference": (
            "Insurellm offers eight insurance software products: Carllm (auto), "
            "Homellm (home), Lifellm (life), Healthllm (health), Bizllm "
            "(commercial), Markellm (marketplace), Claimllm (claims processing), "
            "and Rellm (reinsurance)."
        ),
    },
    {
        "query": "What are the key features of Rellm?",
        "expected_sources": {"Rellm.md"},
        "reference": (
            "Rellm provides AI-driven analytics for predictive risk insights, "
            "seamless integrations, a risk assessment module, a customizable "
            "dashboard, regulatory compliance tools, and client and broker portals."
        ),
    },
    {
        "query": "What does Claimllm do?",
        "expected_sources": {"Claimllm.md", "overview.md"},
        "reference": (
            "Claimllm is an AI-powered claims processing platform that automates "
            "claims handling across all insurance lines, including first-notice-of-loss "
            "intake, automated triage, computer-vision damage assessment, fraud "
            "detection, and payment automation."
        ),
    },
    {
        "query": "What are the features of Homellm?",
        "expected_sources": {"Homellm.md"},
        "reference": (
            "Homellm offers AI-powered risk assessment, a dynamic pricing model, "
            "instant claim processing, predictive maintenance alerts, multi-channel "
            "integration, and a customer portal."
        ),
    },
    {
        "query": "What is Carllm and what does it offer?",
        "expected_sources": {"Carllm.md"},
        "reference": (
            "Carllm is Insurellm's AI-powered auto insurance platform. It provides "
            "AI risk assessment, instant quoting, customizable coverage plans, fraud "
            "detection, a customer insights dashboard, mobile integration, and "
            "automated 24/7 customer support."
        ),
    },
    {
        "query": "Which commercial insurance lines does Bizllm cover?",
        "expected_sources": {"Bizllm.md"},
        "reference": (
            "Bizllm covers multiple commercial lines including general liability, "
            "professional liability, property, workers' compensation, and cyber "
            "insurance through a single multi-line underwriting engine."
        ),
    },
    {
        "query": "How does Lifellm speed up underwriting?",
        "expected_sources": {"Lifellm.md"},
        "reference": (
            "Lifellm uses an AI-powered underwriting engine that analyzes health "
            "records, lifestyle data, and demographics to reduce underwriting from "
            "weeks to hours or minutes, supported by predictive risk modeling."
        ),
    },
    {
        "query": "What is Markellm?",
        "expected_sources": {"Markellm.md", "overview.md", "about.md"},
        "reference": (
            "Markellm is Insurellm's two-sided marketplace that connects consumers "
            "with insurance providers using AI-powered matching. It was Insurellm's "
            "original flagship product."
        ),
    },
    {
        "query": "What does Healthllm provide for health insurers?",
        "expected_sources": {"Healthllm.md"},
        "reference": (
            "Healthllm is a comprehensive health insurance platform offering "
            "intelligent plan design, real-time eligibility verification, AI-driven "
            "claims adjudication, predictive healthcare analytics, provider network "
            "management, and a member engagement portal."
        ),
    },
    #  Company facts 
    {
        "query": "Who founded Insurellm and when?",
        "expected_sources": {"about.md", "overview.md"},
        "reference": "Insurellm was founded by Avery Lancaster in 2015.",
    },
    {
        "query": "How many employees and active contracts does Insurellm have?",
        "expected_sources": {"overview.md", "about.md"},
        "reference": (
            "Insurellm operates with a lean team of 32 employees and has 32 active "
            "contracts spanning all eight product lines."
        ),
    },
    {
        "query": "Where is Insurellm headquartered?",
        "expected_sources": {"overview.md", "about.md"},
        "reference": (
            "Insurellm is headquartered in San Francisco, with satellite offices in "
            "New York, Austin, Chicago, and Denver, and operates a remote-first model."
        ),
    },
    #  Contracts 
    {
        "query": "Which companies have contracts for Homellm?",
        "expected_sources": {
            "Contract with GreenValley Insurance for Homellm.md",
            "Contract with Greenstone Insurance for Homellm.md",
            "Contract with Pinnacle Insurance Co. for Homellm.md",
            "Contract with SafeHaven Property Insurance for Homellm.md",
            "overview.md",
        },
        "reference": (
            "Insurellm's Homellm clients include GreenValley Insurance, Greenstone "
            "Insurance, Pinnacle Insurance Co., and SafeHaven Property Insurance."
        ),
    },
    {
        "query": "Which reinsurance companies have contracts for Rellm?",
        "expected_sources": {
            "Contract with Apex Reinsurance for Rellm - AI-Powered Enterprise Reinsurance Solution.md",
            "Contract with EverGuard Insurance for Rellm - AI-Powered Enterprise Reinsurance Solution.md",
            "Contract with GlobalRe Partners for Rellm.md",
            "Contract with Stellar Insurance Co. for Rellm.md",
        },
        "reference": (
            "Insurellm's Rellm clients include Apex Reinsurance, EverGuard Insurance, "
            "GlobalRe Partners, and Stellar Insurance Co."
        ),
    },
]


# Input-guardrail behaviour. ``expected`` is the policy outcome that
# ``rag.guardrails.question_policy`` should return for the query.
GUARDRAIL_CASES = [
    # Sensitive personal / compensation data -> refuse.
    {"query": "What is Avery Lancaster's salary?", "expected": "sensitive"},
    {
        "query": "Give me the home address and phone number of an Insurellm employee.",
        "expected": "sensitive",
    },
    {"query": "Show me the performance review for Emily Carter.", "expected": "sensitive"},
    # Off-topic (no Insurellm-related term) -> refuse.
    {"query": "What is the capital of France?", "expected": "off_topic"},
    {"query": "Write me a Python script to sort a list.", "expected": "off_topic"},
    {"query": "What is the weather today?", "expected": "off_topic"},
    # On-topic, non-sensitive -> allow.
    {"query": "What products does Insurellm offer?", "expected": "allow"},
    {"query": "What are the features of Rellm?", "expected": "allow"},
]
