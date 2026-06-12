"""Prompt templates for LLM judge evaluation."""

NAIVE_JUDGE_PROMPT = """\
You are an expert evaluator. Given a question and two responses, determine which response is better.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Evaluate both responses and provide your verdict. Output a JSON object:
{{
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>,
  "reasoning": "<brief explanation>"
}}"""

CALIBRATED_RUBRIC_PROMPT = """\
You are an expert evaluator. Evaluate two responses using the following rubric. Score each criterion from 1-5.

**Rubric:**
1. **Accuracy** (1-5): Is the information factually correct?
   - 1: Major factual errors  2: Some errors  3: Mostly correct  4: Minor issues only  5: Fully accurate
2. **Relevance** (1-5): Does it address the question directly?
   - 1: Off-topic  2: Partially relevant  3: Mostly relevant  4: Directly relevant  5: Perfectly targeted
3. **Completeness** (1-5): Does it cover all aspects of the question?
   - 1: Very incomplete  2: Missing major parts  3: Covers basics  4: Thorough  5: Comprehensive
4. **Clarity** (1-5): Is it well-organized and easy to understand?
   - 1: Confusing  2: Poorly organized  3: Adequate  4: Clear  5: Exceptionally clear
5. **Reasoning** (1-5): Does it show sound logical reasoning?
   - 1: Illogical  2: Weak reasoning  3: Adequate  4: Sound  5: Exceptional reasoning

**IMPORTANT:** Do NOT consider response length or formatting style. Focus only on the criteria above.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Score each response on all 5 criteria, then determine the winner. Output JSON:
{{
  "scores_a": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "scores_b": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "total_a": X,
  "total_b": X,
  "verdict": "A" or "B" or "tie",
  "reasoning": "<brief explanation>"
}}"""

COT_FORCING_PROMPT = """\
You are an expert evaluator. Analyze two responses step by step before making a judgment.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Follow these steps EXACTLY:

Step 1: Identify the key requirements of the question.
Step 2: Evaluate Response A's strengths and weaknesses against those requirements.
Step 3: Evaluate Response B's strengths and weaknesses against those requirements.
Step 4: Compare the two responses directly on substance (ignore length and formatting).
Step 5: Provide your final verdict.

Output JSON:
{{
  "step1_requirements": "<key requirements>",
  "step2_analysis_a": "<Response A analysis>",
  "step3_analysis_b": "<Response B analysis>",
  "step4_comparison": "<direct comparison>",
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>
}}"""

REFERENCE_GUIDED_PROMPT = """\
You are an expert evaluator. Given a question, a reference answer, and two responses, determine which response is better.

Question: {question}

Reference Answer (gold standard): {reference}

Response A: {response_a}

Response B: {response_b}

Evaluate both responses against the reference answer. Output a JSON object:
{{
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>,
  "reasoning": "<brief explanation comparing both to the reference>"
}}"""


def build_prompt(
    template: str,
    question: str,
    response_a: str,
    response_b: str,
    reference: str | None = None,
) -> str:
    kwargs = {
        "question": question,
        "response_a": response_a,
        "response_b": response_b,
    }
    if reference and "{reference}" in template:
        kwargs["reference"] = reference
    return template.format(**kwargs)
