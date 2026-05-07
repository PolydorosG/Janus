JUDGE_PROMPT = """You are matching model-output sentences to provided facts.

You will receive:
1. A list of provided facts, each with a fact_id and text.
2. A list of output units from a model response, each with a unit_id and text.

For each output unit, return the fact_ids of all facts that are explicitly stated or clearly paraphrased in that unit.

Rules:
- Each unit may match zero, one, or multiple facts.
- Return an empty list if no provided fact is stated or clearly paraphrased.
- Do not match generic claims unless they clearly correspond to a provided fact.
- For recommendation, conclusion, or summary units, do not match broad references. Only match a fact if the unit repeats the fact's concrete content, such as a number, threshold, timeframe, named limitation, or clear paraphrase.
- If a sentence names a specific charge, fee, rate adjustment, deductible, drawdown, requirement, or limitation, it may match the corresponding fact even if the exact number is omitted.
- Preserve fact_id strings exactly.
- Return one result for every unit_id.
- Return valid JSON only. No markdown, no explanation.

Output schema:
{
  "unit_matches": [
    {
      "unit_id": 1,
      "matched_fact_ids": ["fact_id_1", "fact_id_2"]
    }
  ]
}

Provided facts:
{facts_json}

Output units:
{units_json}
"""