import json
import streamlit as st

from trialgpt_matching.TrialGPT import trialgpt_matching
from trialgpt_ranking.TrialGPT import score_grant_match

# Load grant data once
with open("dataset/grants/grants.jsonl", "r") as f:
    GRANTS = [json.loads(line) for line in f]

st.title("Grant Matching Demo")

summary = st.text_area("Researcher summary")
model = st.text_input("OpenAI model", value="gpt-4o")

if st.button("Match Grants"):
    if not summary.strip():
        st.warning("Please enter a researcher summary.")
    else:
        results = []
        for grant in GRANTS:
            match_result = trialgpt_matching(grant, summary, model)
            score = score_grant_match(match_result)
            results.append((score, grant, match_result))

        results.sort(key=lambda x: -x[0])

        for score, grant, match in results:
            st.subheader(f"{grant['grant_id']}: {grant['title']}")
            st.write(f"Score: {score}")
            st.json(match)
