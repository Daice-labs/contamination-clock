# RP-T05 paraphrase-detection experiment

Regenerates the 14 model-written paraphrases from the deployed laundering arm (texts were not
persisted; the bundled ledger holds their provenance) and tests the standard 13-gram detector
against them. About 14 API calls, under a minute, pennies.

    unzip rpt05-paraphrase-run.zip && cd rpt05-paraphrase-run
    source ../rpt05-run/.venv/bin/activate    # or any venv with: evalplus openai numpy
    ./run.sh sk-ant-YOUR-KEY

Runs on the Anthropic endpoint by default (same model snapshot as the deployed experiment).
Send back `paraphrase_detection_results.json`. Either outcome updates the paper: recall near 0
strengthens the brittleness claim with realistic derivatives; high recall narrows it honestly
(code paraphrases often preserve long verbatim runs).
