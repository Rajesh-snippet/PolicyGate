# PolicyGate

AI output policy guardrail service. Reviews an LLM response before it reaches the user and decides: approve, approve with warning, rewrite, block, or human review.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Endpoints

- `GET /health`
- `POST /guard`

## LLM judge (Groq, free tier)

1. Create a free key at console.groq.com.
2. In `.env`: `JUDGE_PROVIDER=groq` and `GROQ_API_KEY=...` (the model defaults to
   `llama-3.3-70b-versatile`; set `JUDGE_MODEL` to change it).
3. Try it: `python -m evals.run_eval --judge` (live API calls; free tiers have
   rate limits, and the client retries automatically when it is rate limited).

Without a key the service still runs and fails closed: high-risk features go to
human review, and low-risk ones are released with a "not fully checked" warning.