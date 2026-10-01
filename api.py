import os
import anthropic
from fastapi import FastAPI

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@app.get("/classify")
def classify(text: str):
    prompt = f"""Classify this consumer complaint into exactly one of these categories:
credit_reporting, debt_collection, mortgages_and_loans, credit_card, retail_banking

Complaint: {text}

Respond with only the category name, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )

    predicted = response.content[0].text.strip()
    return {"category": predicted}
