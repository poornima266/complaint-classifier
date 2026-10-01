import os
import anthropic
import pandas as pd

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

df = pd.read_csv("complaints.csv")
sample = df.sample(30, random_state=42)

correct = 0
total = 0

for _, row in sample.iterrows():
    narrative = row["narrative"]
    true_label = row["product"]

    prompt = f"""Classify this consumer complaint into exactly one of these categories:
credit_reporting, debt_collection, mortgages_and_loans, credit_card, retail_banking

Complaint: {narrative}

Respond with only the category name, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )

    predicted = response.content[0].text.strip()
    match = predicted == true_label

    total += 1
    if match:
        correct += 1

    print(f"True label:      {true_label}")
    print(f"Claude predicted: {predicted}")
    print(f"Match: {match}")
    print("---")

accuracy = correct / total * 100
print(f"\nFinal accuracy: {correct}/{total} = {accuracy:.1f}%")



