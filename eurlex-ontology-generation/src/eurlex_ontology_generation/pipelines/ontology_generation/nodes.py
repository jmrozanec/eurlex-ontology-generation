import json
import re
import pandas as pd
from ollama import chat


# Generate a candidate ontology for each chunk using a local LLM (Ollama)
def generate_partial_ontologies(
    chunks: pd.DataFrame,
    max_chunks: int,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
) -> pd.DataFrame:

    # Store one ontology candidate for each processed chunk
    results = []

    # Process only the first max_chunks
    for _, row in chunks.head(max_chunks).iterrows():

        # Build the user prompt by inserting the chunk text
        user_prompt = user_prompt_template.format(
            text=row["chunk_text"]
        )

        # Query the local LLM through Ollama
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": temperature,
            },
            # If json_mode is True, forces the model to output a structured JSON string
            format="json" if json_mode else None,
        )

        # Extract the textual response returned by the model
        ontology_text = response["message"]["content"]

        # Parse and validate the JSON returned by the LLM
        ontology_json = extract_json(ontology_text)        

        # Save both the generated ontology and useful metadata
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "CELEX": row["CELEX"],
                # Serialize the parsed JSON back to a string, preserving non-ASCII characters
                "ontology": json.dumps(
                    ontology_json,
                    ensure_ascii=False
                ),
            }
        )

    # Return one ontology candidate for each processed chunk
    return pd.DataFrame(results)


# Helper to safely extract JSON from LLM output
def extract_json(text: str) -> dict:

    try:
        return json.loads(text)     # Direct parsing

    except json.JSONDecodeError:
        # Regex fallback: find anything between the first '{' and last '}'
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Return error payload if all parsing attempts fail
    return {
        "error": "Invalid JSON generated",
        "raw_output": text
    }