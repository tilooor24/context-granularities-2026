from openai import OpenAI


class Model:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """
        Generate a single Java line repair.
        Returns raw text (not wrapped in backticks).
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
        )

        text = response.output_text.strip()

        # Defensive cleanup (models sometimes ignore instructions)
        text = text.replace("```java", "").replace("```", "").strip()

        # If multiple lines slip through, keep only the first non-empty line
        lines = [l for l in text.splitlines() if l.strip()]
        return lines[0] if lines else ""
