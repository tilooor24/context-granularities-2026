import os
import re
import anthropic
from openai import OpenAI


class Model:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 1):
        """
        Unified model interface supporting OpenAI and Claude.
        """
        self.model_name = model_name
        self.temperature = temperature

        # Detect Claude vs OpenAI
        if self._is_claude():
            if self._is_claude():
                raw_key = os.environ.get("ANTHROPIC_API_KEY")
                if not raw_key:
                    raise RuntimeError("Missing ANTHROPIC_API_KEY env var.")
                api_key = raw_key.strip()
                if "\n" in api_key or "\r" in api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY contains newline characters.")
                self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = OpenAI()

    def _is_claude(self):
        return self.model_name.lower().startswith("claude")

    def _claude_generate(self,
                         prompt: str,
                         max_tokens: int = 2048,
                         temperature: float = None):
        """
        Generate text using Claude.
        """
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        message = self.client.messages.create(**kwargs)
        return message.content[0].text

    def generate(self, prompt: str) -> str:
        """
        Generate a single Java line repair.
        Returns raw text (first non-empty line).
        # """
        # _FENCE_RE = re.compile(r"```(?:java)?\s*\n(.*?)```", re.DOTALL)

        # Route to the right provider
        if self._is_claude():
            text = self._claude_generate(prompt, temperature=self.temperature)
        else:
            request = {
                "model": self.model_name,
                "input": prompt,
                "temperature": self.temperature
            }
            response = self.client.responses.create(**request)
            text = response.output_text

        # 1) Prefer extracting from a fenced block (robust to explanations)
        m = re.search(r"```(?:java)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)

        print(f"Text: {text}")
        print(f"m: {m}")

        if m:
            inside = m.group(1)
            lines = [l for l in inside.splitlines() if l.strip() != ""]
            # patch = lines[0].strip() if lines else ""
            print(f"Lines 1: {lines}")
            patch = lines[0] if lines else ""
        else:
            # 2) Fallback: first non-empty line of whole response
            lines = [l for l in text.splitlines() if l.strip() != ""]
            print(f"Lines 2: {lines}")
            patch = lines[0] if lines else ""

        # If still empty, print the raw response EXACTLY (repr shows hidden chars)
        print(f"Patch (repr): {repr(patch)}")
        if not patch:
            print("\n[EMPTY PATCH] raw text repr:", repr(text), flush=True)
            print("[EMPTY PATCH] raw text visible:\n", text, flush=True)

        return patch

        # # Defensive cleanup
        # print(f"Text: {text}", flush=True)
        # # 1) Prefer: content inside the fenced code block
        # m = _FENCE_RE.search(text)
        # if m:
        #     patch = m.group(1).strip()
        #     # If you expect single-line patches, return first non-empty line inside the fence
        #     lines = [l.strip() for l in patch.splitlines() if l.strip()]
        #     print(f"Lines 1: {lines}")
        #     return lines[0] if lines else ""

        # # 2) Fallback: no fence found, do best-effort
        # text = text.strip()
        # lines = [l.strip() for l in text.splitlines() if l.strip()]
        # print(f"Lines 2: {lines}")
        # return lines[0] if lines else ""

        # # text = text.replace("```java", "").replace("```", "").strip()
        # # lines = [l for l in text.splitlines() if l.strip()]
        # # return lines[0] if lines else ""


# from openai import OpenAI


# class Model:
#     def __init__(
#         self,
#         model_name: str = "gpt-4o-mini",
#         temperature: float = 0.0,
#     ):
#         self.client = OpenAI()
#         self.model_name = model_name
#         self.temperature = temperature

#     def generate(self, prompt: str) -> str:
#         """
#         Generate a single Java line repair.
#         Returns raw text (not wrapped in backticks).
#         """
#         response = self.client.responses.create(
#             model=self.model_name,
#             input=prompt,
#             temperature=self.temperature,
#         )

#         text = response.output_text.strip()

#         # Defensive cleanup (models sometimes ignore instructions)
#         text = text.replace("```java", "").replace("```", "").strip()

#         # If multiple lines slip through, keep only the first non-empty line
#         lines = [l for l in text.splitlines() if l.strip()]
#         return lines[0] if lines else ""
