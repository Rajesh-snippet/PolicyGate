import json
import re


def verdict(policy: str, evidence: str, severity: str = "medium", confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "violations": [
                {
                    "policy": policy,
                    "severity": severity,
                    "evidence": evidence,
                    "reason": "test reason",
                    "confidence": confidence,
                }
            ]
        }
    )


EMPTY = json.dumps({"violations": []})


class FakeClient:
    """Stands in for an LLM. `responses` maps policy name -> JSON string,
    Exception, or callable(user_prompt) -> str. Unlisted policies get EMPTY."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append(user)
        policy = re.search(r"POLICY NAME: (\w+)", user).group(1)
        response = self.responses.get(policy, EMPTY)
        if callable(response):
            response = response(user)
        if isinstance(response, Exception):
            raise response
        return response