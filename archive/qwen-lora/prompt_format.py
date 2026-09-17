from __future__ import annotations


def build_messages(instruction: str, context: str = "") -> list[dict[str, str]]:
    user_content = instruction
    if context:
        user_content = f"{instruction}\n\nContext:\n{context}"
    return [
        {
            "role": "system",
            "content": (
                "You answer questions about the synthetic employee test "
                "documents. Use only the supplied employee-domain knowledge."
            ),
        },
        {"role": "user", "content": user_content},
    ]
