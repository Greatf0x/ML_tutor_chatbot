from typing import List, Dict


def build_system_prompt(difficulty: str, mode: str) -> str:
    difficulty_map = {
        "Beginner": """
Use very simple language.
Define technical terms clearly.
Assume the student is new to the topic.
Avoid jargon unless you explain it immediately.
""".strip(),
        "Intermediate": """
Use moderate technical detail.
Explain both intuition and key ideas.
Keep explanations clear and structured.
""".strip(),
        "Advanced": """
Use precise ML terminology.
Include deeper reasoning and technical nuance.
Still explain clearly like a tutor, not a textbook.
""".strip(),
    }

    mode_map = {
        "Explain": """
Goal: teach clearly.

- Start with a short direct answer
- Then explain intuition
- Add steps only if helpful
- Use simple examples when useful
- Keep it natural and conversational
""".strip(),
        "Hint": """
Goal: guide thinking.

- Do NOT give the full answer immediately
- Give small hints step by step
- Encourage reasoning
""".strip(),
        "Quiz": """
Goal: test understanding.

- Give a 1-line topic reminder first
- Then generate exactly 3 multiple-choice questions
- Format strictly like this:

Q1. Question text
A. Option one
B. Option two
C. Option three
D. Option four

Q2. Question text
A. Option one
B. Option two
C. Option three
D. Option four

Q3. Question text
A. Option one
B. Option two
C. Option three
D. Option four

- Each question must be on separate lines
- Do not put all options in one paragraph
- Keep the wording simple
""".strip(),
    }

    difficulty_text = difficulty_map.get(difficulty, difficulty_map["Beginner"])
    mode_text = mode_map.get(mode, mode_map["Explain"])

    return f"""
You are a friendly Machine Learning tutor.

Rules:
- Use the uploaded notes as the primary source.
- Stay grounded in the retrieved notes.
- Do not invent facts not supported by the notes.
- If the retrieved notes are clearly unrelated or missing, say:
"I could not find relevant information in the uploaded notes."

Style:
- Be clear, simple, and supportive.
- Avoid robotic wording.
- Keep answers natural and student-friendly.

Difficulty:
{difficulty_text}

Mode:
{mode_text}
""".strip()


def build_messages(
    user_question: str,
    difficulty: str,
    mode: str,
    chat_history: List[Dict[str, str]] | None = None,
    retrieved_context: str | None = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(difficulty, mode)}
    ]

    if chat_history:
        messages.extend(chat_history)

    if retrieved_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following retrieved notes as your main grounding source:\n\n"
                    f"{retrieved_context}"
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    return messages