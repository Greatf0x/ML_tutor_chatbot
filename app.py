from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any

import streamlit as st
import src.application.ui as ui

from answer_evaluator import evaluate_answer
from src.model.ollama_client import OllamaClient
from src.model.prompt_builder import build_messages
from src.rag.chunker import chunk_text
from src.rag.loader import load_text_from_file
from src.utils.config import CONFIG


STRICT_NO_MATCH_RESPONSE = "I could not find relevant information in the uploaded notes."


def response_token_budget(mode: str) -> int:
    budgets = {
        "Explain": 1024,
        "Quiz": 800,
        "Hint": 400,
    }
    return budgets.get(mode, 1024)


def create_retriever() -> Any:
    from src.rag.retriever import SimpleRetriever
    return SimpleRetriever()


def initialize_session_state() -> None:
    defaults = {
        "chat_history": [],
        "messages_for_display": [],
        "notes_loaded": False,
        "loaded_filename": None,
        "upload_widget_key": 0,
        "active_tab": "Chat",
        "chat_input": "",
        "clear_chat_input": False,
        "pending_prompt": None,
        "pending_display_text": None,
        "pending_topic": None,
        "pending_mode_override": None,
        "last_user_question": None,
        "last_feedback": None,
        "next_message_id": 1,
        "copied_message_text": None,
        "copied_message_id": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "retriever" not in st.session_state:
        st.session_state.retriever = None


def current_timestamp() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def next_message_id() -> int:
    message_id = st.session_state.next_message_id
    st.session_state.next_message_id += 1
    return message_id


def message_record(
    role: str,
    content: str,
    source_snippets: list[str] | None = None,
    track_as_topic: bool | None = None,
) -> dict[str, Any]:
    return {
        "id": next_message_id(),
        "role": role,
        "content": content,
        "timestamp": current_timestamp(),
        "source_snippets": source_snippets or [],
        "rating": None,
        "track_as_topic": role == "user" if track_as_topic is None else track_as_topic,
    }


def reset_notes() -> None:
    st.session_state.retriever = None
    st.session_state.notes_loaded = False
    st.session_state.loaded_filename = None
    st.session_state.upload_widget_key += 1


def clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.messages_for_display = []
    st.session_state.clear_chat_input = True
    st.session_state.last_user_question = None
    st.session_state.last_feedback = None
    st.session_state.pending_prompt = None
    st.session_state.pending_display_text = None
    st.session_state.pending_topic = None
    st.session_state.pending_mode_override = None
    st.session_state.copied_message_text = None
    st.session_state.copied_message_id = None
    st.session_state.next_message_id = 1
    reset_notes()


def process_uploaded_file(uploaded_file: Any) -> None:
    if uploaded_file.name == st.session_state.loaded_filename:
        return

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        text = load_text_from_file(temp_path)
        chunks = chunk_text(text)

        print("\n========== FILE DEBUG ==========")
        print("LOADED FILE:", uploaded_file.name)
        print("EXTRACTED TEXT PREVIEW:\n", text[:2000])
        print("NUMBER OF CHUNKS:", len(chunks))
        if chunks:
            print("FIRST CHUNK:\n", chunks[0][:1000])
        print("================================\n")

        if st.session_state.retriever is None:
            st.session_state.retriever = create_retriever()

        st.session_state.retriever.add_chunks(chunks)
        st.session_state.notes_loaded = True
        st.session_state.loaded_filename = uploaded_file.name
        st.success(f"Loaded notes: {uploaded_file.name}")
        st.info(f"Created {len(chunks)} chunks")
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def prepare_chat_request(
    question: str,
    difficulty: str,
    mode: str,
    retrieval_query: str | None = None,
) -> tuple[list[dict[str, str]] | None, list[str], bool]:
    source_snippets: list[str] = []
    retrieved_context = None

    print("\n========== PREPARE CHAT REQUEST ==========")
    print("QUESTION:", question)
    print("NOTES LOADED:", st.session_state.notes_loaded)
    print("RETRIEVER IS NONE:", st.session_state.retriever is None)

    if not st.session_state.notes_loaded or st.session_state.retriever is None:
        print("REFUSAL REASON: notes not loaded or retriever missing")
        print("==========================================\n")
        return None, [], True

    source_snippets = st.session_state.retriever.retrieve(
        retrieval_query or question,
        top_k=3
    )

    print("RETRIEVED SNIPPETS COUNT:", len(source_snippets))

    if not source_snippets:
        print("REFUSAL REASON: no source snippets retrieved")
        print("==========================================\n")
        return None, [], True

    for i, snippet in enumerate(source_snippets, start=1):
        print(f"\n--- SNIPPET {i} ---\n{snippet[:500]}")

    shortened_snippets = [snippet[:800] for snippet in source_snippets]
    retrieved_context = "\n\n---\n\n".join(shortened_snippets)

    recent_history = st.session_state.chat_history[-4:]

    messages = build_messages(
        user_question=question,
        difficulty=difficulty,
        mode=mode,
        chat_history=recent_history,
        retrieved_context=retrieved_context,
    )

    print("MESSAGE BUILD SUCCESS")
    print("==========================================\n")

    return messages, source_snippets, False


def handle_question(
    user_input: str,
    difficulty: str,
    mode: str,
    model_name: str,
    temperature: float,
) -> None:
    question = user_input.strip()
    if not question:
        return

    st.session_state.messages_for_display.append(message_record("user", question))
    st.session_state.last_user_question = question

    messages, source_snippets, should_refuse = prepare_chat_request(question, difficulty, mode)

    if should_refuse:
        response = STRICT_NO_MATCH_RESPONSE
    else:
        client = OllamaClient(CONFIG.ollama_base_url)
        response = client.chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            num_predict=response_token_budget(mode),
        )

    st.session_state.messages_for_display.append(
        message_record("assistant", response, source_snippets)
    )
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})


def stream_question_response(
    user_input: str,
    difficulty: str,
    mode: str,
    model_name: str,
    temperature: float,
    display_user_text: str | None = None,
    track_as_topic: bool = True,
    update_last_user_question: bool = True,
    retrieval_query: str | None = None,
) -> None:
    question = user_input.strip()
    if not question:
        return

    visible_user_text = (display_user_text or question).strip()
    user_message = message_record("user", visible_user_text, track_as_topic=track_as_topic)
    st.session_state.messages_for_display.append(user_message)

    if update_last_user_question:
        st.session_state.last_user_question = question

    ui.render_message_card(user_message)

    messages, source_snippets, should_refuse = prepare_chat_request(
        question, difficulty, mode, retrieval_query
    )

    assistant_message = message_record("assistant", "", source_snippets)
    body_placeholder = ui.render_streaming_assistant(assistant_message)

    if should_refuse:
        assistant_message["content"] = STRICT_NO_MATCH_RESPONSE
        body_placeholder.markdown(assistant_message["content"])
    else:
        client = OllamaClient(CONFIG.ollama_base_url)
        chunks: list[str] = []

        for chunk in client.stream_chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            num_predict=response_token_budget(mode),
        ):
            chunks.append(chunk)
            body_placeholder.markdown("".join(chunks))

        assistant_message["content"] = "".join(chunks).strip()

    st.session_state.messages_for_display.append(assistant_message)
    st.session_state.chat_history.append({"role": "user", "content": visible_user_text})
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_message["content"]}
    )


def queue_follow_up(
    instruction: str,
    mode_override: str | None = None,
    display_text: str | None = None,
) -> None:
    last_question = st.session_state.get("last_user_question")
    draft_question = st.session_state.get("chat_input", "").strip()
    topic = draft_question or last_question

    if not topic:
        st.warning("Type or ask a question first so Quiz or Hint knows which topic to use.")
        return

    st.session_state.pending_prompt = instruction
    st.session_state.pending_display_text = display_text
    st.session_state.pending_topic = topic
    st.session_state.pending_mode_override = mode_override


def process_pending_prompt(
    difficulty: str,
    mode: str,
    model_name: str,
    temperature: float,
) -> None:
    prompt = st.session_state.pending_prompt
    display_text = st.session_state.pending_display_text
    topic = st.session_state.pending_topic
    pending_mode_override = st.session_state.pending_mode_override

    if not prompt or not topic:
        return

    st.session_state.pending_prompt = None
    st.session_state.pending_display_text = None
    st.session_state.pending_topic = None
    st.session_state.pending_mode_override = None

    visible_text = display_text
    if not visible_text and pending_mode_override == "Quiz":
        visible_text = f"Quiz me on: {topic}"
    if not visible_text and pending_mode_override == "Hint":
        visible_text = f"Give me a hint for: {topic}"
    if not visible_text:
        visible_text = topic

    try:
        with st.spinner("Tutor is thinking..."):
            stream_question_response(
                f"{prompt}\n\nOriginal topic: {topic}",
                difficulty,
                pending_mode_override or mode,
                model_name,
                temperature,
                display_user_text=visible_text,
                track_as_topic=False,
                update_last_user_question=False,
                retrieval_query=topic,
            )
        st.rerun()
    except Exception as exc:
        st.error(f"Error: {exc}")


def find_message(message_id: int) -> dict[str, Any] | None:
    for message in st.session_state.messages_for_display:
        if message["id"] == message_id:
            return message
    return None


def apply_message_action(message_id: int, action: str) -> None:
    message = find_message(message_id)
    if not message:
        return

    if action == "copy":
        st.session_state.copied_message_text = message["content"]
        st.session_state.copied_message_id = message_id
        st.toast("Response text is ready below for quick copy.")
        return

    if action in {"helpful", "needs-work"}:
        message["rating"] = action
        label = "helpful" if action == "helpful" else "needs work"
        st.toast(f"Marked response as {label}.")


def build_recent_topics() -> list[str]:
    topics: list[str] = []
    for message in reversed(st.session_state.messages_for_display):
        if (
            message["role"] == "user"
            and message.get("track_as_topic")
            and message["content"] not in topics
        ):
            topics.append(message["content"])
        if len(topics) == 4:
            break
    return topics


def render_chat_tab(
    model_name: str,
    difficulty: str,
    mode: str,
    temperature: float,
) -> None:
    process_pending_prompt(difficulty, mode, model_name, temperature)

    if not st.session_state.messages_for_display:
        ui.render_empty_chat_state()

    for message in st.session_state.messages_for_display:
        action = ui.render_message_card(message)
        if action:
            apply_message_action(message["id"], action)
            st.rerun()

    if st.session_state.copied_message_text:
        ui.render_copied_message(st.session_state.copied_message_text)

    quick_action = ui.render_quick_actions()
    if quick_action:
        quick_action_label, quick_action_prompt = quick_action
        queue_follow_up(quick_action_prompt, display_text=quick_action_label)
        st.rerun()

    question = ui.render_chat_input()
    if question:
        try:
            stream_question_response(question, difficulty, mode, model_name, temperature)
            st.session_state.clear_chat_input = True
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")

    if st.session_state.last_user_question:
        answer = ui.render_answer_evaluator(st.session_state.last_feedback)
        if answer is not None:
            if answer.strip() and len(answer.strip()) > 10:
                with st.spinner("Evaluating your answer..."):
                    st.session_state.last_feedback = evaluate_answer(
                        st.session_state.last_user_question,
                        answer,
                    )
                st.rerun()
            else:
                st.warning("Write a slightly longer answer so the tutor can evaluate it.")


def render_upload_tab() -> None:
    uploaded_file, reset_clicked = ui.render_upload_tab(
        st.session_state.notes_loaded,
        st.session_state.loaded_filename,
    )
    if reset_clicked:
        reset_notes()
        st.rerun()

    if uploaded_file:
        st.success("File uploaded successfully.")
        st.markdown(f"**File Name:** {uploaded_file.name}")
        st.markdown(f"**File Type:** {uploaded_file.type}")
        st.markdown(f"**File Size:** {uploaded_file.size} bytes")
        process_uploaded_file(uploaded_file)


def render_settings_tab(
    model_name: str,
    difficulty: str,
    mode: str,
    temperature: float,
) -> None:
    clear_chat_clicked = ui.render_settings_tab(
        model_name,
        difficulty,
        mode,
        temperature,
        len(st.session_state.messages_for_display),
        st.session_state.notes_loaded,
        st.session_state.loaded_filename,
    )
    if clear_chat_clicked:
        clear_chat()
        st.rerun()


def render_main_panel(
    model_name: str,
    difficulty: str,
    mode: str,
    temperature: float,
    active_tab: str,
) -> None:
    ui.render_main_panel(CONFIG.app_icon, CONFIG.app_title)
    render_chat_tab(model_name, difficulty, mode, temperature)

    if active_tab == "Upload":
        render_upload_tab()
    elif active_tab == "Settings":
        render_settings_tab(model_name, difficulty, mode, temperature)

    ui.close_main_panel()


def main() -> None:
    st.set_page_config(
        page_title=CONFIG.app_title,
        page_icon=CONFIG.app_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initialize_session_state()
    ui.inject_styles()
    ui.render_shell_start()

    left_col, main_col, right_col = st.columns([1.05, 3.55, 1.2])

    model_name = "mistral"
    difficulty = "Beginner"
    mode = "Explain"
    temperature = 0.2

    with left_col:
        model_name, difficulty, mode, temperature, active_tab, clear_from_left = ui.render_left_panel(
            CONFIG.app_icon,
            st.session_state.active_tab,
            model_name,
            difficulty,
            mode,
            temperature,
        )
        st.session_state.active_tab = active_tab

        if clear_from_left:
            clear_chat()
            st.rerun()

    with main_col:
        render_main_panel(model_name, difficulty, mode, temperature, active_tab)

    with right_col:
        side_action = ui.render_right_panel(mode, build_recent_topics())
        if side_action:
            action_text, action_mode = side_action
            queue_follow_up(action_text, action_mode)
            st.rerun()

    ui.render_shell_end()


if __name__ == "__main__":
    main()