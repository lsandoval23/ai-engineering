"""The prompt is a template with roles, and its variable is exactly ``text``."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.chain import prompt


def test_input_variable_is_exactly_text():
    assert prompt.input_variables == ["text"]
    assert "feedback" in prompt.optional_variables


def test_renders_system_and_human_roles():
    messages = prompt.format_messages(text="La API devuelve timeouts.")
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[1].content == "La API devuelve timeouts."
    assert "technical analyst" in messages[0].content


def test_feedback_placeholder_appends_a_message_only_when_given():
    feedback = HumanMessage(content="Your previous answer was invalid.")
    messages = prompt.format_messages(text="hola", feedback=[feedback])
    assert len(messages) == 3
    assert messages[2] is feedback


def test_user_text_is_substituted_not_interpolated():
    """Braces in user input survive: the template is not an f-string."""
    messages = prompt.format_messages(text="config = {'debug': True}")
    assert messages[1].content == "config = {'debug': True}"
