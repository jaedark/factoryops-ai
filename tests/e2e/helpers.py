from types import SimpleNamespace

from google.genai import types


def build_tool_response(
    tool_name: str,
    tool_args: dict,
):
    function_call = types.FunctionCall(
        name=tool_name,
        args=tool_args,
    )
    content = types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(
                name=tool_name,
                args=tool_args,
            )
        ],
    )
    return SimpleNamespace(
        function_calls=[function_call],
        candidates=[SimpleNamespace(content=content)],
        text=None,
    )


def build_text_response(text: str):
    return SimpleNamespace(
        function_calls=None,
        candidates=[],
        text=text,
    )
