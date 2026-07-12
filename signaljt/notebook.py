"""Inline notebook login widget (Hugging Face style).

Keeps the key out of cell output and the saved ``.ipynb``: the token is typed
into a masked ``ipywidgets`` password box, never echoed.
"""
from __future__ import annotations


def notebook_login(validate: bool = True) -> None:
    """Render an inline login form in a Jupyter/Colab cell."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "notebook_login() needs ipywidgets. Install with: pip install 'signaljt[notebook]'"
        ) from e

    from .auth import login

    token_box = widgets.Password(description="API key:", layout=widgets.Layout(width="420px"))
    button = widgets.Button(description="Login", button_style="success")
    output = widgets.Output()

    def _on_click(_):
        with output:
            output.clear_output()
            token = token_box.value.strip()
            if not token:
                print("Please paste your API key first.")
                return
            try:
                login(api_key=token, validate=validate)
                token_box.value = ""  # clear the secret from the widget state
            except Exception as e:  # noqa: BLE001 - surface to the user
                print("Login failed: %s" % e)

    button.on_click(_on_click)
    display(widgets.VBox([
        widgets.HTML("<b>Log in to the OI-WER API</b><br>Your key is stored locally at "
                     "<code>~/.cache/signaljt/token</code> and never shown in output."),
        widgets.HBox([token_box, button]),
        output,
    ]))