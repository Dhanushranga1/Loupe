def format_response(payload):
    """Called directly by many unrelated call sites below — the deliberate,
    unambiguous god-object/hub outlier this fixture exists to demonstrate."""
    return {"data": payload}
