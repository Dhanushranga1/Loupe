class Widget:
    """A simple widget with a name and a computed display label."""

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def default_name() -> str:
        return "widget"

    @property
    def label(self) -> str:
        """The display label for this widget."""
        return f"Widget({self.name})"
