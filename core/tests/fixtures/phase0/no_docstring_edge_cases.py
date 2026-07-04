class Plain:
    def greet(self) -> str:
        return "hello"


def only_string_body() -> str:
    "this bare string is the only statement, so it is a real docstring"


def not_first_statement() -> str:
    value = "not the docstring"
    "this bare string is not the first statement and must not be treated as a docstring"
    return value
