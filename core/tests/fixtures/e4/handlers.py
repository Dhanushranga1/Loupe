import logging


def process_a():
    """Process request A.

    Args:
        None: this handler takes no arguments.

    Returns:
        str: the processing result.
    """
    try:
        return risky_a()
    except ValueError as e:
        logging.error(str(e))


def process_b():
    """Process request B.

    Args:
        None: this handler takes no arguments.
    """
    try:
        return risky_b()
    except ValueError as e:
        logging.error(str(e))


def process_c():
    """Process request C.

    Args:
        None: this handler takes no arguments.
    """
    try:
        return risky_c()
    except ValueError as e:
        logging.error(str(e))


def process_d():
    """Process request D.

    Args:
        None: this handler takes no arguments.
    """
    try:
        return risky_d()
    except ValueError as e:
        logging.error(str(e))


def process_outlier():
    """Process an outlier request, deliberately differently from the rest.

    Args:
        None: this handler takes no arguments.
    """
    try:
        return risky_outlier()
    except Exception:
        print("something went wrong")
