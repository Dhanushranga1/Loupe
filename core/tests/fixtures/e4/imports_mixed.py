from . import sibling
from .utils import helper
from .models import Order


def uses_relative_imports():
    """Reference the relative imports above so this file has a real body."""
    return sibling, helper, Order
