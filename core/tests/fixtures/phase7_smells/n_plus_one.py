from db import db


def get_all_with_details_smelly(ids):
    """A DB query call made inside a loop, once per iteration — the N+1 smell."""
    results = []
    for item_id in ids:
        item = db.query(item_id)
        results.append(item)
    return results


def get_all_with_details_clean(ids):
    """A single batched query outside any loop — clean."""
    return db.query_many(ids)
