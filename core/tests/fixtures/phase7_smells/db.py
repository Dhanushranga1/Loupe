class Db:
    """Stand-in for a real DB session/client — the query() method is what the
    N+1 check looks for being called from inside a loop."""

    def query(self, item_id):
        return {"id": item_id}

    def query_many(self, item_ids):
        return [{"id": i} for i in item_ids]


db = Db()
