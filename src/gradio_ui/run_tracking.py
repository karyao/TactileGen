"""Session-local publication tickets, reusing diagnostic run IDs."""
import asyncio
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class Ticket:
    run_id: str | None
    review_revision: int = 0
    generation_token: int = 0


class RunTracker:
    def __init__(self):
        self._sessions: dict[str, Ticket] = {}
        self._lock = Lock()
        self._publication_locks = {}
        self._sequence = 0

    def publication(self, session):
        return self._publication_locks.setdefault(session, asyncio.Lock())

    def latest(self, session):
        with self._lock:
            return self._sessions.get(session)

    def invalidate(self, session):
        with self._lock:
            self._sequence += 1
            self._sessions[session] = Ticket(None, generation_token=self._sequence)

    def start(self, session, run_id):
        ticket = Ticket(run_id)
        with self._lock:
            self._sessions[session] = ticket
        return ticket

    def current(self, session, ticket):
        with self._lock:
            return self._sessions.get(session) == ticket

    def matches_review(self, session, run_id, revision):
        with self._lock:
            old = self._sessions.get(session)
            return (old is not None and run_id is not None
                    and old.run_id == run_id and old.review_revision == revision)

    def advance(self, session, run_id, revision, *, edit=False):
        with self._lock:
            old = self._sessions.get(session)
            if old is None or run_id is None or old.run_id != run_id or old.review_revision != revision:
                return None
            ticket = Ticket(run_id, revision + int(edit), old.generation_token + 1)
            self._sessions[session] = ticket
            return ticket

    def remove(self, session):
        with self._lock:
            self._sessions.pop(session, None)

class GuardedUpdates(tuple):
    """Carry publication ownership through Gradio's asynchronous file caching."""
    def __new__(cls, values, session, ticket, cleanup=None):
        instance = super().__new__(cls, values)
        instance.session = session
        instance.ticket = ticket
        instance.cleanup = cleanup
        return instance


def guarded_blocks(gr, tracker):
    """Keep revision changes and Gradio state publication mutually exclusive."""
    class GuardedBlocks(gr.Blocks):
        async def postprocess_data(self, block_fn, predictions, state=None):
            if not isinstance(predictions, GuardedUpdates):
                return await super().postprocess_data(block_fn, predictions, state)
            async with tracker.publication(predictions.session):
                if not tracker.current(predictions.session, predictions.ticket):
                    if predictions.cleanup:
                        predictions.cleanup()
                    predictions = tuple(gr.skip() for _ in block_fn.outputs)
                return await super().postprocess_data(block_fn, predictions, state)
    return GuardedBlocks
