"""Session-local publication tickets, reusing diagnostic run IDs."""
import asyncio
import logging
from contextlib import asynccontextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class Ticket:
    run_id: str | None
    review_revision: int = 0
    generation_token: int = 0


@dataclass
class PublicationEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    expired: bool = False


class RunTracker:
    def __init__(self):
        self._sessions: dict[str, Ticket] = {}
        self._lock = Lock()
        self._entry_lock = Lock()
        self._publication_locks = {}
        self._sequence = 0

    @asynccontextmanager
    async def publication(self, session):
        with self._entry_lock:
            entry = self._publication_locks.setdefault(session, PublicationEntry())
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            with self._entry_lock:
                entry.users -= 1
                self._reclaim(session, entry)

    def _reclaim(self, session, entry):
        if entry.expired and entry.users == 0:
            self._publication_locks.pop(session, None)

    def retain(self, session):
        with self._entry_lock:
            entry = self._publication_locks.setdefault(session, PublicationEntry())
            entry.users += 1
        def release():
            with self._entry_lock:
                entry.users -= 1
                self._reclaim(session, entry)
        return release

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
        with self._entry_lock:
            entry = self._publication_locks.get(session)
            if entry is not None:
                entry.expired = True
                self._reclaim(session, entry)

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
                if (getattr(self, "review_stamp", None) is not None
                        and block_fn.outputs and block_fn.outputs[-1] is self.review_stamp
                        and len(predictions) == len(block_fn.outputs) - 1):
                    predictions = (*predictions, gr.skip())
                return await super().postprocess_data(block_fn, predictions, state)
            async with tracker.publication(predictions.session):
                if not tracker.current(predictions.session, predictions.ticket):
                    if predictions.cleanup:
                        predictions.cleanup()
                    return await super().postprocess_data(
                        block_fn, tuple(gr.skip() for _ in block_fn.outputs), state
                    )
                # Stage component config and State writes until all file caching succeeds.
                staged = copy(state) if state is not None else None
                if staged is not None:
                    staged.state_data = dict(state.state_data)
                    staged._state_ttl = dict(state._state_ttl)
                    staged.config_values = deepcopy(state.config_values)
                    staged.blocks_config = copy(state.blocks_config)
                    staged.blocks_config.blocks = dict(state.blocks_config.blocks)
                try:
                    values = predictions
                    if getattr(self, "review_stamp", None) is not None and block_fn.outputs and block_fn.outputs[-1] is self.review_stamp:
                        ticket = getattr(predictions, "ticket", None)
                        stamp = [ticket.run_id, ticket.review_revision] if ticket and ticket.run_id else None
                        values = (*predictions, gr.update(value=stamp))
                    output = await super().postprocess_data(block_fn, values, staged)
                except asyncio.CancelledError:
                    if getattr(predictions, "cleanup", None):
                        predictions.cleanup()
                    raise
                except Exception:
                    logging.getLogger("src.gradio_app").exception("Result publication failed")
                    if getattr(predictions, "cleanup", None):
                        predictions.cleanup()
                    tracker.invalidate(predictions.session)
                    controls = getattr(self, "analysis_controls", ())
                    errors = getattr(self, "publication_errors", ())
                    recovery = []
                    for component in block_fn.outputs:
                        if component in controls:
                            recovery.append(gr.update(interactive=True))
                        elif component in errors:
                            recovery.append(gr.update(value=(
                                '<div role="alert">Results could not be displayed. '
                                'Analyze the chart again.</div>'
                            ), visible=True))
                        elif component.stateful or component is getattr(self, "review_stamp", None):
                            recovery.append(gr.update(value=None))
                        else:
                            recovery.append(gr.skip())
                    return await super().postprocess_data(block_fn, recovery, state)
                if state is not None:
                    state.state_data = staged.state_data
                    state._state_ttl = staged._state_ttl
                    state.config_values = staged.config_values
                    state.blocks_config = staged.blocks_config
                return output

    return GuardedBlocks
