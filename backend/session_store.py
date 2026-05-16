from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import redis

from models import Message, Session, utc_now


class SessionStore:
    def __init__(self, max_sessions: int, ttl_hours: int) -> None:
        self.max_sessions = max_sessions
        self.ttl = timedelta(hours=ttl_hours)
        self.sessions: "OrderedDict[str, Session]" = OrderedDict()
        self.messages: Dict[str, List[Message]] = {}

    def _is_expired(self, session: Session, now: datetime) -> bool:
        return now - session.updated_at > self.ttl

    def prune(self) -> None:
        now = utc_now()
        expired = [session_id for session_id, session in self.sessions.items() if self._is_expired(session, now)]
        for session_id in expired:
            self.delete_session(session_id)
        while len(self.sessions) > self.max_sessions:
            session_id, _ = self.sessions.popitem(last=False)
            self.messages.pop(session_id, None)

    def create_session(self, title: Optional[str] = None, session_id: Optional[str] = None) -> Session:
        self.prune()
        session = Session(title=title or "New conversation")
        if session_id:
            session.id = session_id
        self.sessions[session.id] = session
        self.messages[session.id] = []
        self.prune()
        return session

    def get_or_create_session(self, session_id: Optional[str], title: Optional[str] = None) -> Session:
        self.prune()
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            self.sessions.move_to_end(session_id)
            return session
        return self.create_session(title=title, session_id=session_id)

    def list_sessions(self) -> List[Session]:
        self.prune()
        return list(reversed(self.sessions.values()))

    def get_session(self, session_id: str) -> Optional[Session]:
        self.prune()
        session = self.sessions.get(session_id)
        if session:
            self.sessions.move_to_end(session_id)
        return session

    def delete_session(self, session_id: str) -> bool:
        existed = session_id in self.sessions
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)
        return existed

    def add_message(self, message: Message) -> Message:
        session = self.get_or_create_session(message.session_id)
        if session.title == "New conversation" and message.role.value == "user":
            session.title = message.text[:40]
        session.updated_at = utc_now()
        items = self.messages.setdefault(session.id, [])
        items.append(message)
        session.message_count = len(items)
        self.sessions[session.id] = session
        self.sessions.move_to_end(session.id)
        return message

    def get_messages(self, session_id: str) -> List[Message]:
        self.prune()
        return list(self.messages.get(session_id, []))


class RedisSessionStore(SessionStore):
    def __init__(self, max_sessions: int, ttl_hours: int, redis_url: str) -> None:
        super().__init__(max_sessions=max_sessions, ttl_hours=ttl_hours)
        self.redis_url = redis_url
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.session_index = "quadtrix:sessions"

    def _ttl_seconds(self) -> int:
        return int(self.ttl.total_seconds())

    def _score(self, session: Session) -> float:
        return session.updated_at.timestamp()

    def _session_key(self, session_id: str) -> str:
        return f"quadtrix:session:{session_id}"

    def _messages_key(self, session_id: str) -> str:
        return f"quadtrix:messages:{session_id}"

    def prune(self) -> None:
        cutoff = (utc_now() - self.ttl).timestamp()
        expired = self.client.zrangebyscore(self.session_index, "-inf", cutoff)
        for session_id in expired:
            self.delete_session(str(session_id))
        count = self.client.zcard(self.session_index)
        if count > self.max_sessions:
            overflow = int(count - self.max_sessions)
            oldest = self.client.zrange(self.session_index, 0, overflow - 1)
            for session_id in oldest:
                self.delete_session(str(session_id))

    def create_session(self, title: Optional[str] = None, session_id: Optional[str] = None) -> Session:
        self.prune()
        session = Session(title=title or "New conversation")
        if session_id:
            session.id = session_id
        self.client.setex(self._session_key(session.id), self._ttl_seconds(), session.model_dump_json())
        self.client.delete(self._messages_key(session.id))
        self.client.expire(self._messages_key(session.id), self._ttl_seconds())
        self.client.zadd(self.session_index, {session.id: self._score(session)})
        self.prune()
        return session

    def get_or_create_session(self, session_id: Optional[str], title: Optional[str] = None) -> Session:
        self.prune()
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(title=title, session_id=session_id)

    def list_sessions(self) -> List[Session]:
        self.prune()
        session_ids = self.client.zrevrange(self.session_index, 0, self.max_sessions - 1)
        sessions: List[Session] = []
        for session_id in session_ids:
            session = self.get_session(str(session_id))
            if session:
                sessions.append(session)
        return sessions

    def get_session(self, session_id: str) -> Optional[Session]:
        raw = self.client.get(self._session_key(session_id))
        if not raw:
            self.client.zrem(self.session_index, session_id)
            return None
        session = Session.model_validate_json(str(raw))
        self.client.zadd(self.session_index, {session.id: self._score(session)})
        return session

    def delete_session(self, session_id: str) -> bool:
        deleted = bool(self.client.delete(self._session_key(session_id), self._messages_key(session_id)))
        self.client.zrem(self.session_index, session_id)
        return deleted

    def add_message(self, message: Message) -> Message:
        session = self.get_or_create_session(message.session_id)
        if session.title == "New conversation" and message.role.value == "user":
            session.title = message.text[:40]
        session.updated_at = utc_now()
        session.message_count = int(self.client.rpush(self._messages_key(session.id), message.model_dump_json()))
        self.client.setex(self._session_key(session.id), self._ttl_seconds(), session.model_dump_json())
        self.client.expire(self._messages_key(session.id), self._ttl_seconds())
        self.client.zadd(self.session_index, {session.id: self._score(session)})
        return message

    def get_messages(self, session_id: str) -> List[Message]:
        self.prune()
        rows = self.client.lrange(self._messages_key(session_id), 0, -1)
        return [Message.model_validate_json(str(row)) for row in rows]


def build_session_store(max_sessions: int, ttl_hours: int, redis_url: str) -> SessionStore:
    if redis_url:
        return RedisSessionStore(max_sessions=max_sessions, ttl_hours=ttl_hours, redis_url=redis_url)
    return SessionStore(max_sessions=max_sessions, ttl_hours=ttl_hours)
