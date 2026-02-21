# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import asyncio
from tempfile import TemporaryDirectory

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from tests._support.builder import prepare_sqlite_db


def test_session_manager_with_sqlite_storage() -> None:
    async def scenario(db_path: str) -> None:

        storage = SQLiteSessionStorage(db_path=db_path)
        manager = SessionManager(storage=storage)

        session = await manager.create_session(
            session_id=None,
            user_id="user-1",
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            metadata={"source": "test"},
        )

        message = ChatMessage(
            role="user",
            content="integration test",
        )

        await manager.append_message(
            session_id=session.id,
            message=message,
        )

        storage = SQLiteSessionStorage(db_path=db_path)
        manager = SessionManager(storage=storage)

        restored_session = await manager.get_session(session.id)
        assert restored_session is not None
        assert restored_session.id == session.id

        history = await manager.get_history_for_session(
            session_id=session.id,
            native_tools=True,
        )

        assert len(history) == 1
        assert history[0].content == "integration test"
    
    db_path = prepare_sqlite_db("sessions.db")
    asyncio.run(scenario(db_path))