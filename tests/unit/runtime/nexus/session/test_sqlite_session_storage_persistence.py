# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import asyncio
from tempfile import TemporaryDirectory

from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.llm.messages import ChatMessage
from tests._support.builder import prepare_sqlite_db


def test_sqlite_session_storage_persistence():

    async def scenario(tmp_path: str) -> None:
        db_path = f"{tmp_path}/sessions.db"

        # First instance
        storage = SQLiteSessionStorage(db_path=db_path)

        session = await storage.create_session(
            session_id=None,
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            user_id="user-1",
            metadata={"k": "v"},
        )

        message = ChatMessage(
            role="user",
            content="hello world",
        )

        await storage.append_message(
            session_id=session.id,
            message=message,
        )

        # Re-create storage (simulate restart)
        storage = SQLiteSessionStorage(db_path=db_path)

        restored_session = await storage.get_session(session.id)
        assert restored_session is not None
        assert restored_session.id == session.id
        assert restored_session.metadata == {"k": "v"}

        history = await storage.get_history(
            session_id=session.id,
            native_tools=True,
        )

        assert len(history) == 1
        assert history[0].content == "hello world"


    db_path = prepare_sqlite_db("trace.db")
    asyncio.run(scenario(db_path))