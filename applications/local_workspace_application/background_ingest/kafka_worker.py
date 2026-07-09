# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker entrypoint alias (LKW.4E)."""

from local_workspace_application.host.background_worker_main import main

if __name__ == "__main__":
    raise SystemExit(main())
