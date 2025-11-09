# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import logging

# Ustawienia globalne loggera
logging.basicConfig(
    level=logging.INFO,  # pokazuj INFO i wyżej (DEBUG pokaże więcej)
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True  # nadpisuje poprzednie konfiguracje
)
