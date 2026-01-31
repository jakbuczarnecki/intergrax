# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

class BudgetExceededError(Exception):
    """
    Raised when a budget or quota constraint is violated.
    """
    pass
