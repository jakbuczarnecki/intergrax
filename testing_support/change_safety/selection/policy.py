# © Artur Czarnecki. All rights reserved.

from testing_support.change_safety.data.loaders import load_evol_x_data_document


def selection_policy_forbid_empty() -> bool:
    policy = load_evol_x_data_document("selection_policy.v1.json")
    return bool(policy.get("forbid_empty_selection"))
