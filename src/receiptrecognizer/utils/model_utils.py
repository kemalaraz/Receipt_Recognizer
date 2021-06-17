from typing import Dict
from collections import OrderedDict

def copyStateDict(state_dict:Dict) -> Dict:
    """
    In order to convert the model dictionary to torch.

    Args:
        state_dict (Dict): Model dict.

    Returns:
        Dict: Converted torch model state dict.
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name.replace("basenet", "backbone")] = v
    return new_state_dict