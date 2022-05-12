REGISTRY = {}

from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC
from .lilac_controller import LilacMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["separate_mac"]=SeparateMAC
REGISTRY["lilac_mac"]=LilacMAC