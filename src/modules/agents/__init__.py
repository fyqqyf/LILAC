REGISTRY = {}

from .rnn_agent import RNNAgent
from .gmm_role import GMMroleAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["GMMrole"]=GMMroleAgent
