from .q_learner import QLearner
from .lilac_learner import LilacLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY['lilac_learner']=LilacLearner
