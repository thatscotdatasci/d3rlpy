from d3rlpy.datasets import get_cartpole 
from d3rlpy.algos import DQN

##############################################################################
# Very simple script that only aims to save the learned policy to another file
##############################################################################

dataset, env = get_cartpole()

dqn = DQN()
dqn.build_with_dataset(dataset)
dqn.load_model('/home/ajc348/rds/hpc-work/d3rlpy/d3rlpy_logs/DQN_20220510100030/model_24740.pt')

# save the greedy-policy as TorchScript
dqn.save_policy('/home/ajc348/rds/hpc-work/d3rlpy/d3rlpy_logs/DQN_20220510100030/policy.pt')

# save the greedy-policy as ONNX
# dqn.save_policy('/home/ajc348/rds/hpc-work/d3rlpy/d3rlpy_logs/DQN_20220510100030/policy.onnx', as_onnx=True)
