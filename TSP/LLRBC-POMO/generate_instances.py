import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
from TSPEnv import TSPEnv as Env

sizes ={'U':20,'Ring':20,'E':50,'Grid':50,'GM':100,'C':100}
for d in sizes.keys():

    env_params = {
        'problem_size': sizes[d],
        'pomo_size': sizes[d],
        'distribution': d
    }
    env = Env(**env_params)
    seed = 0
    num = 10*1000
    env.generate_and_save_probelms(num,seed,folder='../generated_instances/')