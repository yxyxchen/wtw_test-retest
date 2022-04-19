import numpy as np
import pandas as pd
import importlib
import code 

code.interact(local = dict(locals(), **globals()))
bonus_sess1 = pd.read_csv("data/passive/bonus_sess1_batch2_B.csv", header = None, names = ['worker_id', 'bonus'])
bonus_sess2 = pd.read_csv("data/passive/bonus_sess2_batch2_B.csv", header = None, names = ['worker_id', 'bonus'])


df = bonus_sess1.merge(bonus_sess2, on = 'worker_id', suffixes = ['_sess1', '_sess2'])
df['max_bonus'] = np.maximum(df['bonus_sess1'], df['bonus_sess2'])
df['over_paid'] = df['max_bonus'] - df['bonus_sess2']
df[['worker_id', 'max_bonus']].to_csv("test.csv", header = None, index = None)