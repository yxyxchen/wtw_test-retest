########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import re
import itertools
import copy # pay attention to copy 
import code
from subFxs import expParas
from datetime import datetime


def manual_check(sess):
    """ rm task files where participants didn't star the task and list task files with duplicated worker IDs
    """
    taskdata_dir  = os.path.join("..", "task_code", "active_sess%d"%sess, "manual_check", "data_ok")
    files = glob.glob(os.path.join(taskdata_dir, "*"))
    hdrdata = pd.DataFrame()
    hdrdata_empty = pd.DataFrame()
    for rawdatapath in files:
        # read raw data
        rawdata = pd.read_csv(rawdatapath)

        # check workerId
        workerId = np.unique(rawdata['workerId'])[0]
        # print("Load " + rawdatapath)
        if workerId:
            print("Parse data for worker " + workerId)
            # pass 
        else:
            print("No worker ID is recorded!")
            workerId = "unknown"

        # check whether this participant didn't start this task 
        if max(rawdata.trial_index) < 10:
            print(rawdatapath + " didn't start this game")
            os.remove(rawdatapath)
            hdrdata_empty_entry = pd.DataFrame({
                    "worker_id": workerId,
                    "path": rawdatapath.split("/")[-1]
                }, index = [0])
            hdrdata_empty = pd.concat([hdrdata_empty, hdrdata_empty_entry])
        else:
            hdrdata_entry = pd.DataFrame({
                    "worker_id": workerId,
                    "path": rawdatapath.split("/")[-1]
                }, index = [0])
        hdrdata = pd.concat([hdrdata, hdrdata_entry])

    # code.interact(local = dict(globals(), **locals()))
    hdrdata['duptask'] = hdrdata['worker_id'].duplicated(keep = False)
    hdrdata.loc[hdrdata.duptask, :].to_csv(os.path.join("..", "task_code", "active_sess%d"%sess, "manual_check", "duplicated.csv"))
    hdrdata_empty.to_csv(os.path.join("..", "task_code", "active_sess%d"%sess, "manual_check", "empty.csv"))

if __name__ == "__main__":
    print("check data files for SESS1")
    manual_check(1)
    print("check data files for SESS2")
    manual_check(2)
    
    # parsedata(2)
