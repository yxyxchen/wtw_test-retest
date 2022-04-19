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
import code


def manual_check(expname, sess):
    """ rm task files where participants didn't star the task and list task files with duplicated worker IDs
    """
    taskdata_dir  = os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "data_ok")
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
            print("Remove this file!")
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
    hdrdata.loc[hdrdata.duptask, :].to_csv(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "duplicated.csv"))
    hdrdata_empty.to_csv(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "empty.csv"))

def delete_data_saved_twice(expname, sess):
    duplicated_df = pd.read_csv(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "duplicated.csv"), index_col = 0)
    
    duplicated_count_df = duplicated_df.worker_id.value_counts()
    # code.interact(local = dict(globals(), **locals()))
    for worker_id in duplicated_count_df.index:
        if duplicated_count_df[worker_id] != 2:
            print(worker_id)
        else:
            row_nums = dict()
            for path in duplicated_df.loc[duplicated_df.worker_id  ==  worker_id, 'path']:
                tmp = pd.read_csv(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "data_ok", path))
                print(tmp.shape[0])
                row_nums[tmp.shape[0]] = path
            print(row_nums)
            if 19 in row_nums and 20 in row_nums:
                os.remove(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "data_ok", row_nums[19]))
            else:
                if worker_id == "A2NA6X1SON3KFH" and sess == 2:
                    # this participant did the task twice. Delete the data for the second time
                    os.remove(os.path.join("..", "task_code", "%s_sess%d"%(expname, sess), "manual_check", "data_ok", "wtw_online_sess2_passive_A2NA6X1SON3KFH_SESSION_2022-02-09_19h39.17.264.csv"))
                else:
                    code.interact(local = dict(locals(), **globals()))
                # A2NA6X1SON3KFH completed the tasks twice, delete the second one, in pass_sess2
                # I also need to manually delete my own data
                print('irregular row numbers for', worker_id)


if __name__ == "__main__":
    print("check data files for SESS2")
    manual_check('passive', 2)
    delete_data_saved_twice('passive', 2)
    # print("check data files for SESS2")
    # manual_check(2)
    
    # parsedata(2)
