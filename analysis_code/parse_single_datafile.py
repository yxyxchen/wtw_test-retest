
def parse_data(rawdatapath, cleandatadir):
    """ Script to parse data of online experiments
    Inputs:
        rawdatapath: Path of the raw data file. 
        cleandatadir: Directory to save the clean data file
    Usage:
        python parse_data.py rawdatapath cleandatadir
    """

    import numpy as np
    import pandas as pd
    import re
    import glob
    import os

    # read raw data
    rawdata = pd.read_csv(rawdatapath)

    # check workerId
    workerId = np.unique(rawdata['workerId'])[0]
    if workerId:
        print("Parse data for worker " + workerId)
    else:
        print("No worker ID is recorded!")
        workerId = "unknown"
        
        
    # find rows that record task data
    taskdata = rawdata[[bool(re.search("wtw-.*-block", x)) for x in rawdata.trial_type]]

    # check the number of saved task blocks 
    numblock = taskdata.shape[0]
    print("%d blocks are saved."%numblock)

    # parse task data 
    def parse_task_variable(variable, data):
        """Parse a given task variable from rawdata
        """
        vals = [x for x in data[variable].split(",")]
        if variable != "condition":
            vals = [float(x) for x in vals]
        if variable in ['scheduledDelay', 'RT', 'timeWaited', 'rewardedTime', 'sellTime', 'trialStartTime', 'trialReadyTime']:
            vals = [float(x) / 1000 for x in vals] # convert ms into s
        return vals

    # potential recorded variables. Notice, trialReadyTime is not recorded in passive-waiting tasks
    variables = ['trialIdx', 'condition', 'scheduledDelay', 'scheduledReward', 'rewardedTime', 'RT', 'timeWaited', 'trialEarnings', 'totalEarnings', 'sellTime', 'trialStartTime', 'trialReadyTime']

    # loop over blocks
    cleandata = []
    for i in range(numblock):
        taskdata_in_this_block = taskdata.iloc[i]
        cleandata_in_this_block = dict()
        num_valid_entry = 10000000000 # keep track of the number of valid entries. Notice that sometimes a trial ends midway 
        for variable in variables:
            if variable in taskdata_in_this_block:
                cleandata_in_this_block[variable] = parse_task_variable(variable, taskdata_in_this_block)
                if len(cleandata_in_this_block[variable]) < num_valid_entry:
                    num_valid_entry = len(cleandata_in_this_block[variable])

        # make sure all variables have the same number of entries
        for variable in cleandata_in_this_block:
            cleandata_in_this_block[variable] = cleandata_in_this_block[variable][:num_valid_entry]
        # append data 
        cleandata.append(pd.DataFrame(cleandata_in_this_block))
        # check the block duration
        if 'trialReadyTime' in cleandata_in_this_block:
            blockduration = max(cleandata_in_this_block['sellTime']) - cleandata_in_this_block['trialReadyTime'][0]
        else:
            blockduration = max(cleandata_in_this_block['sellTime']) - cleandata_in_this_block['trialStartTime'][0]
        print('The block duration, measured by last sellTime - first trialStartTime, is %.2f s'%blockduration)
    cleandata = pd.concat(cleandata)

    # save clean data 
    cleandata.to_csv(os.path.join(cleandatadir, workerId+".csv"))



if __name__ == "__main__":
    import sys
    rawdatapath = sys.argv[1]
    cleandatadir = sys.argv[2]
    parse_data(sys.argv[1], sys.argv[2])