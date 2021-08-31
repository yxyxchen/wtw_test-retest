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

############################# parse_original data files ############################
def parsedata(sess):
    ############### input variables ###############
    taskdir  = os.path.join("..", "task-code", "task-sess%d"%sess)
    taskdata_outdir = os.path.join("data")
    if not os.path.exists(taskdata_outdir):
        os.makedirs(taskdata_outdir) 

    consentfile = os.path.join(taskdir, "consent.csv")
    consent_out = os.path.join(taskdata_outdir, "consent_sess%d.csv"%sess)
    consentfile_sess1 = os.path.join(taskdata_outdir, "consent_sess1.csv")

    selfreportfile = os.path.join(taskdir, "selfreport.csv")
    selfreport_out = os.path.join(taskdata_outdir, "selfreport_sess%d.csv"%sess)

    DD_choices = {
        "1": "smaller",
        "2": "larger"
    }
    BIS_choices = {
        "1": "Rarely/Never",
        "2": "Occasionally",
        "3": "Often",
        "4": "Almost Always / Always"
    }
    UP_choices = {
        "1": "Agree strongly",
        "2": "Agree somewhat",
        "3": "Disagree somewhat",
        "4": "Disagree strongly"
    }

    taskdata_dir = os.path.join(taskdir, "data")
    hdrdata_out = os.path.join("data", "hdrdata_sess%d.csv"%sess)
    ##################### sub functions ###########################
    def parse_consent_data():
        """Parse consent data for the given session. 
        """
        #################### parse consent data ##############################
        # In session 1, we read in demographic and generate unidentifiable IDs. 
        # In session 2, we match these variables as recorded in session 1.
        consentdata = pd.read_csv(consentfile)
        if sess == 1:
            consentdata = consentdata[['workerId', "EndDate", 'demo1', 'demo2', 'demo3', 'demo4', 'demo5', 'demo7']]
            consentdata = consentdata.rename(columns={"workerId": "worker_id", 
                                                      "EndDate": "consent_date",
                                                      "demo1": "age",
                                                      "demo2": "gender",
                                                      "demo3": "education",
                                                      "demo4": "handness",
                                                      "demo5": "language",
                                                      "demo7": "race"})
            # delete duplicated entries. Sometimes a participant can sign a consent multiple times. 
            dup = consentdata.worker_id.duplicated(keep = 'last')
            if any(dup):
                print("Some participants signed the consent multiple times.")
                print(consentdata.loc[consentdata.worker_id.duplicated(keep = False)])
                print("Deleting duplicated entries.....")
                consentdata = consentdata.loc[~dup]
                print("Duplicated entries deleted!")
            # add unidentifiable IDs
            consentdata.sort_values("consent_date", ignore_index = True)
            consentdata.insert(0, "id", ["s" + str(x + 1).zfill(4) for x in range(consentdata.shape[0])]) 
        else:
            # Yeah I need to debug this issue later
            consentdata = consentdata[['workerId', "EndDate"]]
            consentdata = consentdata.rename(columns={"workerId": "worker_id", 
                                                      "EndDate": "consent_date"})

            # match unidentifiable IDs
            consentdata_sess1 = pd.read_csv(consentfile_sess1)
            consentdata = consentdata.join(consentdata_sess1.drop("consent_date", axis = 1).set_index("worker_id"), on = "worker_id").drop("worker_id", axis = 1)
            tmp = consentdata.pop("id"); consentdata.insert(0, "id", tmp)
            consentdata.sort_values(by = "id", inplace = True)
        ################## save data #######################
        consentdata.to_csv(consent_out, index = False)

    def parse_selfreport_data():
        """Parse selfreport data for session 1 
        """
        ################ parse data ##########################
        # code.interact(local = dict(globals(), **locals()))
        selfreportdata = pd.read_csv(selfreportfile)
        consentdata = pd.read_csv(consentfile_sess1)
        selfreportdata = selfreportdata.drop(['StartDate', 'Status', 'IPAddress', 'Progress',\
                                              'RecordedDate', 'ResponseId', 'RecipientLastName',\
                                              'RecipientFirstName', 'RecipientEmail','ExternalReference',\
                                              'LocationLatitude', 'LocationLongitude', 'DistributionChannel',\
                                              'UserLanguage', 'assignmentId', 'hitId'], axis = 1)
        selfreportdata = selfreportdata.rename(columns={"workerId": "worker_id", "Finished": "selfreport_finished", "Duration (in seconds)": "selfreport_duration",
                                "EndDate": "selfreport_date"})
        ## sort entries by date
        selfreportdata = selfreportdata.sort_values("selfreport_date", ignore_index = True)

        ## match id and delete worker_id
        # code.interact(local = dict(globals(), **locals()))
        selfreportdata = pd.merge(selfreportdata, consentdata[['worker_id', 'id']], how = "left", left_on = ["worker_id"], right_on = ["worker_id"])
        selfreportdata.pop("worker_id")
        tmp = selfreportdata.pop("id"); selfreportdata.insert(0, "id", tmp) # 
        selfreportdata.sort_values('id', inplace  = True, ignore_index = True) # sort by id 

        # save data
        selfreportdata.to_csv(selfreport_out,  index = False)

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
        print("Load " + rawdatapath)
        if workerId:
            print("Parse data for worker " + workerId)
        else:
            print("No worker ID is recorded!")
            workerId = "unknown"
            
        # check counterbalance group
        cb = np.unique(rawdata['cb'])[0]
        if cb:
            print("Counterbalance group: " + cb)
        else:
            print("No counterbalance group is recorded!")
            cb = "unknown"

        # record blockdurations 
        blockdurations = []

        # find rows that record task data
        taskdata = rawdata[[bool(re.search("wtw-.*-block", x)) for x in rawdata.trial_type]]

        # check the number of saved task blocks 
        numblock = taskdata.shape[0]
        # print("%d blocks are saved."%numblock)

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
            # print('The block duration, measured by last sellTime - first trialStartTime, is %.2f s'%blockduration)
            blockdurations.append(blockduration)
        if numblock > 0:
            cleandata = pd.concat(cleandata)

        # save clean data 
        # cleandata.to_csv(os.path.join(cleandatadir, workerId+".csv"))
        return cleandata, workerId, cb, numblock, blockdurations
    def parse_task_data(sess):
        """Loop over all taskdata files under ../task-code/task-sess*. Convert all taskdata files that are not empty
        """
        consentdata = pd.read_csv(consentfile_sess1)
        files = glob.glob(os.path.join(taskdata_dir, "*"))
        hdrdata = pd.DataFrame()
        bonusdata = pd.DataFrame()

        # Loop over all taskdata files under ../task-code/task-sess*
        if os.path.exists(os.path.join('log','hdrdata_empty_sess%d.csv'%sess)):
            os.remove(os.path.join('log','hdrdata_empty_sess%d.csv'%sess))
        date_parser = re.compile(os.path.join(taskdata_dir, "wtw_.*_PARTICIPANT_SESSION_(.*)_(.*)h([0-9][0-9]).*.csv"))
        for file in files:
            cleandata, worker_id, cb, numblock, blockdurations = parse_data(file, taskdata_outdir)
            try:
                thisid = consentdata['id'][np.where(consentdata['worker_id'] == worker_id)[0]].values[0]
            except:
                code.interact(local = dict(globals(), **locals()))
            # if the file is not empty, convert it to a readable format and add an entry to hdrdata and bonusdata 
            if numblock > 0:
                dateval = date_parser.findall(file)[0][0].replace("-", "/")
                hourval = date_parser.findall(file)[0][1]
                minval = hour = date_parser.findall(file)[0][2]
                date = datetime.strptime(dateval + " " + hourval + ":" + minval, '%Y/%m/%d %H:%M')
                thisentry = pd.DataFrame(                    {
                    "id": thisid,
                    "cb": cb,
                    "sess": sess,
                    "date": date,
                    "cptask": all(np.array(blockdurations) > expParas.blocksec - 20) and numblock == 2
                    }, index = [0])
                hdrdata = pd.concat([hdrdata, thisentry])
                cleandata.to_csv(os.path.join(taskdata_outdir, "task-" + thisid + "-sess%s.csv"%sess), index = False)
                thisbonusentry = pd.DataFrame({
                    "worker_id": worker_id,
                    "bonus": max(cleandata.totalEarnings) / 100,
                    "cb": cb
                    }, index = [0])
                bonusdata = pd.concat([bonusdata, thisbonusentry])
            # if the file is empty, add an entry to ./log/hdrdata_empty.csv
            else:
                print("Find a task file which is empty: " + thisid)
                logdf = pd.DataFrame({
                    "id": thisid,
                    "sess": sess,
                    "cb": cb,
                    "date": date
                    }, index = [0])
                if os.path.exists(os.path.join('log','hdrdata_empty_sess%d.csv'%sess)):
                    logdf.to_csv(os.path.join('log', 'hdrdata_empty_sess%d.csv'%sess), mode='a', header=False, index = False)
                else:
                    logdf.to_csv(os.path.join('log','hdrdata_empty_sess%d.csv'%sess), header=False, index = False)

        # exclude participants who didn't complete the task and who took the task for multiple times
        hdrdata = hdrdata.sort_values("id", ignore_index = True)
        hdrdata['duptask'] = hdrdata.id.duplicated(keep = False)
        hdrdata['included'] = np.logical_and(~hdrdata.duptask, hdrdata.cptask) # 

        # print summary messages
        print("Loop over %d files"%len(files))
        print("Loop over %d files that are not empty"%hdrdata.shape[0])
        print("Loop over %d participannts"%len(np.unique(hdrdata.id)))
        # print("%d files are empty"%len()) yeah I can make it better tomorrow
        print("%d participants completed the task"%len(np.unique(hdrdata.id[hdrdata.cptask])))
        tmp = hdrdata.id[np.logical_and(hdrdata.duptask, hdrdata.cptask)].unique()
        print("Among those participants, %d of them did the task for multiple times"%len(tmp))
        for x in tmp:
            print( x + " did the task for %d times and her/his files were excluded"% np.sum(hdrdata.id == x))

        # save hdrdata and bonus data
        # yeah I need to change bonus data as well..
        hdrdata.to_csv(hdrdata_out, index = False)
        for i in ['A', 'B', 'C', 'D']:
            bonusdata_out = os.path.join("data", "bonus_sess%d_%s.csv"%(sess, i))
            bonusdata.loc[bonusdata.cb == i].iloc[:, 0:2].to_csv(bonusdata_out, index=False, header = False)


    ############### main function #########
    parse_consent_data()
    if sess == 1:
        parse_selfreport_data()
    parse_task_data(sess)

    ######################## initial quality check ###############
    # I want to use these as global variables 
    selfreportfile = os.path.join("data", "selfreport_sess%d.csv"%sess)
    consentfile = os.path.join("data", "consent_sess%d.csv"%sess)
    hdrdatafile = os.path.join("data", "hdrdata_sess%d.csv"%sess)
    
    ################## read in data #############
    selfdata = pd.read_csv(selfreportfile)
    # code.interact(local = dict(locals(), **globals()))
    consentdata = pd.read_csv(consentfile)
    hdrdata = pd.read_csv(hdrdatafile)
    # code.interact(local = dict(globals(), **locals()))

    # print approved participants without task data 
    tmp = pd.merge(selfdata, consentdata, how = "left", left_on = "id", right_on = "id")
    # code.interact(local = dict(globals(), **locals()))
    print(tmp.loc[~np.isin(selfdata.id, hdrdata.id), ["id", "worker_id"]])

    # check whether any participant signed the consent for multiple times
    # Qualtrics should prevent it from happening though 
    # if any(consentdata.worker_id.duplicated()):
    #     consentdata[consentdata.worker_id.duplicated(keep = False), :]
    #     print("Some participants completed the consent for multiple times!")

    # check whether any participant completed questionaires for multiple times 
    # this is problematic 

    # check whether all participants who completed the task also completed questionaires
    # if not, then there might be something wrong with the payment 
    
    # this is ok


if __name__ == "__main__":
    print("Parse data files for SESS1")
    parsedata(1)
    # print("Parse data files for SESS2")
    # parsedata(2)
