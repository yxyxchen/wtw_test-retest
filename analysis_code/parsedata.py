########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import re
import itertools
import copy # pay attention to copy 
import code
import math
from datetime import datetime


############################# parse_original data files ############################
def parsedata(expname, sess):
    ############### input variables ###############
    taskdir = os.path.join("..", "task_code", "%s_sess%d"%(expname, sess))
    outdir = os.path.join("data", expname)
    if not os.path.exists(outdir):
        os.makedirs(outdir) 

    checked_data_dir = os.path.join(taskdir, "manual_check", "data_ok") # parse files that we have manually checked
    hdrdata_out = os.path.join(outdir, "hdrdata_sess%d.csv"%sess)

    consentfile = os.path.join(taskdir, "consent.csv")
    consent_out = os.path.join(outdir, "consent_sess%d.csv"%sess)
    consentfile_sess1 = os.path.join(outdir, "consent_sess1.csv")

    selfreportfile = os.path.join(taskdir, "selfreport.csv")
    selfreport_out = os.path.join(outdir, "selfreport_sess%d.csv"%sess)

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
    ##################### sub functions ###########################
    def parse_consent_data():
        """Parse consent data for the given session. 
        """
        #################### parse consent data ##############################
        # In session 1, we read in demographic and generate unidentifiable IDs. 
        # In session 2, we match these variables as recorded in session 1.
        consentdata = pd.read_csv(consentfile)
        if sess == 1:
            consentdata = consentdata[['workerId', "EndDate", 'demo1', 'demo2', 'demo3', 'demo4', 'demo5', 'demo7', 'assignmentId']]
            consentdata = consentdata.rename(columns={"workerId": "worker_id", 
                                                      "EndDate": "consent_date",
                                                      "demo1": "age",
                                                      "demo2": "gender",
                                                      "demo3": "education",
                                                      "demo4": "handness",
                                                      "demo5": "language",
                                                      "demo7": "race"})
            # for the active experiment, all data before 09/01/2021 are batch1, and the rest are batch2.
            if expname == 'active':
                consentdata['batch'] = [1 if datetime.strptime(x, '%m/%d/%y %H:%M') < datetime.strptime("09/1/21", "%m/%d/%y") else 2 for x in consentdata.consent_date] 
            else: 
                consentdata['batch'] = [1 if datetime.strptime(x, '%m/%d/%y %H:%M') < datetime.strptime("01/14/22", "%m/%d/%y") else 2 for x in consentdata.consent_date] 

            # delete duplicated entries. Sometimes a participant can sign a consent multiple times. 
            dup = consentdata.worker_id.duplicated(keep = 'last')
            if any(dup):
                print("Some participants signed the consent multiple times.")
                print("Please check whether they took multiple HITs and whether their answers were consistent.")
                print(consentdata.loc[consentdata.worker_id.duplicated(keep = False)])
                print("Deleting duplicated entries.....")
                consentdata = consentdata.loc[~dup]
                print("Duplicated entries deleted!")
            # add unidentifiable IDs
            consentdata.sort_values("consent_date", ignore_index = True)
            consentdata.insert(0, "id", ["s" + str(x + 1).zfill(4) for x in range(consentdata.shape[0])]) 
            # rm assignmentId 
            consentdata.pop("assignmentId")
        else:
            # read consent data
            consentdata = consentdata[['workerId', "EndDate"]]
            consentdata = consentdata.rename(columns={"workerId": "worker_id", 
                                                      "EndDate": "consent_date"})
            # delete duplicated entries. Sometimes a participant can sign a consent multiple times. 
            dup = consentdata.worker_id.duplicated(keep = 'last')
            if any(dup):
                print("Some participants signed the consent multiple times.")
                print("Please check whether they took multiple HITs and whether their answers were consistent.")
                print(consentdata.loc[consentdata.worker_id.duplicated(keep = False)])
                print("Deleting duplicated entries.....")
                consentdata = consentdata.loc[~dup]
                print("Duplicated entries deleted!")
            else:
                print('No participants signed the consent multiple times.')
            # join by consentdata_sess1
            consentdata_sess1 = pd.read_csv(consentfile_sess1)
            consentdata = consentdata.join(consentdata_sess1.drop("consent_date", axis = 1).set_index("worker_id"), on = "worker_id")
            tmp = consentdata.pop("id"); consentdata.insert(0, "id", tmp)
            consentdata.sort_values(by = "id", inplace = True)
        ################## save data #######################
        print("Save consent data for %d participants."%consentdata.shape[0])
        consentdata.to_csv(consent_out, index = False)

    def parse_selfreport_data():
        """Parse selfreport data for session 1 
        """
        ################ parse data ##########################
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

        # detect duplicated entries 
        dup = selfreportdata.id.duplicated(keep = 'first')
        if any(dup):
            print("Some participants fill the questionaires multiple times.")
            print(selfreportdata.loc[selfreportdata.id.duplicated(keep = False)])
            print("Deleting duplicated entries.....")
            selfreportdata = selfreportdata.loc[~dup]
            print("Duplicated entries deleted!")
        else:
            print('No participants fill the questionaires multiple times.')
        # save data
        print("Save selfreport data for %d participants."%selfreportdata.shape[0])
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
        if not workerId:
            workerId = "unknown" 
            if verbose: print("No worker ID is recorded!") 
            
        # check counterbalance group
        cb = np.unique(rawdata['cb'])[0]
        if not cb:
            cb = "unknown"
            if verbose: print("No counterbalance group is recorded!") 

        # find rows that record task data
        taskdata = rawdata[[bool(re.search("wtw-.*-block", x)) for x in rawdata.trial_type]]

        # check the number of saved task blocks 
        numblock = taskdata.shape[0]

        # initialize outputs 
        cleandata = []
        blockduration_ = np.full(2, 0)
        screenExitTime_ = np.full(2, np.nan)
        screenEnterTime_ = np.full(2, np.nan)

        # parse task data 
        def parse_task_variable(variable, data):
            """Parse a given task variable from rawdata
            """
            vals = [x for x in data[variable].split(",")]
            if variable != 'condition':
                vals = [float(x) for x in vals]
            if variable in ['scheduledDelay', 'RT', 'timeWaited', 'rewardedTime', 'sellTime', 'trialStartTime', 'trialReadyTime']:
                vals = [float(x) / 1000 for x in vals] # convert ms into s
            return vals

        # potential recorded variables. Notice, trialReadyTime is not recorded in passive-waiting tasks
        variables = ['trialIdx', 'condition', 'scheduledDelay', 'scheduledReward', 'rewardedTime', 'RT', 'timeWaited', 'trialEarnings', 'totalEarnings', 'sellTime', 'trialStartTime', 'trialReadyTime']

        # loop over blocks
        for i in range(numblock):
            taskdata_in_this_block = taskdata.iloc[i]
            cleandata_in_this_block = dict()
            num_valid_entry = 10000000000 # keep track of the number of valid entries. Notice that sometimes a trial ends midway 
            for variable in variables:
                if variable in taskdata_in_this_block:
                    # code.interact(local = dict(globals(), **locals()))
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
            blockduration_[i] = blockduration
            if expname == 'passive':
                tmp = taskdata_in_this_block['screenEnterTime']
                if not math.isnan(tmp):
                    screenEnterTime_[i] = tmp

                tmp = taskdata_in_this_block['screenExitTime']
                if not math.isnan(tmp):
                    screenExitTime_[i] = tmp

        # stack cleandata
        if numblock > 0:
            cleandata = pd.concat(cleandata)

        # save clean data 
        # cleandata.to_csv(os.path.join(cleandatadir, workerId+".csv"))
        if expname == 'active':
            return cleandata, workerId, cb, numblock, blockduration_
        else:
            return cleandata, workerId, cb, numblock, blockduration_, screenEnterTime_, screenExitTime_
    
    def parse_task_data(sess):
        """Loop over all taskdata files under ../task_code/passive_sess*/manual_check/data_ok. Convert all taskdata files that are not empty
        """
        consentdata = pd.read_csv(consentfile_sess1) # for batch and id information
        
        files = glob.glob(os.path.join(checked_data_dir, "*"))
        hdrdata = pd.DataFrame()
        bonusdata = pd.DataFrame()

        # print out what this function does
        print("Loop over all taskdata files. Generate a hdrdata file, a bonus file, and all processed taskdata files." )
        print("In the hdrdata file, record whether the participant quit midway." )
        print("For the bonus file, don't exclude IDs that don't exit in the selfreport file, which means those participant didn't get paid.")

        # Loop over all taskdata files under ../task_code/passive_sess*
        if os.path.exists(os.path.join('log','hdrdata_empty_sess%d.csv'%sess)):
            os.remove(os.path.join('log','hdrdata_empty_sess%d.csv'%sess))
        for file in files:
            if expname == 'active':
                cleandata, worker_id, cb, numblock, blockduration_ = parse_data(file, outdir)
            else:
                cleandata, worker_id, cb, numblock, blockduration_, screenEnterTime_, screenExitTime_ = parse_data(file, outdir)
            
            # mannually exclude some participants
            if worker_id == 'A35S5YV2XL971J':
                print("delete participant A35S5YV2XL971J, who didn't sign the consent yet got paid")
                continue 
            elif expname == 'active' and sess == 2 and worker_id in ['A235DXY5FJN0IW', 'A1F2APQMJSQVFK', 'AU849EHZNGV2Z', 'A1FVXS8IM5QYO8', 'AU34T9OMHN4Z4', 'A2NBBQ3DKW5MV3', 'AQ1PTX596CKUH',\
            "A2WWYVKGZZXBOB", "AP9WIQ4P78XLH", "A2HHKBW8HQWI66"]:
                # mannually exclude participants who I shouldn't invite for the second session
                print("remove participant " + worker_id + " who I shouldn't invite for the second session.")
                continue
            else:
                try:
                    thisid = consentdata['id'][np.where(consentdata['worker_id'] == worker_id)[0]].values[0]
                    thisbatch = consentdata['batch'][np.where(consentdata['worker_id'] == worker_id)[0]].values[0]
                except:
                    print("We can't find " + worker_id + 'in consent data.')
                    # I can't use consentdata from the second session since I don't have batch info and id info
                    code.interact(local = dict(globals(), **locals()))

            thisentry = pd.DataFrame({
                "id": thisid,
                "cb": cb,
                "sess": sess,
                "nblock": numblock,
                "batch": thisbatch,
                "quit_midway": numblock < 2,
                }, index = [0])
            # 
            thisentry["block1_duration"] = blockduration_[0] 
            thisentry["block2_duration"] = blockduration_[1] 
            if expname == 'passive':
                thisentry["block1_screenEnterTime"] = screenEnterTime_[0]
                thisentry["block2_screenEnterTime"] = screenEnterTime_[1]
                thisentry["block1_screenExitTime"] = screenExitTime_[0]
                thisentry["block2_screenExitTime"] = screenExitTime_[1]
            hdrdata = pd.concat([hdrdata, thisentry])
            if numblock > 0:
                cleandata.to_csv(os.path.join(outdir, "task-" + thisid + "-sess%s.csv"%sess), index = False)
            thisbonusentry = pd.DataFrame({
                "worker_id": worker_id,
                "bonus": max(cleandata.totalEarnings) / 100 if numblock > 0 else 0,
                "cb": cb,
                "batch": thisbatch
                }, index = [0])
            bonusdata = pd.concat([bonusdata, thisbonusentry])

        # save hdrdata and bonus data
        hdrdata = hdrdata.sort_values("id", ignore_index = True)
        hdrdata.to_csv(hdrdata_out, index = False)

        bonusdata.sort_values("cb", ignore_index = True)
        # only assign bonus to participants who completed the task, the assumption is that whoever completed self-report is approved...
        selfreportdata = pd.read_csv(selfreportfile)
        bonusdata = bonusdata[np.isin(bonusdata.worker_id, selfreportdata.workerId)] 
        # code.interact(local = dict(globals(), **locals()))
        for i in ['A', 'B', 'C', 'D']:
            for j in [1, 2]:
                bonusdata_out = os.path.join(outdir, "bonus_sess%d_batch%d_%s.csv"%(sess, j, i))
                bonusdata.loc[np.logical_and(bonusdata.cb == i, bonusdata.batch == j)].iloc[:, 0:2].to_csv(bonusdata_out, index=False, header = False)

        # print summary messages
        print("/////////////////////////////")
        print("Loop over %d participants"%len(files))
        print("hdrdata size: %d"%hdrdata.shape[0])
        print("%d participants completed this task"%np.sum(~hdrdata.quit_midway))
        print("bonusdata size: %d"%bonusdata.shape[0])
        print("%d participants completed this task and the questionaires."%bonusdata.shape[0])

    ############### main function #########
    print("Parse consent data")
    parse_consent_data()
    print("""
        -------------------------------------------------------
        -------------------------------------------------------
    """)   
    print("Parse selfreport data")
    
    parse_selfreport_data()
    print("Please compare selfreport data with CloudResearch records.")
    print("Each approved participant should have completed the selfreport questionaires.")
    print("""
        -------------------------------------------------------
        -------------------------------------------------------
    """)
    print("Parse task data")
    parse_task_data(sess)

    ######################## initial quality check ###############
    # if numbers don't match
    # I want to use these as global variables 
    # selfreportfile = os.path.join("data", "selfreport_sess%d.csv"%sess)
    # consentfile = os.path.join("data", "consent_sess%d.csv"%sess)
    # hdrdatafile = os.path.join("data", "hdrdata_sess%d.csv"%sess)
    
    # ################## read in data #############
    # selfdata = pd.read_csv(selfreportfile)
    # consentdata = pd.read_csv(consentfile)
    # hdrdata = pd.read_csv(hdrdatafile)

    # # print approved participants without task data 
    # tmp = pd.merge(selfdata, consentdata, how = "left", left_on = "id", right_on = "id")
    # alldata = pd.merge(hdrdata, consentdata, how = "left", left_on = "id", right_on = "id")
    # print(tmp.loc[~np.isin(selfdata.id, hdrdata.id), ["id", "worker_id"]])

    # 
    # df_all = pd.DataFrame()
    # for i in ["A", "B", "C", "D"]:
    #     df = pd.read_csv("~/Downloads/approve_batch2_%s.csv"%i)
    #     df_all = pd.concat([df_all, df])
    # df_all.loc[df_all.AmazonIdentifier.duplicated(keep = False),]
    # so a participant, "A35S5YV2XL971J" is probably a hacker... since he didn't fill the consent, he didn't have task data and he didn't fill the selfreport...
    # we deleted it from both self-report and task data in the second session

if __name__ == "__main__":

    expname = 'passive'
    print("Parse data files for SESS1")
    parsedata(expname, 1)
    # print("Parse data files for SESS2")
    # parsedata(expname, 2)

    # expname = 'passive'
    # print("Parse data files for SESS1")
    # parsedata(expname, 1)
    # print("Parse data files for SESS1")
    # parsedata(expname, 2)
