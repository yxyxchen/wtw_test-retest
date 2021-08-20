########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import re
import itertools
import copy # pay attention to copy 
import code

############################# parse_original data files ############################
def parsedata(sess):
	############### variables in the enclosing scope ###############
	taskdir  = os.path.join("..", "task_sess%d"%sess)
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
	bonusdata_out = os.path.join("data", "bonus_sess%d.csv"%sess)
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
			# add unidentifiable IDs
			consentdata.sort_values("consent_date", ignore_index = True)
			consentdata.insert(0, "id", ["s" + str(x + 1).zfill(4) for x in consentdata.index]) 
		else:
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
		selfreportdata = pd.merge(selfreportdata, consentdata[['worker_id', 'id']], how = "left", left_on = ["worker_id"], right_on = ["worker_id"])
		selfreportdata.pop("worker_id")
		tmp = selfreportdata.pop("id"); selfreportdata.insert(0, "id", tmp) # 
		selfreportdata.sort_values('id', inplace  = True) # sort by id 

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
	        blockdurations.append(blockdurations)
	    if numblock > 0:
	        cleandata = pd.concat(cleandata)

	    # save clean data 
	    # cleandata.to_csv(os.path.join(cleandatadir, workerId+".csv"))
	    return cleandata, workerId, cb, numblock, blockdurations
	def parse_task_data():
		"""Parse task data
		"""
		########### parse data ###################
		consentdata = pd.read_csv(consentfile_sess1)
		files = glob.glob(os.path.join(taskdata_dir, "*"))
		hdrdata = {
		    "id": [],
		    "cb": [],
		    "sess": [],
		    "date": [],
		}
		bonusdata = {
		    "worker_id": [],
		    "bonus": []
		}

		date_parser = re.compile(os.path.join(taskdata_dir, "wtw_.*_PARTICIPANT_SESSION_(.*)_.*.csv"))
		for file in files:
		    cleandata, worker_id, cb, numblock, blockdurations = parse_data(file, taskdata_outdir)
		    thisid = consentdata['id'][np.where(consentdata['worker_id'] == worker_id)[0]].values[0]
		    hdrdata['id'].append(thisid)
		    hdrdata['cb'].append(cb)
		    hdrdata['sess'].append(1)
		    hdrdata['date'].append(date_parser.findall(file)[0])
		    
		    bonusdata['worker_id'].append(worker_id)
		    bonusdata['bonus'].append(max(cleandata.totalEarnings) / 100)
		    cleandata.to_csv(os.path.join(taskdata_outdir, "task-" + thisid + "-sess%s.csv"%sess), index = False)

		# save hdrdata and bonus data
		hdrdata = pd.DataFrame(hdrdata).sort_values("id", ignore_index = True)
		hdrdata.to_csv(hdrdata_out, index = False)

		bonusdata = pd.DataFrame(bonusdata)
		bonusdata.to_csv(bonusdata_out, index=False, header = False)

	############### main function #########
	parse_consent_data()
	if sess == 1:
		parse_selfreport_data()
	parse_task_data()

if __name__ == "__main__":
	print("Parse data files for SESS1")
	parsedata(1)
	print("Parse data files for SESS2")
	parsedata(2)
