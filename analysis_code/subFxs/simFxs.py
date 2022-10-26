import numpy as np
import numpy.random as rd
from subFxs import expParas
import code
import pandas as pd
import re
###################### Helper Functions ###################
def RL_initialize(ts, paras):
	""" A helper function to initialize action values and reward rate estimate for R-Learning models

	inputs:
		ts: a vector of time steps
		paras: a dictionary of parameters
	"""
	reward_rate = np.mean(expParas.optimRewardRates)
	Qquit = 0
	Qwaits = -0.1 * ts + paras['eta'] + Qquit
	return Qwaits, Qquit, reward_rate

def QL_initialize(ts, paras):
	""" A helper function to initialize action values for Q-Learning models
	
	inputs:
		ts: a vector of time steps
		paras: a dictionary of parameters
	"""
	Qquit = np.mean(expParas.optimRewardRates) / 0.15
	# code.interact(local = dict(locals(), **globals()))
	Qwaits = -0.1 * ts + paras['eta'] + Qquit

	return Qwaits, Qquit


def softmax_dec(Qwait, Qquit, paras):
	""" A helper function to make decisions via the softmax function
	"""
	pWait =  1 / (1  + np.exp((Qquit - Qwait)* paras['tau']))
	if rd.uniform(0, 1) < pWait:
		action = 'wait'
	else:
		action = 'quit'
	return action


def QL2_learn(Qwaits, Qquit, ts, timeWaited, trialEarnings, paras, stepsize = 0.5, empirical_iti = expParas.iti):
	""" Learning rule for QL2
	"""
	# update Qwaits
	Gts = np.exp(np.log(paras['gamma']) * (timeWaited - ts)) * (trialEarnings + Qquit)
	# here is the problem maybe?
	# let me check later...
	update_filter = ts < timeWaited
	if trialEarnings > 0:
		Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * (Gts[update_filter] - Qwaits[update_filter])
	else:
		Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * paras['nu'] * (Gts[update_filter] - Qwaits[update_filter])

	# update Qquit
	Gt = np.exp(np.log(paras['gamma']) * (timeWaited - (-empirical_iti))) * (trialEarnings + Qquit)
	if trialEarnings > 0:
		Qquit = Qquit + paras['alpha'] * (Gt - Qquit)
	else:
		Qquit = Qquit + paras['alpha'] *  paras['nu'] * (Gt - Qquit)
	return Qwaits, Qquit

def QL1_learn(Qwaits, Qquit, ts, timeWaited, trialEarnings, paras, stepsize = 0.5, empirical_iti = expParas.iti):
	""" Learning rule for QL1 

	Inputs:
		Qwaits: value estimates of waiting at different time steps 
		Qquit: value estimates of quitting
		ts: a vector of time steps
		timeWaited: time spent on waiting in this trial 
		trialEarnings: payoff in this trial
		paras: a dict of parameters
		empirical_iti: actual average of iti durations
	"""
	# update Qwaits
	Gts = np.exp(np.log(paras['gamma']) * (timeWaited - ts)) * (trialEarnings + Qquit)
	update_filter = ts < timeWaited
	Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * (Gts[update_filter] - Qwaits[update_filter])

	# update Qquit
	Gt = np.exp(np.log(paras['gamma']) * (timeWaited - (-empirical_iti))) * (trialEarnings + Qquit)
	Qquit = Qquit + paras['alpha'] * (Gt - Qquit)
	return Qwaits, Qquit

def RL1_learn(Qwaits, Qquit, reward_rate, ts, timeWaited, trialEarnings, paras, stepsize = 0.5, empirical_iti = expParas.iti):
	""" Learning rule for RL1

	Inputs:
		Qwaits: value estimates of waiting at different time steps 
		Qquit: value estimates of quitting
		reward_rate: estimate of reward rate
		ts: a vector of time steps
		timeWaited: time spent on waiting in this trial 
		trialEarnings: payoff in this trial
		paras: a dict of parameters
		empirical_iti: actual average of iti durations
	"""
	# update Qwaits
	Gts = - reward_rate * (timeWaited - ts) + (trialEarnings + Qquit)
	update_filter = ts < timeWaited
	Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * (Gts[update_filter] - Qwaits[update_filter])


	# update Qquit
	Gt = - reward_rate * (timeWaited - (-empirical_iti)) + (trialEarnings + Qquit)
	delta = (Gt - Qquit)
	Qquit = Qquit + paras['alpha'] * delta

	# update reward rate 
	reward_rate = reward_rate + paras['beta'] * delta

	return Qwaits, Qquit, reward_rate

def RL2_learn(Qwaits, Qquit, reward_rate, ts, timeWaited, trialEarnings, paras, stepsize = 0.5, empirical_iti = expParas.iti):
	""" Learning rule for RL2

	Inputs:
		Qwaits: value estimates of waiting at different time steps 
		Qquit: value estimates of quitting
		reward_rate: estimate of reward rate
		ts: a vector of time steps
		timeWaited: time spent on waiting in this trial 
		trialEarnings: payoff in this trial
		paras: a dict of parameters
		empirical_iti: actual average of iti durations
	"""
	# update Qwaits
	Gts = - reward_rate * (timeWaited - ts) + (trialEarnings + Qquit)
	update_filter = ts < timeWaited
	if trialEarnings > 0:
		Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * (Gts[update_filter] - Qwaits[update_filter])
	else:
		Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * paras['nu'] * (Gts[update_filter] - Qwaits[update_filter])

	 # update Qquit
	Gt = - reward_rate * (timeWaited - (-empirical_iti)) + (trialEarnings + Qquit)
	delta = (Gt - Qquit)
	if trialEarnings > 0:
		Qquit = Qquit + paras['alpha'] * delta
	else:
		Qquit = Qquit + paras['alpha'] * paras['nu'] * delta

	 # update reward rate 
	reward_rate = reward_rate + paras['beta'] * delta

	return Qwaits, Qquit, reward_rate


def ind_fit_sim(modelname, paras, condition_, blockIdx_, scheduledDelay_, scheduledReward_, observed_trialEarnings_, observed_timeWaited_, stepsize, empirical_iti = expParas.iti):
	# check whether whether these inputs are not series. series might have wierd indices


	# to generate exactly the same data as the input data

	# outputs I need, should be the same as the readin data 
	# trialIdx | condition | scheduledDelay | rewardedTime | RT | timeWaited | trialEarnings | totalEarnings | sellTime | trialStartTime | trialReadyTime

	# code.interact(local = dict(locals(), **globals()))
	# number of trials
	nTrial = len(scheduledDelay_)

	# initialize value functions
	ts = np.arange(0, max(expParas.tMaxs), stepsize) 
	if modelname[:2] == 'QL':
		Qwaits, Qquit = QL_initialize(ts, paras)
	elif modelname[:2] == 'RL':
		Qwaits, Qquit, reward_rate = RL_initialize(ts, paras)

	# initialize outputs 
	trialEarnings_ = np.zeros(nTrial)
	timeWaited_ = np.zeros(nTrial)
	sellTime_ = np.zeros(nTrial)

	# trakc time within a trial
	for tIdx in range(nTrial):
		# reset elapsedTime at the beginning of each block
		if tIdx == 0 or blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			elapsedTime = 0

		if tIdx >0 and blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			if re.search('reset', modelname):
				if modelname[:2] == 'QL':
					Qwaits, Qquit = QL_initialize(ts, paras)
				elif modelname[:2] == 'RL':
					Qwaits, Qquit, reward_rate = RL_initialize(ts, paras)

		t = 0
		exit_status = False

		# save trialwise input variables
		scheduledDelay = scheduledDelay_[tIdx]
		scheduledReward = scheduledReward_[tIdx]

		# decide whether to wait or quit
		while t < max(expParas.tMaxs):
			# make a choice based on the softmax decision rule 
			action =  softmax_dec(Qwaits[ts == t], Qquit, paras)
			
			# whether to exit this trial and save data 
			if (action == 'wait' and t + stepsize >= scheduledDelay):
				trialEarnings = scheduledReward
				timeWaited = scheduledDelay
				exit_status = True

			if action == 'quit':
				trialEarnings = 0
				timeWaited = t + 0.5 * stepsize
				exit_status = True

			# save trial-wise outputs if exists
			if exit_status:
				sellTime = elapsedTime + timeWaited 
				trialEarnings_[tIdx] = trialEarnings
				timeWaited_[tIdx] = timeWaited
				sellTime_[tIdx] = sellTime
				break 

			t += stepsize

		# update elapsedTime
		elapsedTime = elapsedTime + timeWaited + empirical_iti

		# update value functions
		if modelname[:3] == 'QL1':
			# code.interact(local = dict(locals(), **globals()))
			Qwaits, Qquit = QL1_learn(Qwaits, Qquit, ts, observed_timeWaited_[tIdx], observed_trialEarnings_[tIdx], paras, empirical_iti = expParas.iti)
		elif modelname[:3] == "QL2":
			Qwaits, Qquit = QL2_learn(Qwaits, Qquit, ts, observed_timeWaited_[tIdx], observed_trialEarnings_[tIdx], paras, empirical_iti = expParas.iti)
		elif modelname[:3] == 'RL1':
			Qwaits, Qquit, reward_rate = RL1_learn(Qwaits, Qquit, reward_rate, ts, observed_timeWaited_[tIdx], observed_trialEarnings_[tIdx], paras, empirical_iti = expParas.iti)
		elif modelname[:3] == 'RL2':
			Qwaits, Qquit, reward_rate = RL2_learn(Qwaits, Qquit, reward_rate, ts, observed_timeWaited_[tIdx], observed_trialEarnings_[tIdx], paras, empirical_iti = expParas.iti)
		
	# # find the duration of each block
	# blocks = np.unique(blockIdx_)
	# blockdurations = [np.max(sellTime_[blockIdx_ == i]) for i in blocks]
	# accumSellTime_ = sellTime_ + 

	outputs = pd.DataFrame({
		"totalTrialIdx": np.arange(nTrial),
		"blockIdx": blockIdx_,
		'condition': condition_,
		"scheduledDelay": scheduledDelay_,
		"timeWaited": timeWaited_,
		"trialEarnings": trialEarnings_,
		"sellTime": sellTime_
	})
	return outputs

def ind_sim(modelname, paras, condition_, blockIdx_, scheduledDelay_, scheduledReward_, stepsize, empirical_iti = expParas.iti):
	# check whether whether these inputs are not series. series might have wierd indices


	# to generate exactly the same data as the input data

	# outputs I need, should be the same as the readin data 
	# trialIdx | condition | scheduledDelay | rewardedTime | RT | timeWaited | trialEarnings | totalEarnings | sellTime | trialStartTime | trialReadyTime

	# code.interact(local = dict(locals(), **globals()))
	# number of trials
	nTrial = len(scheduledDelay_)

	# initialize value functions
	ts = np.arange(0, max(expParas.tMaxs), stepsize) 
	if modelname[:2] == 'QL':
		Qwaits, Qquit = QL_initialize(ts, paras)
	elif modelname[:2] == 'RL':
		Qwaits, Qquit, reward_rate = RL_initialize(ts, paras)

	# initialize outputs 
	trialEarnings_ = np.zeros(nTrial)
	timeWaited_ = np.zeros(nTrial)
	sellTime_ = np.zeros(nTrial)

	# trakc time within a trial
	for tIdx in range(nTrial):
		# reset elapsedTime at the beginning of each block
		if tIdx == 0 or blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			elapsedTime = 0

		if blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			if re.search('reset', modelname):
				if modelname[:2] == 'QL':
					Qwaits, Qquit = QL_initialize(ts, paras)
				elif modelname[:2] == 'RL':
					Qwaits, Qquit, reward_rate = RL_initialize(ts, paras)

		t = 0
		exit_status = False

		# save trialwise input variables
		scheduledDelay = scheduledDelay_[tIdx]
		scheduledReward = scheduledReward_[tIdx]

		# decide whether to wait or quit
		while t < max(expParas.tMaxs):
			# make a choice based on the softmax decision rule 
			action =  softmax_dec(Qwaits[ts == t], Qquit, paras)
			
			# whether to exit this trial and save data 
			if (action == 'wait' and t + stepsize >= scheduledDelay):
				trialEarnings = scheduledReward
				timeWaited = scheduledDelay
				exit_status = True

			if action == 'quit':
				trialEarnings = 0
				timeWaited = t + 0.5 * stepsize
				exit_status = True

			# save trial-wise outputs if exists
			if exit_status:
				sellTime = elapsedTime + timeWaited 
				trialEarnings_[tIdx] = trialEarnings
				timeWaited_[tIdx] = timeWaited
				sellTime_[tIdx] = sellTime
				break 

			t += stepsize

		# update elapsedTime
		elapsedTime = elapsedTime + timeWaited + empirical_iti

		# update value functions
		if modelname[:3] == 'QL1':
			# code.interact(local = dict(locals(), **globals()))
			Qwaits, Qquit = QL1_learn(Qwaits, Qquit, ts, timeWaited, trialEarnings, paras, empirical_iti = expParas.iti)
		elif modelname[:3] == "QL2":
			Qwaits, Qquit = QL2_learn(Qwaits, Qquit, ts, timeWaited, trialEarnings, paras, empirical_iti = expParas.iti)
		elif modelname[:3] == 'RL1':
			Qwaits, Qquit, reward_rate = RL1_learn(Qwaits, Qquit, reward_rate, ts, timeWaited, trialEarnings, paras, empirical_iti = expParas.iti)
		elif modelname[:3] == 'RL2':
			Qwaits, Qquit, reward_rate = RL2_learn(Qwaits, Qquit, reward_rate, ts, timeWaited, trialEarnings, paras, empirical_iti = expParas.iti)
		
	# # find the duration of each block
	# blocks = np.unique(blockIdx_)
	# blockdurations = [np.max(sellTime_[blockIdx_ == i]) for i in blocks]
	# accumSellTime_ = sellTime_ + 

	outputs = pd.DataFrame({
		"totalTrialIdx": np.arange(nTrial),
		"blockIdx": blockIdx_,
		'condition': condition_,
		"scheduledDelay": scheduledDelay_,
		"timeWaited": timeWaited_,
		"trialEarnings": trialEarnings_,
		"sellTime": sellTime_
	})
	return outputs


















