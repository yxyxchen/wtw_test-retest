import numpy as np
import numpy.random as rd
from subFxs import expParas
import code
import pandas as pd
import re
import math
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


def QL_ind_initialize(ts, paras):
	Qquit = np.mean(expParas.optimRewardRates) / 0.15
	# code.interact(local = dict(locals(), **globals()))
	Qwaits = -0.1 / paras["tau"] * ts + paras['eta'] / paras["tau"] + Qquit

	return Qwaits, Qquit


def QL_slope_initialize(ts, paras):
	Qquit = np.mean(expParas.optimRewardRates) / 0.15
	# code.interact(local = dict(locals(), **globals()))
	Qwaits = -0.1 / paras["eta"] * ts + 1 + Qquit

	return Qwaits, Qquit


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
		if 'alphaU' in paras:
			Qwaits[update_filter] = Qwaits[update_filter] + paras['alphaU'] * (Gts[update_filter] - Qwaits[update_filter])
		else:	
			Qwaits[update_filter] = Qwaits[update_filter] + paras['alpha'] * paras['nu'] * (Gts[update_filter] - Qwaits[update_filter])

	# update Qquit
	Gt = np.exp(np.log(paras['gamma']) * (timeWaited - (-empirical_iti))) * (trialEarnings + Qquit)
	if trialEarnings > 0:
		Qquit = Qquit + paras['alpha'] * (Gt - Qquit)
	else:
		if 'alphaU' in paras:
			Qquit = Qquit + paras['alphaU'] * (Gt - Qquit)
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

	# initialize outputs 
	trialEarnings_ = np.zeros(nTrial)
	timeWaited_ = np.zeros(nTrial)
	sellTime_ = np.zeros(nTrial)
	Qwaits_ = np.zeros((len(ts), 10)) # record action values at 10 time points 
	Qquit_ = np.zeros(10)
	current_record_point = 0
	n_lp_trial = np.sum(condition_ == "LP")
	lp_record_unit = math.floor(n_lp_trial / 5)
	n_hp_trial = np.sum(condition_ == "HP")
	hp_record_unit = math.floor(n_hp_trial / 5)

	# trakc time within a trial
	for tIdx in range(nTrial):
		# reset elapsedTime at the beginning of each block
		if tIdx == 0 or blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			elapsedTime = 0

		if tIdx == 0 or blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
			if re.search('reset', modelname):
				if modelname == "QL2reset_ind":
					Qwaits, Qquit = QL_ind_initialize(ts, paras)
				elif re.search('slope', modelname):
					Qwaits, Qquit = QL_slope_initialize(ts, paras)
				elif modelname[:2] == 'QL':
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

		# record action values at key timepoints 
		if blockIdx_[tIdx] == 1:
			if tIdx % lp_record_unit == (lp_record_unit -1):
				Qwaits_[:, current_record_point] = Qwaits
				Qquit_[current_record_point] = Qquit
				current_record_point += 1	
		elif blockIdx_[tIdx] == 2:
			if (tIdx - n_lp_trial) % hp_record_unit == (hp_record_unit -1):
				Qwaits_[:, current_record_point] = Qwaits
				Qquit_[current_record_point] = Qquit
				current_record_point += 1	
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


	# make value df
	rv_ = Qwaits_ - np.tile(Qquit_,len(ts)).reshape(len(ts),10)
	value_df = pd.DataFrame({
		"time": np.tile(ts, 10),
		"record_time": np.tile(np.repeat((np.arange(5)+1) * 2, len(ts)), 2),
		"decision_value": rv_.transpose().reshape(-1) * paras["tau"],
		"relative_value": rv_.transpose().reshape(-1), 
		"condition" : np.repeat(np.repeat(("LP", "HP"), 5), len(ts))
		}) 

	outputs = pd.DataFrame({
		"totalTrialIdx": np.arange(nTrial),
		"blockIdx": blockIdx_,
		'condition': condition_,
		"scheduledDelay": scheduledDelay_,
		"timeWaited": timeWaited_,
		"trialEarnings": trialEarnings_,
		"sellTime": sellTime_
	})
	return outputs, Qwaits_, Qquit_, value_df

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

	# initialize outputs 
	trialEarnings_ = np.zeros(nTrial)
	timeWaited_ = np.zeros(nTrial)
	sellTime_ = np.zeros(nTrial)
	Qwaits_ = np.zeros((len(ts), 10)) # record action values at 10 time points 
	Qquit_ = np.zeros(10)
	current_record_point = 0
	n_lp_trial = np.sum(condition_ == "LP")
	lp_record_unit = math.floor(n_lp_trial / 5)
	n_hp_trial = np.sum(condition_ == "HP")
	hp_record_unit = math.floor(n_hp_trial / 5)
	
	# trakc time within a trial
	try:
		for tIdx in range(nTrial):
			# reset elapsedTime at the beginning of each block
			if tIdx == 0 or blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
				elapsedTime = 0
			if tIdx == 0:
				if re.search('slope', modelname):
					Qwaits, Qquit = QL_slope_initialize(ts, paras)
				elif modelname[:2] == 'QL':
					Qwaits, Qquit = QL_initialize(ts, paras)
				elif modelname[:2] == 'RL':
					Qwaits, Qquit, reward_rate = RL_initialize(ts, paras)
			elif blockIdx_[tIdx - 1] != blockIdx_[tIdx]:
				if re.search('reset', modelname):
					if re.search('slope', modelname):
						Qwaits, Qquit = QL_slope_initialize(ts, paras)
					elif modelname[:2] == 'QL':
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
			# record action values at key timepoints 
			if blockIdx_[tIdx] == 1:
				if tIdx % lp_record_unit == (lp_record_unit -1):
					Qwaits_[:, current_record_point] = Qwaits
					Qquit_[current_record_point] = Qquit
					current_record_point += 1	
			elif blockIdx_[tIdx] == 2:
				if (tIdx - n_lp_trial) % hp_record_unit == (hp_record_unit -1):
					Qwaits_[:, current_record_point] = Qwaits
					Qquit_[current_record_point] = Qquit
					current_record_point += 1	
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
	except:
		code.interact(local = dict(locals(), **globals()))

	# make value df
	rv_ = Qwaits_ - np.tile(Qquit_,len(ts)).reshape(len(ts),10)
	value_df = pd.DataFrame({
		"time": np.tile(ts, 10),
		"record_time": np.tile(np.repeat((np.arange(5)+1) * 2, len(ts)), 2),
		"decision_value": rv_.transpose().reshape(-1) * paras["tau"],
		"relative_value": rv_.transpose().reshape(-1), 
		"condition" : np.repeat(np.repeat(("LP", "HP"), 5), len(ts))
		})

	outputs = pd.DataFrame({
		"totalTrialIdx": np.arange(nTrial),
		"blockIdx": blockIdx_,
		'condition': condition_,
		"scheduledDelay": scheduledDelay_,
		"timeWaited": timeWaited_,
		"trialEarnings": trialEarnings_,
		"sellTime": sellTime_
	})
	return outputs, Qwaits_, Qquit_, value_df


















