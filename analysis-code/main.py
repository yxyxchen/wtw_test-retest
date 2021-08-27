########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import importlib
import re
import matplotlib.pyplot as plt
import itertools
import copy # pay attention to copy 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sksurv.nonparametric import kaplan_meier_estimator as km
from scipy.interpolate import interp1d
import code
# my customized modules
import subFxs
from subFxs import analysisFxs
from subFxs import expParas
plt.style.use('classic')


############################## load data functions ##################################
def loaddata(hdrdatafile, trialdatafiles):
	"""load hdrdata and trialdata from given folders 
	"""
	hdrdata   = pd.read_csv(hdrdatafile)
	nsub = hdrdata.included.sum()
	trialdata_  = {}
	# code.interact(local = dict(locals(), **globals()))
	for i in range(nsub):
		trialdata = pd.read_csv(trialdatafiles[i])
		
		# add blockIdx
		trialdata['blockIdx'] = [1 if x == "LP" else 2 for x in trialdata['condition']]

		# add totalTrialIdx
		ntrial_firstblock = int(trialdata.trialIdx[trialdata.condition == "LP"].max())
		trialdata['totalTrialIdx'] = trialdata['trialIdx'] + np.equal(trialdata['condition'], "HP") * ntrial_firstblock

		# add accumSellTime
		trialdata['accumSellTime'] = trialdata['sellTime'] + (trialdata['blockIdx'] - 1) * expParas.blocksec 

		# fill in 
		trialdata_[(hdrdata.id[i], hdrdata.sess[i])] =  trialdata

	return hdrdata, trialdata_

# this can be very different across datasets

############################# individual-level analysis functions #####################
def parse_ind_selfreport(row, plot_k = False, plot_upps = False, plot_BIS = False):
	""" process selfreport data for a single participant 

	Inputs:
		row: an entry of individual selfreport data
		plot_k: whether to plot diagnostic figures for the delayed reward discounting questionaire
		plot_upps: whether to plot diagnostic figures for UPPS
		plot_BIS: whether to plot diagnostic figures for BIS

	Outputs:
		out: a panda dataframe that contains parameters and scores for the given individual

	"""
	# estimate the hyperbolic discounting parameter
	try:
		k,logk,se_logk  = analysisFxs.calc_k(row[4:27+4], False)
	except:
		k = None,
		logk = None,
		se_logk = None

	# score UPPS
	NU, PU, PM, PS, SS, UPPS  = analysisFxs.score_upps(row[61:120], False)

	# score BIS
	BIS_subscores, Attentional, Motor, Nonplanning, BIS = analysisFxs.score_BIS(row[31:61])

	# assumeble outputs
	out = pd.DataFrame({
		"k":k,
		"logk": k,
		"se_logk": se_logk,
		"NU": NU,
		"PU": PU,
		"PM": PM,
		"PS": PS,
		"SS": SS,
		"UPPS": UPPS,
		"Attentional": Attentional,
		"Motor": Motor,
		"Nonplanning": Nonplanning
		}, index = [row.id])

	return out

def ind_MF(trialdata, key, Time, TaskTime, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False):
	"""Conduct model-free (MF) analysis for a single participant 
	
	Inputs:
		trialdata: a pd dataframe that contains task data
		key: in the format of (id, sessIdx). For example, ("s0001", 1)
		plot_RT: whether to plot 
		plot_trial:
	"""
	# initialize the output
	stats = [] # for scalar outputs
	objs = {} # for others

	# RT visual check
	if plot_RT:
		analysisFxs.rtplot_multiblock(trialdata)

	# trial-by-trial behavior visual check
	if plot_trial:
		analysisFxs.trialplot_multiblock(trialdata)

	# WTW timecourse
	wtw, WTW, TaskTime = analysisFxs.wtwTS(trialdata, expParas.tMax, TaskTime, plot_WTW)
	objs['wtw'] = wtw
	objs['TaskTime'] = TaskTime
	objs['WTW'] = WTW

	################## calculate summary stats for each block ###############
	################## this part of code  can be modified for different experiments ##########
	# initialize the figure 
	if plot_KMSC:
		fig, ax = plt.subplots()
		ax.set_xlim([0, expParas.tMax])
		ax.set_ylim([-0.1, 1.1])
		ax.set_xlabel("Elapsed time (s)")
		ax.set_ylabel("Survival rate")
		
	for i in range(expParas.nBlock):
		blockdata = trialdata[trialdata.blockIdx == i + 1]
		condition = expParas.conditions[i]

		# Survival analysis
		time, psurv, Time, Psurv, auc, std_wtw = analysisFxs.kmsc(blockdata, expParas.tMax, Time, False)
		if plot_KMSC:
			ax.plot(time, psurv, color = expParas.conditionColors[i], label = expParas.conditions[i])

		# RT stats 
		ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se = analysisFxs.desc_RT(blockdata)
		
		#code.interact(local = dict(globals(), **locals()))

		# organize the output
		tmp = pd.DataFrame({"id": key[0], "sess": key[1], "block": i + 1, "auc": auc, "std_wtw": std_wtw,\
			"ready_RT_mean": ready_RT_mean,"ready_RT_se": ready_RT_se,"sell_RT_median": sell_RT_median,\
			"sell_RT_mean": sell_RT_mean, "sell_RT_se": sell_RT_se,\
			"condition": expParas.conditions[i]}, index = [i])
		stats.append(tmp)

		objs['time_block'+str(i+1)] = time
		objs['psurv_block'+str(i+1)] = psurv
		objs['Time_block'+str(i+1)] = Time
		objs['Psurv_block'+str(i+1)] = Psurv
	stats = pd.concat(stats, ignore_index = True)

	############ return  #############
	return stats, objs

############################# group-level analysis functions #####################
def parse_group_selfreport(selfreportfile):
	# process selfreport data
	selfreport = pd.read_csv(selfreportfile)
	selfdata = pd.DataFrame()
	for i, row in selfreport.iterrows():
		try:
			out  = parse_ind_selfreport(row, plot_k = False, plot_upps = False, plot_BIS = False)
			selfdata = pd.concat([selfdata, out])
		except:
			code.interact(local = dict(globals(), **locals()))
	return selfdata 

def plot_group_WTW(WTW_, TaskTime):
	"""Plot group-level WTW timecourse 
	"""
	fig, ax = plt.subplots()
	df = pd.DataFrame({
			"mu": np.apply_along_axis(np.mean, 0, WTW_),
			"se": np.apply_along_axis(analysisFxs.calc_se, 0, WTW_),
			"TaskTime": TaskTime
		})
	# code.interact(local = dict(globals(), **locals()))
	df = df.assign(ymin = df.mu - df.se, ymax = df.mu + df.se)
	df.plot("TaskTime", "mu", color = "black", ax = ax)
	ax.fill_between(df.TaskTime, df.ymin, df.ymax, facecolor='grey', edgecolor = "none",alpha = 0.4, interpolate=True)
	ax.set_xlabel("")
	ax.set_ylabel("WTW (s)")
	ax.set_xlabel("Task time (s)")
	ax.vlines(expParas.blocksec, 0, expParas.tMax, color = "red", linestyles = "dotted") # I might want to change it later

def plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time):
	""" Plot group-level survival curves 
	"""
	fig, ax = plt.subplots()
	df1 = pd.DataFrame({
			"mu": np.apply_along_axis(np.mean, 0, Psurv_block1_),
			"se": np.apply_along_axis(analysisFxs.calc_se, 0, Psurv_block1_),
			"Time": Time
		})
	df1 = df1.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)
	df2 = pd.DataFrame({
			"mu": np.apply_along_axis(np.mean, 0, Psurv_block2_),
			"se": np.apply_along_axis(analysisFxs.calc_se, 0, Psurv_block2_),
			"Time": Time
		})
	df2 = df2.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)

	df1.plot("Time", "mu", color = expParas.conditionColors[0], ax = ax)
	ax.fill_between(df1.Time, df1.ymin, df1.ymax, facecolor= expParas.conditionColors[0], edgecolor = "none",alpha = 0.4, interpolate=True)
	df2.plot("Time", "mu", color = expParas.conditionColors[1], ax = ax)
	ax.fill_between(df2.Time, df2.ymin, df2.ymax, facecolor= expParas.conditionColors[1], edgecolor = "none",alpha = 0.4, interpolate=True)
	ax.set_xlabel("Elapsed time (s)")
	ax.set_ylabel("Survival rate (%)")
	ax.set_ylim((0, 1))
	ax.set_xlim((0, expParas.tMax))

def group_MF(sess, plot_each = False, plot_group = False):
	hdrdatafile = os.path.join("data", 'hdrdata_sess%d.csv'%sess)
	trialdatafiles = glob.glob(os.path.join("data", "task*sess%d.csv"%sess))
	hdrdata, trialdata_ = loaddata(hdrdatafile, trialdatafiles)

	# check sample size 
	nsub = hdrdata.shape[0]
	print("Analyze %d participants in SESS%d"%(nsub, sess))

	# initialize outputs 
	Time = expParas.Time
	TaskTime = expParas.TaskTime
	stats_ = []
	if plot_group:
		Psurv_block1_ = np.empty([nsub, len(Time)])
		Psurv_block2_ = np.empty([nsub, len(Time)])
		WTW_ = np.empty([nsub, len(TaskTime)])

	# run MF for each participant 
	for idx, row in hdrdata.iterrows():
		if plot_each:
			stats, objs  = ind_MF(trialdata_[(row.id, row.sess)], (row.id, row.sess),Time, TaskTime, plot_RT = True, plot_trial = True, plot_KMSC = True, plot_WTW = True)
			plt.show()
			input("Press Enter to continue...")
			plt.clf()
		else:
			stats, objs  = ind_MF(trialdata_[(row.id, row.sess)], (row.id, row.sess), Time, TaskTime)

		stats_.append(stats)
		if plot_group:
			Psurv_block1_[idx, :] = objs['Psurv_block1']
			Psurv_block2_[idx, :] = objs['Psurv_block2']
			WTW_[idx, :] = objs['WTW']

	# code.interact(local = dict(globals(), **locals()))
	statsdf = pd.concat(stats_)

	if plot_group:
		plot_group_WTW(WTW_, TaskTime)
		plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time)

		

# def test_retest(:
# 	hdrdatafile = os.path.join("data", 'hdrdata_sess%d.csv'%sess)
# 	trialdatafiles = glob.glob(os.path.join("data", "task*sess%d.csv"%sess))

# 	hdrdata, trialdata_ = loaddata(hdrdatafile, trialdatafiles)
# 	for idx, row in hdrdata.iterrows():
# 		stats, objs  = ind_MF(trialdata_[(row.id, row.sess)], (row.id, row.sess), plot_RT = True, plot_trial = True, plot_KMSC = True, plot_WTW = True)
# 		plt.show()
# 		input("Press Enter to continue...")
# 		plt.clf()

############################## main ##########################
if __name__ == "__main__":
	# print(group_selfreport(os.path.join("data", "selfreport_sess1.csv")))
	
	group_MF(1, plot_each = True, plot_group = False) 

	# print(hdrdata)
	# print(trialdata[(hdrdata.id[0], hdrdata.sess[0])])
	# print(trialdata_[('s0001', 1)].keys())
	
	# 
	# 	plot_KMSC = True, plot_WTW = True)
	# code.interact(local=dict(globals(), **locals()))
	# Time = np.linspace(0, 600 * 2, 600*2)
	# wtw, WTW, Time = analysisFxs.wtwTS(trialdata_[('s0007', 1)], expParas.tMax, Time, True)
	plt.show()

############

	
