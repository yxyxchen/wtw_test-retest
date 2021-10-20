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
import seaborn as sns
from scipy.stats import pearsonr
plt.style.use('classic')

# global
datadir = "data"
logdir = "analysis_log"



############################## load data functions ##################################
def loaddata(sess):
	"""load hdrdata and trialdata from given folders 
	"""
	hdrdata   = pd.read_csv(os.path.join(datadir, "hdrdata_sess%d.csv"%sess))
	print("Load %d files"%hdrdata.shape[0])
	print("Exclude %d participants who quit midway"%sum(hdrdata.quit_midway))
	print("\n")
	hdrdata = hdrdata.loc[np.logical_not(hdrdata.quit_midway)].reset_index(drop=True)

	nsub = hdrdata.shape[0]
	trialdata_  = {}
	for i in range(nsub):
		thisid = hdrdata.id.iloc[i]
		trialdata = pd.read_csv(os.path.join(datadir, "task-" + thisid + "-sess%d.csv"%sess))
		
		# add blockIdx
		trialdata['blockIdx'] = [1 if x == "LP" else 2 for x in trialdata['condition']]

		# add totalTrialIdx
		ntrial_firstblock = int(trialdata.trialIdx[trialdata.condition == "LP"].max())
		trialdata['totalTrialIdx'] = trialdata['trialIdx'] + np.equal(trialdata['condition'], "HP") * ntrial_firstblock

		# add accumSellTime
		trialdata['accumSellTime'] = trialdata['sellTime'] + (trialdata['blockIdx'] - 1) * expParas.blocksec 

		# fill in 
		trialdata_[(hdrdata.id.iloc[i], hdrdata.sess.iloc[i])] =  trialdata

	return hdrdata, trialdata_
# this can be very different across datasets

############################# individual-level analysis functions #####################
def parse_ind_selfreport(sess, row):
	""" process selfreport data for a single participant 

	Inputs:
		row: an entry of individual selfreport data
		plot_k: whether to plot diagnostic figures for the delayed reward discounting questionaire
		plot_upps: whether to plot diagnostic figures for UPPS
		plot_BIS: whether to plot diagnostic figures for BIS

	Outputs:
		out: a panda dataframe that contains parameters and scores for the given individual

	"""
	# the hyperbolic discounting parameter is calculated in R scripts under the MCQ folder

	# score BIS
	if sess == 1:
		BIS_subscores, Attentional, Motor, Nonplanning, BIS = analysisFxs.score_BIS(row[31:61], isplot)

		# score UPPS
		NU, PU, PM, PS, SS, UPPS  = analysisFxs.score_upps(row[61:120], isplot)

		# score PANAS
		if not np.isnan(row[120]): # the first batch doesn't have PANAS choices
			PAS, NAS = analysisFxs.score_PANAS(row[120:140], isplot)
		else:
			PAS = np.nan
			NAS = np.nan
		# assumeble outputs
		out = pd.DataFrame({
			"id": row.id,
			"duration": row.selfreport_duration,
			"NU": NU,
			"PU": PU,
			"PM": PM,
			"PS": PS,
			"SS": SS,
			"UPPS": UPPS,
			"Attentional": Attentional,
			"Motor": Motor,
			"Nonplanning": Nonplanning,
			"PAS": PAS,
			"NAS": NAS
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
		time, psurv, Time, Psurv, block_auc, block_std_wtw = analysisFxs.kmsc(blockdata, expParas.tMax, Time, False)
		if plot_KMSC:
			ax.plot(time, psurv, color = expParas.conditionColors[i], label = expParas.conditions[i])

		# Survival analysis into subblocks 
		sub_aucs = []
		sub_std_wtws = []
		for k in range(4):
			# code.interact(local = dict(locals(), **globals()))
			filter = np.logical_and(blockdata['sellTime'] >= k * expParas.blocksec / 4, blockdata['sellTime'] < (k + 1) * expParas.blocksec / 4)
			time, psurv, Time, Psurv, auc, std_wtw = analysisFxs.kmsc(blockdata.loc[filter, :], expParas.tMax, Time, False)
			sub_aucs.append(auc) 
			sub_std_wtws.append(std_wtw)

		# RT stats 
		ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se = analysisFxs.desc_RT(blockdata)
		
		# organize the output
		tmp = pd.DataFrame({"id": key[0], "sess": key[1], "block": i + 1, "auc": block_auc, "std_wtw": block_std_wtw,\
			"auc1": sub_aucs[0], "auc2": sub_aucs[1], "auc3": sub_aucs[2], "auc4": sub_aucs[3], \
			"std_wtw1": sub_std_wtws[0], "std_wtw2": sub_aucs[1], "std_wtw3": sub_std_wtws[2], "std_wtw4": sub_std_wtws[3], \
			"ready_RT_mean": ready_RT_mean,"ready_RT_se": ready_RT_se,"sell_RT_median": sell_RT_median,\
			"sell_RT_mean": sell_RT_mean, "sell_RT_se": sell_RT_se,\
			"condition": expParas.conditions[i]}, index = [i])
		stats.append(tmp) 

		objs['time_block'+str(i+1)] = time
		objs['psurv_block'+str(i+1)] = psurv
		objs['Time_block'+str(i+1)] = Time
		objs['Psurv_block'+str(i+1)] = Psurv
	stats = pd.concat(stats, ignore_index = True)
    	
	############ return  ############# y
	return stats, objs


#################################
def corr_analysis():
	# load summary statistics 
	stats_sess1 = pd.read_csv(os.path.join(logdir, "stats_sess1.csv"))
	stats_sess1['sess'] = 1
	stats_sess2 = pd.read_csv(os.path.join(logdir, "stats_sess2.csv"))
	stats_sess2['sess'] = 2
	stats = stats_sess1.merge(stats_sess2, on = ["id", "block", "condition"], suffixes = ['_sess1', '_sess2'])
	code.interact(local = dict(locals(), **globals()))

	# correlations with log_k
	k_df = pd.read_csv("k.csv")
	k_df['gm_logk'] = np.log(k_df['GMK'])
	k_df = k_df.loc[np.logical_and.reduce([k_df.SmlCons >= 0.8, k_df.LrgCons >= 0.8, k_df.MedCons >= 0.8])]
	fig,ax = plt.subplots(1, 1)
	ax.hist(x = np.log(k_df['GMK']))
	ax.set_xlabel("log k")

	# 
	plotdf = k_df.merge(stats_sess2, left_on = "SubjID", right_on = "id")
	g = sns.FacetGrid(
	    data=plotdf, col = 'condition'
	)
	g.map(sns.scatterplot, "gm_logk", "auc")

	# AUC timelines
	plotdf1 = pd.melt(stats_sess1.loc[stats_sess1.block == 1, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
	plotdf2 = pd.melt(stats_sess1.loc[stats_sess1.block == 2, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
	plotdf1['time'] = [int(x[-1]) for x in plotdf1.variable]
	plotdf2['time'] = [int(x[-1]) + 4 for x in plotdf2.variable]
	plotdf = pd.concat([plotdf1, plotdf2]).reset_index()
	g = sns.lineplot(
	    data=plotdf, x="time", y="value", err_style="bars", ci=95
	)
	g.set(xlim=(0, 9))
	g.set(ylabel='AUC (s)')

	plotdf1 = pd.melt(stats_sess2.loc[stats_sess2.block == 1, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
	plotdf2 = pd.melt(stats_sess2.loc[stats_sess2.block == 2, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
	plotdf1['time'] = [int(x[-1]) for x in plotdf1.variable]
	plotdf2['time'] = [int(x[-1]) + 4 for x in plotdf2.variable]
	plotdf = pd.concat([plotdf1, plotdf2]).reset_index()
	g = sns.lineplot(
	    data=plotdf, x="time", y="value", err_style="bars", ci=68
	)
	g.set(xlim=(0, 9))
	g.set(ylabel='AUC (s)')
	plt.savefig(os.path.join("..", "figures", "auc_timeline.png"))

	# correlations
	g = sns.FacetGrid(stats, col = "condition", hue = "condition")
	g.map(sns.scatterplot, "auc4_sess1", "auc4_sess2")
	g.axes.flat[0].plot(np.linspace(0, 12, 20), np.linspace(0, 12, 20), color = 'black')
	g.axes.flat[1].plot(np.linspace(0, 12, 20), np.linspace(0, 12, 20), color = 'black')
	LPcorr, _ = pearsonr(stats.auc4_sess1.loc[stats.block == 1], stats.auc4_sess2.loc[stats.block == 1])
	LPcorr
	HPcorr, _ = pearsonr(stats.auc4_sess1.loc[stats.block == 2], stats.auc4_sess2.loc[stats.block == 2])
	HPcorr

	# plot correlations 
	# fig,ax = plt.subplots(1, 2)
	# df_block1 = pd.merge(stats_sess1.loc[stats_sess1.block == 1], stats_sess2.loc[stats_sess1.block == 1], how = 'right', on = "id", suffixes = ['_s1', '_s2'])
	# ax[0].scatter(df_block1.auc_s1, df_block1.auc_s2)
	# ax[0].set_xlabel("LP AUC (s) SESS1")
	# ax[0].set_ylabel("LP AUC (s) SESS2")
	# m, b = np.polyfit(df_block1.auc_s1, df_block1.auc_s2, 1)
	# plt.plot(df.adapt_sess1, m*df.adapt_sess1 + b)

	# df_block2 = pd.merge(stats_sess1.loc[stats_sess1.block == 2], stats_sess2.loc[stats_sess1.block == 2], how = 'right', on = "id", suffixes = ['_s1', '_s2'])
	# ax[1].scatter(df_block2.auc_s1, df_block2.auc_s2)
	# ax[1].set_xlabel("HP AUC (s) SESS1")
	# ax[1].set_ylabel("HP AUC (s) SESS2")

	# auc adaption 
	fig,ax = plt.subplots(1, 1)
	adapt1 = stats_sess1['auc4'].loc[np.equal(stats_sess1.block, 1)].values - stats_sess1['auc1'].loc[stats_sess1.block == 2].values
	adapt2 = stats_sess2['auc4'].loc[stats_sess1.block == 1].values - stats_sess2['auc1'].loc[stats_sess1.block == 2].values
	df1 = pd.DataFrame({"id": stats_sess1.id[stats_sess1.block == 1].values, "adapt": adapt1})
	junk = {"id": stats_sess2.id[stats_sess2.block == 1].values, "adapt": adapt2}
	df2 = pd.DataFrame(junk)
	df = df1.merge(df2, on = "id", suffixes = ["_sess1", "_sess2"])
	ax.scatter(df.adapt_sess1, df.adapt_sess2)
	ax.set_xlabel("LP AUC - HP AUC, SESS1")
	ax.set_ylabel("LP AUC - HP AUC, SESS2")
	m, b = np.polyfit(df.adapt_sess1, df.adapt_sess2, 1)
	# plt.plot(df.adapt_sess1, m*df.adapt_sess1 + b)
	corr, _ = pearsonr(df.adapt_sess1, df.adapt_sess2)
	print('Pearsons correlation: %.3f' % corr)

		# with 
	plotdf = k_df.merge(df, left_on = "SubjID", right_on = "id")
	fig,ax = plt.subplots(1, 1)
	ax.scatter("gm_logk", "adapt_sess1", data = plotdf)
	ax.set_ylabel("LP AUC - HP AUC")
	ax.set_xlabel("log k")

	plotdf = k_df.merge(df, left_on = "SubjID", right_on = "id")
	fig,ax = plt.subplots(1, 1)
	ax.scatter("gm_logk", "adapt_sess2", data = plotdf)
	ax.set_ylabel("LP AUC - HP AUC")
	ax.set_xlabel("log k")

	################## Yeah I might want to look at these auc directly 
	fig,ax = plt.subplots(1, 2)
	ax[0].scatter(stats_sess1.auc.loc[stats_sess1.block == 1], stats_sess1.auc.loc[stats_sess1.block == 2])
	ax[0].set_xlabel("SESS1 LP AUC (s)")
	ax[0].set_ylabel("SESS1 HP AUC (s)")
	ax[1].scatter(stats_sess2.auc.loc[stats_sess2.block == 1], stats_sess2.auc.loc[stats_sess2.block == 2])
	ax[1].set_xlabel("SESS2 LP AUC (s)")
	ax[1].set_ylabel("SESS2 HP AUC (s)")
############################# group-level analysis functions #####################
def parse_group_selfreport(sess):
	# process selfreport data
	code.interact(local = dict(globals(), **locals()))
	# read the input file
	selfreportfile = os.path.join(datadir, 'selfreport_sess%d.csv'%sess)
	selfreport = pd.read_csv(selfreportfile)

	# score all other questionaires except MCQ
	selfdata = pd.DataFrame()
	for i, row in selfreport.iterrows():
		out  = parse_ind_selfreport(row, sess)
		selfdata = pd.concat([selfdata, out])
	selfdata.to_csv(os.path.join(logdir, "selfreport_sess%d.csv"%sess), index = False)


def plot_group_WTW(WTW_, TaskTime, filename):
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
	# code.interact(local = dict(locals(), **globals()))
	plt.savefig(os.path.join("..", "figures", filename))

def plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time, filename):
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
	plt.savefig(os.path.join("..", "figures", filename))

def group_quality_check(sess, plot_quality_check = False):
	# quality check the data
	hdrdatafile = os.path.join("data", 'hdrdata_sess%d.csv'%sess)
	trialdatafiles = glob.glob(os.path.join("data", "task*sess%d.csv"%sess))
	hdrdata, trialdata_ = loaddata(sess)
	logdir = "analysis_log"
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	# check sample size 
	nsub = hdrdata.shape[0]
	print("Check %d participants in SESS%d"%(nsub, sess))

	# check data quality 
	stats_ = []
	for idx, row in hdrdata.iterrows():
		this_trialdata = trialdata_[(row.id, row.sess)]
		stats = pd.DataFrame({
			"id": row.id,
			"batch": row.batch,
			"cb": row.cb,
			'ntrial_block1': this_trialdata.trialIdx[this_trialdata.blockIdx == 1].max(),
			'ntrial_block2': this_trialdata.trialIdx[this_trialdata.blockIdx == 2].max(),
			'total_timeWaited_block1': np.sum(this_trialdata.timeWaited[this_trialdata.blockIdx == 1]),
			'total_timeWaited_block2': np.sum(this_trialdata.timeWaited[this_trialdata.blockIdx == 2]),
		}, index = [0])
		ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se = analysisFxs.desc_RT(this_trialdata)
		stats['sell_RT_median'] = sell_RT_median
		stats['sell_RT_mean'] = sell_RT_mean
		stats_.append(stats)

	# plot tasktime and medium RT 
	statsdf = pd.concat(stats_)
	statsdf['tasktime_block1'] = statsdf.eval("ntrial_block1 * @expParas.iti + total_timeWaited_block1")
	statsdf['tasktime_block2'] = statsdf.eval("ntrial_block2 * @expParas.iti + total_timeWaited_block2")
	
	if plot_quality_check:
		_, ax_tasktime = plt.subplots(1,2)
		statsdf.hist('tasktime_block1', ax = ax_tasktime[0])
		ax_tasktime[0].set_xlabel("Block1 Tasktime (s)")
		statsdf.hist('tasktime_block2', ax = ax_tasktime[1])
		ax_tasktime[1].set_xlabel("Block 2 Tasktime (s)")
		_, ax_sellRT = plt.subplots(1,2)
		statsdf.hist("sell_RT_median", ax = ax_sellRT[0])
		statsdf.hist("sell_RT_mean", ax = ax_sellRT[1])

	# exclude participants with low quality data 
	excluded = statsdf.loc[np.logical_or.reduce(
		[statsdf.sell_RT_median > 1.2,
		statsdf.tasktime_block1 < 450,
		statsdf.tasktime_block2 < 450]
		), :].drop(['ntrial_block1', 'ntrial_block2', 'total_timeWaited_block1', 'total_timeWaited_block2'], axis=1)

	consentdata = pd.read_csv(os.path.join(datadir, "consent_sess%d.csv"%sess))
	# code.interact(local = dict(globals(), **locals()))
	excluded = pd.merge(excluded, right = consentdata[['id', 'worker_id']], how = "left", left_on = "id", right_on = "id")
	excluded.to_csv(os.path.join(logdir, "excluded_participants_sess%d.csv"%sess), index = False)

	# filter excluded data 
	hdrdata = hdrdata.loc[~np.isin(hdrdata.id, excluded.id), :]
	hdrdata.reset_index(drop=True, inplace  = True)
	trialdata_ = {x:y for x, y in trialdata_.items() if x[0] not in excluded.id}
	print("Exclude %d participants with low data quality!"%excluded.shape[0])
	print("\n")

	return hdrdata, trialdata_

def group_MF(sess, plot_each = False, plot_group = False):
	# 
	print("Performing Group-Level Model-Free Analysis for SESS%d"%sess)
	print("====================================================")

	# load data and exclude participants with low data quality
	hdrdata, trialdata_ = group_quality_check(sess, plot_quality_check = False)

	# check sample sizes 
	nsub = hdrdata.shape[0]
	print("Analyze %d valid participants in SESS%d"%(nsub, sess))
	print("\n")
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
		this_trialdata = trialdata_[(row.id, row.sess)]
		if plot_each:
			stats, objs  = ind_MF(this_trialdata, (row.id, row.sess),Time, TaskTime, plot_RT = True, plot_trial = True, plot_KMSC = False, plot_WTW = True)
			plt.show()
			input("Press Enter to continue...")
			plt.clf()
		else:
			stats, objs  = ind_MF(this_trialdata, (row.id, row.sess), Time, TaskTime)
			stats_.append(stats)

		if plot_group:
			Psurv_block1_[idx, :] = objs['Psurv_block1']
			Psurv_block2_[idx, :] = objs['Psurv_block2']
			WTW_[idx, :] = objs['WTW']

	# plot the group-level results
	if plot_group:
		plot_group_WTW(WTW_, TaskTime, 'sess%d_wtw.png'%sess)
		plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time, 'sess%d_sv.png'%sess)

	# save some data 
	# code.interact(local = dict(globals(), **locals()))
	stats_ = pd.concat(stats_)

	stats_.to_csv(os.path.join(logdir, "stats_sess%d.csv"%sess), index = False)

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
	parse_group_selfreport(2)
	# print(group_selfreport(os.path.join("data", "selfreport_sess1.csv")))
	# corr_analysis()
	# group_MF(1, plot_each = False, plot_group = True) 
	# group_MF(2, plot_each = False, plot_group = True) 
	# plt.show()

############

	
