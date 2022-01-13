class WTW:

	def __init__(self, data, iti):
		self.data = data
		self.iti = iti

	def sanity_check(self):
		import numpy as np
		import pandas as pd

		valid = True

		######### check NaN values ##############
		# RT is only recorded when participants sell a matured token
		if not all(np.isnan(self.data.RT[self.data.trialEarnings == 0])):
			print("RT invalid")
			valid = False

		# rewardedTime is only recorded when participants sell a matured token
		if not all(np.isnan(self.data.rewardedTime[self.data.trialEarnings == 0])): 
			print("rewardedTime invalid")
			valid = False

		######### check calculated timing variables ##############
		# timeWaited = sellTime - trialStartTime
		if any(abs(self.data.sellTime - self.data.timeWaited - self.data.trialStartTime) > 1e-5):
			print("timeWaited != sellTime - trialStartTime")
			valid = False

		# RT = sellTime - rewardedTime
		rewardedTrials = self.data[self.data.trialEarnings != 0]
		if any(abs(rewardedTrials.sellTime - rewardedTrials.RT - rewardedTrials.rewardedTime) > 1e-5):
			print("RT != sellTime - rewardedTime")
			valid = False

		######### check sequential timing variables ################
		# next trialStartTime/trialReadyTime = sellTime + iti
		if "trialReadyTime" in self.data:
			if any(abs(self.data.trialReadyTime[1:].values - self.data.sellTime[:-1].values - self.iti) > 0.1):
				print('next trialReadyTime != sellTime + iti')
				valid = False
		else:
			if any(abs(self.data.trialStartTime[1:].values - self.data.sellTime[:-1].values - self.iti) > 0.1):
				print('next trialStartTime != sellTime + iti')
				valid = False

		if valid:
			print("The data look fine!!")







