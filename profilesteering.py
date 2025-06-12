   # Copyright 2023 University of Twente

   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at

       # http://www.apache.org/licenses/LICENSE-2.0

   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.
   
import operator
import numpy as np
import random

class ProfileSteering():
	def __init__(self, devices):
		self.devices = devices
		self.p = [] # p in the PS paper
		self.x = [] # x in the PS paper
	
	def init(self, p):
		# Set the desired profile and reset xrange
		self.p = list(p)
		self.x = [0] * len(p)
	
		# Ask all devices to propose an initial planning
		for device in self.devices:
			r = device.init(p) 	# request device to create a planning
			self.x = list(map(operator.add, self.x, r))	# Perform the summation by adding the overall profile to the planning
		
		#print("Initial planning", self.x)
		return self.x
	
	def rerun(self,x):
		# ask all devices to return to initial planning
		for device in self.devices:
			device.profile = device.initial_profile
			device.burden = 0	#reset their burden
		self.x = x		# reset aggregate profile
		return 0
	
	def gini(values):												# function to return Gini coefficient of an array of values, based on A. Sen as half of the relative mean absolute difference
		n = values.size												# number of values
		if n == 0:
			return 0												# return 0 if an empty array is handed
		mean = values.mean()										# mean of all values
		diff_sum = np.abs(values[:, None] - values).sum()			# double summation of |xi - xj| for all n x n combinations
		denominator = 2 * n**2 * mean								# denominator for dividing the double summation over 2 and normalizing over calculations (n^2) and mean 
		return diff_sum / denominator if denominator != 0 else 0	# return calculation, tackle divisions by 0
		
	def iterative(self, e_min, max_iters, tau):
		# Initialize trackers for each iteration
		tr_improvement = np.array([])
		tr_objective = np.array([])
		tr_gini = np.array([])

		# Iterative Loop
		for i in range(0, max_iters):	# Note we deviate here slightly by also defining a maximum number of iterations
			# Init
			best_improvement = 0
			best_device = None
			lowest_score = np.inf		# initialize score as infinity to ensure the first device is an improvement
			
			# difference profile
			d = list(map(operator.sub, self.x, self.p)) # d = x - p
			
			# Vanilla PS mechanism:
			# request a new candidate profile from each device
			#for device in self.devices:
			#	improvement = device.plan(d)
			#	if improvement > best_improvement:
			#		best_improvement = improvement
			#		best_device = device

			# Fairer PS mechanism:
			# request a new candidate profile from each device
			for device in self.devices:
				improvement, add_burden = device.plan(d)
			# score all devices that have e_m > 0 (essentially exclude baseloads)
			contributing_devices = [cd for cd in self.devices if cd.candidate_improvement > 0]	# filter out contributing devices, calculate values required for normalization
			random.shuffle(contributing_devices)		# shuffle device order, for fair picking in case of ties
			sum_B = sum(cd.candidate_burden for cd in contributing_devices)
			sum_E = sum(cd.candidate_improvement for cd in contributing_devices)
			M = len(contributing_devices)	
			for device in contributing_devices:
				score_pt1 = device.candidate_burden / ((1/M)*sum_B)
				score_pt2 = device.candidate_improvement / ((1/M)*sum_E)
				score = tau*score_pt1-(1-tau)*score_pt2
				# track lowest scoring device
				if (score < lowest_score):
						lowest_score = score			
						best_device = device
						best_improvement = best_device.candidate_improvement
					
			# Now set the winner (best scoring device) and update its planning+burden and improvement tracker
			if best_device is not None:
				diff = best_device.accept()
				self.x = list(map(operator.add, self.x, diff))
				tr_improvement = np.append(tr_improvement,best_improvement)

			# Update objective + fairness trackers
			new_objective = np.linalg.norm(np.array(self.x)-np.array(self.p))	# update objective score 2-norm x-p
			tr_objective = np.append(tr_objective,new_objective)

			burdens = [device.burden for device in self.devices if device.type != "BL"] #Exclude BL as they are passive devices
			new_gini = ProfileSteering.gini(np.array(burdens))	# update Gini coefficient
			tr_gini = np.append(tr_gini,new_gini)
			#print("Iteration", i, "-- Winning device is is ", best_device.type, best_device, " with score ",  lowest_score, " -- Improvement is ", best_device.candidate_improvement, " -- This device's burden is now ", best_device.burden, " -- Inequality is now ", new_gini)

			# Now check if the improvement is good enough
			if best_improvement < e_min:
				break # Break the loop
				
				
			
		return self.x, tr_improvement, tr_objective, tr_gini # Return the profile, improvements, objective and gini over each iteration