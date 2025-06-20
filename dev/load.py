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

import random
import operator 
import numpy as np

class Load():
	def __init__(self, seed):
		self.profile = []	# x_m in the PS paper
		self.candidate = []	# ^x_m in the PS paper
		self.type = "BL"
		self.burden = 0						# total bore burden / discomfort of this device
		self.candidate_improvement = 0		# last proposed improvement
		self.candidate_burden = 0			# last proposed burden their candidate would inflict
		self.initial_profile = []			# saved initial profile, to be saved for reruns
		self.seed = seed					# set seed for random profile generator
		
		# Device specific params
		self.max = 5000	# W
	
	def init(self, p):
		# Create a baseload for a given number of intervals
		self.profile = [] # Empty the profile
		
		# We create a random list of power values, but it can be any list
		for i in range(0, len(p)):
			random.seed(self.seed+i)		# set seed, different for each profile value					
			self.profile.append(self.max*random.random())
			
		self.initial_profile = self.profile	
			
		return list(self.profile)
			
	def plan(self, d):
		assert(len(d) == len(self.profile))
		p_m = list(map(operator.sub, self.profile, d)) # p_m = x_m - d
		
		self.candidate = list(self.profile)	# A baseload offers no flex, so we can just return the profile
											# Note that we need to create a new list due to "hidden pointers" in Python
											
		# Calculate the improvement by this device:
		self.candidate_improvement = np.linalg.norm(np.array(self.profile)-np.array(p_m)) - np.linalg.norm(np.array(self.candidate)-np.array(p_m))
		
		# Calculate the additional burden / discomfort this change would inflict on this device:
		self.candidate_burden = np.linalg.norm(np.array(self.candidate)-np.array(self.initial_profile), ord=1) 	# deviation from initial profile, will be 0
		
		# Return the improvement and additional burden
		# Note that e_m should be 0 for a static device
		# print("Improvement: ", self, e_m)
		return self.candidate_improvement, self.candidate_burden
		
	def accept(self):
		# We are chosen as winner, replace the profile:
		diff = list(map(operator.sub, self.candidate, self.profile))
		self.profile = list(self.candidate)
		self.burden = self.candidate_burden			# update bore burden / discomfort 
		
		# Note we can send the difference profile only as incremental update
		return diff		# return 0 difference as baseload does not change anything when picked 