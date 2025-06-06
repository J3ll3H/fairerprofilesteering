#!/usr/bin/python3

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



# Implementation of the algorithm presented in:
# M.E.T. Gerards et al., "Demand Side Management using Profile Steering", IEEE PowerTech, 2015, Eindhoven.
# https://research.utwente.nl/en/publications/demand-side-management-using-profile-steering
# Interactive demo: https://foreman.virt.dacs.utwente.nl/~gerardsmet/vis/ps.html

# Optimization of devices (OptAlg file) based on the work of Thijs van der Klauw)
# https://ris.utwente.nl/ws/portalfiles/portal/12378855/thesis_T_van_der_Klauw.pdf



# Importing models
from dev.load import Load
from dev.battery import Battery
from dev.electricvehicle import ElectricVehicle
from dev.heatpump import HeatPump

# Import Profile Steering
from profilesteering import ProfileSteering

# Import libraries
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for number math
from ttictoc import tic,toc         # for timekeeping, install using cmd: pip install ttictoc

# Settings
intervals = 96                      # number of 15min intervals: 96 is 24hours
desired_profile = [0]*intervals		# d in the PS paper
power_profile = [0]*intervals		# x in the PS paper

tau = 0.9      # [0;1] focus on fairness vs flexibility: 1 is fully fairness, 0 is fully flexibility

e_min = 0.001		# e_min in the PS paper (0.001)
max_iters = 100		# maximum number of iterations

nr_baseloads = 100
nr_batteries = 2
nr_evs = 10
nr_heatpumps = 5


# Create the model:
# Initialisation
devices = []
# Add some baseloads
for i in range(0,nr_baseloads):
	devices.append(Load())
# Add some batteries
for i in range(0,nr_batteries):
	devices.append(Battery())
# Add some EVs
for i in range(0,nr_evs):
	devices.append(ElectricVehicle())
# Add some Heatpumps
for i in range(0,nr_heatpumps):
	devices.append(HeatPump())
	
	
# Run the Profile Steering algorithm
tic()
ps = ProfileSteering(devices)
power_profile = ps.init(desired_profile)        # Initial planning
initial_profile = power_profile                 # Store initial planning
power_profile, tr_improvement, tr_objective, tr_gini = ps.iterative(e_min, max_iters, tau)  # Iterative phase
print('Elapsed time (s): ', toc())

# And now power_profile has the result
#print("Resulting profile", power_profile)

# PLOTS
# Tools like matplotlib let you plot this in a nice way
# Other tools may also have this available

# Initialize grid of subplots
fig, axes = plt.subplots(1, 5, figsize=(22, 5))  # x rows, y columns

# First subplot: Initial & Optimized planning
t = np.arange(0., intervals/4, 0.25)              #sample values at 15mins (scale is in hours)
axes[0].plot(t, initial_profile, 'r', label='Initial')
axes[0].plot(t, power_profile, 'b', label='Optimized')
axes[0].set_xlabel('Planning [h]')
axes[0].set_ylabel('Power profile [W]')
axes[0].set_title('Power profile before and after optimizing')
axes[0].legend()

# Second subplot: Iteration vs Improvement
axes[1].plot(tr_improvement, 'g')
axes[1].set_xlabel('Iterations [#]')
axes[1].set_ylabel('Improvement')
axes[1].set_title('Improvement per Iteration')
axes[1].set_yscale('log')
axes[1].grid(True)

# Third subplot: Iteration vs Objective
axes[2].plot(tr_objective, 'b')
axes[2].set_xlabel('Iterations [#]')
axes[2].set_ylabel('Objective score (2-norm)')
axes[2].set_title('Objective Score per Iteration')
axes[2].grid(True)

# Fourth subplot: Iteration vs Fairness (Gini Coefficient)
axes[3].plot(tr_gini, 'y')
axes[3].set_xlabel('Iterations [#]')
axes[3].set_ylabel('Gini Coefficient')
axes[3].set_title('Inequality per Iteration (Gini Coefficient)')
axes[3].grid(True)

# Fifth subplot: Final Distribution of Burdens for all Devices (exclude baseloads)
device_list = []
for i, device in enumerate(devices, start=1):
	#device_list.append(f"{i}: {device.type}")
	if device.type != "BL":
	    device_list.append(i)           # give an id to each device
burdens = [device.burden for device in devices if device.type != "BL"]
axes[4].bar(device_list, burdens)
axes[4].set_title('Distribution of burdens')
axes[4].set_xlabel('Devices')
axes[4].set_ylabel('Burden')

# Finalize plot
plt.tight_layout()
plt.show()