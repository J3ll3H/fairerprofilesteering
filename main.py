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
import matplotlib.pyplot as plt 
import numpy as np

# Initialisation
devices = []

# Settings
intervals = 96
desired_profile = [0]*intervals		# d in the PS paper
power_profile = [0]*intervals		# x in the PS paper

e_min = 0.001		# e_min in the PS paper
max_iters = 100		# maximum number of iterations

# Create the model:

# number of devices
nr_baseloads = 100
nr_batteries = 100
nr_evs = 100
nr_heatpumps = 100

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
ps = ProfileSteering(devices)
power_profile = ps.init(desired_profile)        # Initial planning
initial_profile = power_profile                 # Store initial planning
power_profile = ps.iterative(e_min, max_iters)  # Iterative phase

# And now power_profile has the result
#print("Resulting profile", power_profile)

# Tools like matplotlib let you plot this in a nice way
# Other tools may also have this available

# Plot initial VS optimized planning
t = np.arange(0., intervals/4, 0.25)              #sample values at 15mins (scale is in hours)
plt.plot(t, initial_profile, 'r', label='Initial')
plt.plot(t, power_profile, 'b', label='Optimized')
plt.xlabel('Planning [h]')
plt.ylabel('Power profile [W?]')
plt.title('Power profile before and after optimizing')
plt.legend()
plt.show()