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

tau = [0, 0.5, 0.95, 1]      # list of MAX 7 tau's  [0;1] to calculate. focus fairness vs flexibility: 1 is fully fairness, 0 is fully flexibility

e_min = 0.00		# e_min in the PS paper (0.001)
max_iters = 1000		# maximum number of iterations

nr_baseloads = 100
nr_batteries = 25
nr_evs = 25
nr_heatpumps = 25


# Create the model:
# Initialisation
devices = []
# Add some baseloads
for i in range(0,nr_baseloads):
	devices.append(Load(i))
# Add some batteries
for i in range(0,nr_batteries):
	devices.append(Battery())
# Add some EVs
for i in range(0,nr_evs):
	devices.append(ElectricVehicle(i))
# Add some Heatpumps
for i in range(0,nr_heatpumps):
	devices.append(HeatPump(i))
		
# give an id to each device (exclude baseloads). Shorter: device_list.append(i)           
device_list = []
id_counter = 1 
for device in devices:
    if device.type != "BL":  
        device_list.append(f"{id_counter}-{device.type}")
        id_counter += 1
	
	
# Run the Profile Steering algorithm
initial_profile = [None] * len(tau)
power_profile = [None] * len(tau)
tr_improvement = [None] * len(tau)
tr_objective = [None] * len(tau)
tr_gini = [None] * len(tau)
burdens = [None] * len(tau)
tic()                                               # track time
ps = ProfileSteering(devices)                       # initialize devices in algorithm
initial_profile = ps.init(desired_profile)          # Initial planning
for t in range(len(tau)):                           # run for all Tau's
    print("New iteration starting with tau = ", tau[t])
    ps.rerun(initial_profile)  # reset devices
    power_profile[t], tr_improvement[t], tr_objective[t], tr_gini[t] = ps.iterative(e_min, max_iters, tau[t])  # Iterative phase
    burdens[t] = [device.burden for device in devices if device.type != "BL"]   # save burdens of each device
print('Elapsed time (s): ', toc())

# And now power_profile has the result
#print("Resulting profile", power_profile)

# PLOTS
# Tools like matplotlib let you plot this in a nice way
# Other tools may also have this available

# Initialize grid of subplots
fig, axes = plt.subplots(1, 5, figsize=(22, 5))  # x rows, y columns
cmap = plt.get_cmap('tab10')  # get a color map (up to 10)

# First subplot: Initial & Optimized planning
t = np.arange(0., intervals/4, 0.25)              #sample values at 15mins (scale is in hours)
axes[0].plot(t, initial_profile, 'r', alpha=0.1, label='Initial for all tau')  # initial, same for all tau
for i in range(len(tau)):
    axes[0].plot(t, power_profile[i], color=cmap(i % cmap.N), alpha=0.8-len(tau)/10, label=f'Optimized with tau={tau[i]}')
axes[0].set_xlabel('Planning [h]')
axes[0].set_ylabel('Power profile [W]')
axes[0].set_title('Power profile before and after optimizing')
axes[0].legend()
# print maximum peaks [W]
print(f'Max power peak initially: {np.max(initial_profile):.0f} W')
print('\n'.join([f'Max power for tau = {tau[i]}: {np.max(power_profile[i]):.0f} W' for i in range(len(tau))]))

# Second subplot: Iteration vs Improvement
for i in range(len(tau)):
    axes[1].plot(tr_improvement[i], color=cmap(i % cmap.N), alpha=0.8-len(tau)/10, label=f'tau={tau[i]}')
axes[1].set_xlabel('Iterations [#]')
axes[1].set_ylabel('Improvement')
axes[1].set_title('Improvement per Iteration')
axes[1].set_yscale('log')
axes[1].grid(True)
axes[1].legend()

# Third subplot: Iteration vs Objective
for i in range(len(tau)):
    axes[2].plot(tr_objective[i], color=cmap(i % cmap.N), alpha=0.8-len(tau)/10, label=f'tau={tau[i]}')
axes[2].set_xlabel('Iterations [#]')
axes[2].set_ylabel('Objective score (2-norm)')
axes[2].set_title('Objective Score per Iteration')
axes[2].grid(True)
axes[2].legend()

# Fourth subplot: Iteration vs Fairness (Gini Coefficient)
for i in range(len(tau)):
    axes[3].plot(tr_gini[i], color=cmap(i % cmap.N), alpha=0.8-len(tau)/10, label=f'tau={tau[i]}')
axes[3].set_xlabel('Iterations [#]')
axes[3].set_ylabel('Gini Coefficient')
axes[3].set_title('Inequality per Iteration (Gini Coefficient)')
axes[3].grid(True)
axes[3].legend()

# Fifth subplot: Final Distribution of Burdens for all Devices (exclude baseloads)
bar_width = 0.8 / len(tau)
bar_x = np.arange(len(device_list))     # nr of positions on bar charts
#plot each device, each tau next to each other
for i in range(len(tau)):
    color = cmap(i % cmap.N)
    offset = i * bar_width
    axes[4].bar(bar_x + offset, burdens[i], width=bar_width, color=color, label=f'tau={tau[i]}')
axes[4].set_xticks(bar_x + bar_width * (len(tau) - 1) / 2)
axes[4].set_xticklabels(device_list, rotation=90, ha='right')
axes[4].set_title('Distribution of burdens')
axes[4].set_xlabel('Devices')
axes[4].set_ylabel('Burden')
axes[4].legend()

# Finalize plot
plt.tight_layout()
plt.show()