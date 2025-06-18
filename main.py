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

# Adaption for 'Fairer Profile Steering' done by Jelle de Haan in June 2025 for a BSc Thesis



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

tau = [0, 0.5, 0.75, 0.95, 1]      # list of tau's  [0;1] to calculate. -1 = vanilla PS. focus fairness vs flexibility: 1 is fully fairness, 0 is fully flexibility
#tau = np.round(np.arange(0, 1.001, 0.05), 2).tolist()   # list from 0-1 with 0.05 precision for a detailed graph across tau, need to comment out legends of other graphs for this

nr_baseloads = 100
nr_batteries = 25
nr_evs = 25
nr_heatpumps = 25

e_min = 0.00		    # e_min in the PS paper (0.001)
max_iters = 2000		# maximum number of iterations


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
times = [None] * len(tau)
#tic()      # track time
ps = ProfileSteering(devices)                       # initialize devices in algorithm
initial_profile = ps.init(desired_profile)          # Initial planning
for t in range(len(tau)):                           # run for all Tau's
    tic()                                           # track time
    print("Starting new sweep with tau = ", tau[t])
    ps.rerun(initial_profile)  # reset devices
    power_profile[t], tr_improvement[t], tr_objective[t], tr_gini[t] = ps.iterative(e_min, max_iters, tau[t])  # Iterative phase
    burdens[t] = [device.burden for device in devices if device.type != "BL"]   # save burdens of each device
    times[t] = toc()    
#print('Elapsed time (s): ', toc())

# And now power_profile has the result
#print("Resulting profile", power_profile)

# PRINTS
# Print out table with inf-norm, 2-norm, G, time
cw = 20 # columnwidth
print(f"{'Controller':<{cw}}{'Inf-norm (W)':<{cw}}{'2-Norm':<{cw}}{'Gini':<{cw}}{'Time (s)':<{cw}}")
print('-' * 5 * cw)

# Initial
initial_objective = np.linalg.norm(np.array(initial_profile)-np.array(desired_profile))	# Initial objective score 2-norm x-p
print(f"{'Initial':<{cw}}{np.max(initial_profile):<{cw}.2f}{initial_objective:<{cw}.2f}")

# for different tau
for i in range(len(tau)): 
    tau_label = "Regular PS" if tau[i] == -1 else f"tau = {tau[i]:.2f}"  # Fix for alignment
    print(f"{tau_label:<{cw}}{np.max(power_profile[i]):<{cw}.2f}{tr_objective[i][-1]:<{cw}.2f}{tr_gini[i][-1]:<{cw}.4f}{times[i]:<{cw}.2f}")


# PLOTS
# Tools like matplotlib let you plot this in a nice way
# Other tools may also have this available

# Initialize grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 8))  # x rows, y columns
axes = axes.flatten()       # flatten rows/columns as one list
cmap = plt.get_cmap('Set1')  # get a color map (up to 10)
trp = 0.6   # transparency

# First subplot: Power Profile indication
t = np.arange(0., intervals/4, 0.25)              #sample values at 15mins (scale is in hours)
axes[0].plot(t, initial_profile, 'r', alpha=0.2, label='Initial for all $\\tau$')  # initial, same for all tau
for i in range(len(tau)):
    if (tau[i] == -1):
        axes[0].plot(t, power_profile[i], color=cmap(i % cmap.N), alpha=trp, label=f'Optimized with regular PS')
    else:    
        axes[0].plot(t, power_profile[i], color=cmap(i % cmap.N), alpha=trp, label=f'Optimized with $\\tau$={tau[i]}')
axes[0].set_xlabel('Planning [h]')
axes[0].set_ylabel('Power profile [W]')
axes[0].set_title('Aggregate power profile')
axes[0].legend()

# Second subplot: Objective over Iterations
for i in range(len(tau)):
    if (tau[i] == -1):
        axes[1].plot(tr_objective[i], color=cmap(i % cmap.N), alpha=trp, label=f'Regular PS')
    else:
        axes[1].plot(tr_objective[i], color=cmap(i % cmap.N), alpha=trp, label=f'$\\tau$={tau[i]}')
axes[1].set_xlabel('Iterations [#]')
axes[1].set_ylabel('Objective score (2-norm)')
axes[1].set_title('Objective Score per Iteration')
axes[1].grid(True)
axes[1].legend()

# Third subplot: Improvement over Iterations
for i in range(len(tau)):
    if (tau[i] == -1):
        axes[2].plot(tr_improvement[i], color=cmap(i % cmap.N), alpha=trp, label=f'Regular PS')
    else:
        axes[2].plot(tr_improvement[i], color=cmap(i % cmap.N), alpha=trp, label=f'$\\tau$={tau[i]}')
axes[2].set_xlabel('Iterations [#]')
axes[2].set_ylabel('Improvement of objective (logarithmic)')
axes[2].set_title('Improvement per Iteration')
axes[2].set_yscale('log')
axes[2].grid(True)
axes[2].legend()

# Fourth subplot: Final Distribution of Burdens for all Controllable Devices
## Bar chart::
#bar_width = 0.8 / len(tau)
#bar_x = np.arange(len(device_list))     # nr of positions on bar charts
##plot each device, each tau next to each other
#for i in range(len(tau)):
#    color = cmap(i % cmap.N)
#    offset = i * bar_width
#    if (tau[i] == -1):
#        axes[3].bar(bar_x + offset, burdens[i], width=bar_width, color=color, label=f'Regular PS')
#    else:
#        axes[3].bar(bar_x + offset, burdens[i], width=bar_width, color=color, label=f'tau={tau[i]}')
#axes[3].set_xticks(bar_x + bar_width * (len(tau) - 1) / 2)
#axes[3].set_xticklabels(device_list, rotation=90, ha='right')
#axes[3].set_title('Distribution of burdens')
#axes[3].set_xlabel('Devices')
#axes[3].set_ylabel('Burden')
#axes[3].legend()
# Violin chart:
# Prepare the data
positions = np.arange(len(tau))
#labels = ['Regular PS' if tau_val == -1 else f'tau={tau_val}' for tau_val in tau]
labels = [
    f"$\\tau$ = {t}\nGini = {g:.2f}" if t != -1 else f"Reg. PS\nGini = {g:.2f}"
    for t, g in zip(tau, [tr_gini[i][-1] for i in range(len(tau))])
]
burden_data = [burdens[i] for i in range(len(tau))]  # Each item is a list of burdens for that tau
# Create the violin plot
parts = axes[3].violinplot(burden_data, positions=positions, showmedians=True, widths=0.8)
# Set colors using your colormap
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cmap(i % cmap.N))
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
# Styling
axes[3].set_xticks(positions)
axes[3].set_xticklabels(labels, rotation=30, ha='right')
axes[3].set_title('Final Distribution of Burdens')
axes[3].set_ylabel('Burden of Controllable Device')

# Fifth subplot: Fairness (GC) over Iterations
for i in range(len(tau)):
    if (tau[i] == -1):
        axes[4].plot(tr_gini[i], color=cmap(i % cmap.N), alpha=trp, label=f'Regular PS')
    else:
        axes[4].plot(tr_gini[i], color=cmap(i % cmap.N), alpha=trp, label=f'$\\tau$={tau[i]}')
axes[4].set_xlabel('Iterations [#]')
axes[4].set_ylabel('Gini Coefficient')
axes[4].set_title('Inequality per Iteration')
axes[4].set_ylim(0, 1)  # limit to [0;1]
axes[4].grid(True)
axes[4].legend()

## Sixth subplot: Inequality (GC) + Objective (2-norm) over tau
## Filter out "Regular PS" (tau == -1)
#filtered_tau = [tau[i] for i in range(len(tau)) if tau[i] != -1]
#filtered_gini = [tr_gini[i][-1] for i in range(len(tau)) if tau[i] != -1]
#filtered_obj = [tr_objective[i][-1] for i in range(len(tau)) if tau[i] != -1]
## Plot Gini on left y-axis
#axes[5].plot(filtered_tau, filtered_gini, marker='o', color='tab:red')
#axes[5].set_ylabel('Inequality (Gini Coefficient)', color='tab:red')
#axes[5].tick_params(axis='y', labelcolor='tab:red')
#axes[5].set_ylim(0, 1)  # limit to [0;1]
# Create twin axis for Objective
#ax2 = axes[5].twinx()
#ax2.plot(filtered_tau, filtered_obj, marker='o', color='tab:blue')
#ax2.set_ylabel('Objective (2-norm)', color='tab:blue')
#ax2.tick_params(axis='y', labelcolor='tab:blue')
#x2.set_ylim(min(filtered_obj)-1, max(filtered_obj)+1) 
# Shared x-axis settings
#axes[5].set_xlabel('Tunable Focus on Fairness [$\\tau$]')
#axes[5].set_title('Inequality and Objective across $\\tau$ after 2000 iterations')
#axes[5].grid(True)

# Sixth subplot: HARDCODED! Inequality (GC) + Iterations to converge across tau
convergencespeed =  [
    {"tau": 0.00, "gini": 0.4902, "iterations": 55},
    {"tau": 0.05, "gini": 0.4721, "iterations": 55},
    {"tau": 0.10, "gini": 0.4624, "iterations": 55},
    {"tau": 0.15, "gini": 0.4542, "iterations": 55},
    {"tau": 0.20, "gini": 0.4523, "iterations": 55},
    {"tau": 0.25, "gini": 0.4475, "iterations": 55},
    {"tau": 0.30, "gini": 0.4278, "iterations": 55},
    {"tau": 0.35, "gini": 0.4195, "iterations": 55},
    {"tau": 0.40, "gini": 0.4118, "iterations": 55},
    {"tau": 0.45, "gini": 0.3755, "iterations": 60},
    {"tau": 0.50, "gini": 0.3672, "iterations": 60},
    {"tau": 0.55, "gini": 0.3621, "iterations": 65},
    {"tau": 0.60, "gini": 0.3567, "iterations": 130},
    {"tau": 0.65, "gini": 0.3448, "iterations": 240},
    {"tau": 0.70, "gini": 0.3421, "iterations": 255},
    {"tau": 0.75, "gini": 0.3435, "iterations": 290},
    {"tau": 0.80, "gini": 0.3434, "iterations": 320},
    {"tau": 0.85, "gini": 0.3459, "iterations": 335},
    {"tau": 0.90, "gini": 0.3495, "iterations": 395},
    {"tau": 0.95, "gini": 0.3100, "iterations": 680},
    {"tau": 1.00, "gini": 0.2883, "iterations": 1835},
]
# Extract values
taus = [v["tau"] for v in convergencespeed]
ginis = [v["gini"] for v in convergencespeed]
iters = [v["iterations"] for v in convergencespeed]
# Plot Gini on left y-axis
axes[5].plot(taus, ginis, marker='o', color='tab:red')
axes[5].set_ylabel('Inequality (Gini Coefficient)', color='tab:red')
axes[5].tick_params(axis='y', labelcolor='tab:red')
axes[5].set_ylim(0, 1)  # limit to [0;1]
# Create twin axis for iterations
ax2 = axes[5].twinx()
ax2.plot(taus, iters, marker='o', color='tab:blue')
ax2.set_ylabel('# of iterations to converge to final objective', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
# Shared x-axis settings
axes[5].set_xlabel('Tunable Focus on Fairness [$\\tau$]')
axes[5].set_title('Inequality and Convergence Speed across $\\tau$')
axes[5].grid(True)


# Finalize and render plot
plt.tight_layout()
plt.show()