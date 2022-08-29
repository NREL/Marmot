# -*- coding: utf-8 -*-
"""
This module plots total generation and total installed capacity together, using custom .csv inputs instead of the standard Marmot formatted .h5 file.

@author: Marty Schwarz
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

#####
#Step 1: Load data.
#####

#Total gen dataframe: Index should be regions, columns should be generation types.
Total_Gen_Stack = pd.read_csv('/path/to/total/generation_file.csv',index_col = 'regions')

#Curtailment dataframe (if applicable): Index should be regions, cwith a single column representing curtailment.
Stacked_Curt = pd.read_csv('/path/to/total/curtailment_file.csv',index_col = 'regions')
Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), column = 'Curtailment', value=Stacked_Curt)
Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

#Demand dataframe: Index should be regions, with a single column representing load. 
Total_Load = pd.read_csv('/path/to/total/load_file.csv',index_col = 'regions')

#Unserved energy dataframe (if applicable): Index should be regions, with a single column representing unserved energy. 
Unserved_Energy = pd.read_csv('/path/to/total/use_file.csv',index_col = 'regions')
if Unserved_Energy.empty: #If unserved energy doesn't exist, create a copy of the load dataframe, with all values = 0.
    Unserved_Energy = Total_Load.copy()
    Unserved_Energy.iloc[:,0] = 0

#Subtract unserved energy from load for graphing.
if (Unserved_Energy == 0).all() == False:
    Unserved_Energy = Total_Load - Unserved_Energy

#Pump load dataframe: Index should be regions, with a single column representing pump load. 
Pump_Load = pd.read_csv('/path/to/total/pump_load_file.csv',index_col = 'regions')
if Pump_Load.empty: #If pump load doesn't exist, create a copy of the load dataframe, with all values = 0.
    Pump_Load = Total_Load.copy()
    Pump_Load.iloc[:,0] = 0

if (Pump_Load == 0).all() == False:
    Total_Demand = Total_Load - Pump_Load
else:
    Total_Demand = Total_Load

#Capacity dataframe: Index should be regions, columns should be generation types.
Total_Installed_Capacity = pd.read_csv('/path/to/capacity_file.csv',index_col = 'regions')

#Color dictionary (Marty to provide mapping file).
PLEXOS_color_dict = pd.read_csv('/path/to/colormapping_file.csv')
PLEXOS_color_dict = PLEXOS_color_dict.rename(columns={PLEXOS_color_dict.columns[0]:'Generator',PLEXOS_color_dict.columns[1]:'Colour'})
PLEXOS_color_dict["Generator"] = PLEXOS_color_dict["Generator"].str.strip()
PLEXOS_color_dict["Colour"] = PLEXOS_color_dict["Colour"].str.strip()
PLEXOS_color_dict = PLEXOS_color_dict[['Generator','Colour']].set_index("Generator").to_dict()["Colour"]

#####
#Step 2: Set unit conversion return divisor and energy units
#####
max_value = max(Total_Gen_Stack.sum(axis=1))
if max_value < 1000 and max_value > 1:
    divisor = 1
    units = 'MW'
elif max_value < 1:
    divisor = 0.001
    units = 'kW'
elif max_value > 999999.9:
    divisor = 1000000
    units = 'TW'
else:
    divisor = 1000
    units = 'GW'   
unitconversion = {'units':units, 'divisor':divisor}

Total_Gen_Stack = Total_Gen_Stack/unitconversion['divisor']
Unserved_Energy = Unserved_Energy.T/unitconversion['divisor']
Total_Load = Total_Load.T/unitconversion['divisor']
Total_Demand = Total_Demand.T/unitconversion['divisor']
Pump_Load = Pump_Load.T/unitconversion['divisor']

#Data table of values to return to main program
Data_Table_Out = pd.concat([Total_Load.T,
                            Total_Demand.T,
                            Unserved_Energy.T,
                            Total_Gen_Stack],  axis=1, sort=False)
Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")

#####
#Step 3: Prepare total installed capacity plot
#####

#Plotting options
y_axes_decimalpt = 0 #Set number of decimal points for y axis.
capacity_units = MW #Set units of capacity data.
tick_labels = Total_Gen_Stack.index #Replace a list of custom names if desired.
num_labels = 7 #If number of bars is greater than this number, rotate x-axis labels.
angle = 45 #Rotation angle for x-tick labels.

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.35, hspace=0.2)
axs = axs.ravel()

Total_Installed_Capacity.plot.bar(stacked=True, ax=axs[0],color=[PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity.columns],edgecolor='black', linewidth='0.1')
axs[0].set_ylabel(f"Total Installed Capacity ({capacity_units})", color='black', rotation='vertical')
axs[0].get_legend().remove()

#####
#Step 4: Prepare total gen plot
#####

Total_Gen_Stack.plot.bar(stacked=True, ax = axs[1],color=[PLEXOS_color_dict.get(x, '#333333') for x in Total_Gen_Stack.columns], edgecolor='black', linewidth='0.1')

axs[1].set_ylabel(f"Total Generation ({unitconversion['units']}h)",color='black', rotation='vertical')
axs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))

#Draw patches
for n, region in enumerate(Total_Gen_Stack.index):
    x = [ax[1].patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
    height1 = [float(Total_Load[region].sum())]*2
    lp1 = plt.plot(x,height1, c='black', linewidth=3)
    if Pump_Load[region] > 0:
        height2 = [float(Total_Demand[region])]*2
        lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)

    if Unserved_Energy[region] > 0:
        height3 = [float(Unserved_Energy_Out[region])]*2
        plt.plot(x,height3, c='#DD0200', linewidth=1.5)
        axs[1].fill_between(x, height3, height1,facecolor = '#DD0200',alpha=0.5)

#Set legend
handles, labels = axs[1].get_legend_handles_labels()

#Combine all legends into one.
if Pump_Load.values.sum() > 0:
    handles.append(lp2[0])
    handles.append(lp1[0])
    labels += ['Demand','Demand + \n Storage Charging']
else:
    handles.append(lp1[0])
    labels += ['Demand']

if Unserved_Energy.values.sum() > 0:
    custom_legend_elements = Patch(facecolor='#DD0200',
                               alpha=0.5, edgecolor='#DD0200',
                               label='Unserved Energy')
    handles.append(custom_legend_elements)
    labels += ['Unserved Energy']

axs[1].legend(reversed(handles),reversed(labels), loc='lower left',bbox_to_anchor=(1.05,0), facecolor='inherit', frameon=True)

#####
#Step 5: Final plot tweaks and saving.
#####
for i in range(2):
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
    axs[i].tick_params(axis='y', which='major', length=5, width=1)
    axs[i].tick_params(axis='x', which='major', length=5, width=1)

    if (len(tick_labels)) >= num_labels:
        axs[i].set_xticklabels(tick_labels, rotation=angle, ha="right")
    else:
        labels = [textwrap.fill(x, 10, break_long_words=False) for x in tick_labels]
        axs[i].set_xticklabels(tick_labels, rotation=0)
    
#Add labels to panels
axs[0].set_title("set left panel title here", fontdict={"weight": "bold", "size": 11}, loc='left', pad=4)
axs[1].set_title("set right panel title here", fontdict={"weight": "bold", "size": 11}, loc='left', pad=4)

fig.savefig('/path/to/output_image.svg',dpi=600,bbox_inches='tight')
Data_Table_Out.to_csv('/path/to/output_csv.csv')
