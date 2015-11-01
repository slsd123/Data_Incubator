import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from ggplot import *
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.gridspec as gridspec

def error(df):
  x_vals = []
  for values in df.index:
    x_vals.append(str(values))
  n_groups = len(df.index)
  rects    = 9

  fig, ax = plt.subplots()

  index = np.arange(n_groups)
  bar_width = 0.075

  opacity = 0.4

  rects1 = plt.bar(index - 3*bar_width, df['TheilSen Percent Error'], bar_width,
                 alpha=opacity,
                     color='b',
                   label='Theil Sen', align='center')

  rects1 = plt.bar(index - 2*bar_width, df['Linear Percent Error'], bar_width,
                 alpha=opacity,
                     color='0.75',
                   label='Linear', align='center')

  rects2 = plt.bar(index - 1*bar_width, df['Polynomial Percent Error'], bar_width,
                 alpha=opacity,
                     color='g',
                   label='Polynomial', align='center')

  rects3 = plt.bar(index - 0*bar_width, df['Ridge Percent Error'], bar_width,
                 alpha=opacity,
                     color='m',
                   label='Ridge', align='center')

  rects4 = plt.bar(index + 1*bar_width, df['Lasso Percent Error'], bar_width,
                 alpha=opacity,
                     color='chartreuse',
                   label='Lasso', align='center')

  rects5 = plt.bar(index + 2*bar_width, df['Elastic Percent Error'], bar_width,
                 alpha=opacity,
                     color='b',
                   label='Elastic', align='center')

  rects6 = plt.bar(index + 3*bar_width, df['BayRidge Percent Error'], bar_width,
                 alpha=opacity,
                     color='k',
                   label='Bayesian Ridge', align='center')

  rects7 = plt.bar(index + 4*bar_width, df['GPML Percent Error'], bar_width,
                 alpha=opacity,
                     color='y',
                   label='GPML', align='center')


  plt.xlabel('Year')
  plt.ylabel('% Error')
  plt.title('Predicted Crop Yield Percent Error')
  plt.xticks(index + bar_width, x_vals)
  plt.legend(ncol=2, loc='upper center')

  plt.ylim(ymax = 100)

  plt.savefig('Output/Percent_Errors.png', bbox_inches='tight')

  plt.show()

def pred_yield(df):
  x_vals = []
  for values in df.index:
    x_vals.append(str(values))
  n_groups = len(df.index)
  rects    = 9

  fig, ax = plt.subplots()

  index = np.arange(n_groups)
  bar_width = 0.075

  opacity = 0.4

  rects1 = plt.bar(index - 3*bar_width, df['Linear'], bar_width,
                 alpha=opacity,
                     color='0.75',
                   label='Linear', align='center')

  rects2 = plt.bar(index - 2*bar_width, df['TheilSen'], bar_width,
                 alpha=opacity,
                     color='b',
                   label='Theil Sen', align='center')

  rects3 = plt.bar(index - 1*bar_width, df['Polynomial'], bar_width,
                 alpha=opacity,
                     color='g',
                   label='Polynomial', align='center')

  rects4 = plt.bar(index - 0*bar_width, df['Ridge'], bar_width,
                 alpha=opacity,
                     color='m',
                   label='Ridge', align='center')

  rects5 = plt.bar(index + 1*bar_width, df['Lasso'], bar_width,
                 alpha=opacity,
                     color='chartreuse',
                   label='Lasso', align='center')

  rects6 = plt.bar(index + 2*bar_width, df['Elastic'], bar_width,
                 alpha=opacity,
                     color='b',
                   label='Elastic', align='center')

  rects7 = plt.bar(index + 3*bar_width, df['BayRidge'], bar_width,
                 alpha=opacity,
                     color='k',
                   label='Bayesian Ridge', align='center')

  rects8 = plt.bar(index + 4*bar_width, df['GPML'], bar_width,
                 alpha=opacity,
                     color='y',
                   label='GPML', align='center')

  rects9 = plt.bar(index + 5*bar_width, df['Crop Yield'], bar_width,
                   color='r',
                   label='Actual Yield', align='center')

  plt.xlabel('Year')
  plt.ylabel('Yield')
  plt.title('Predicted and Actual Crop Yield')
  plt.xticks(index + bar_width, x_vals)
  plt.legend(ncol=3, loc='upper center')

  plt.ylim(ymax = 200)
  plt.xlim([-4*bar_width,4*bar_width*9-3*bar_width])

  plt.savefig('Output/Predicted_Yield.png', bbox_inches='tight')

  plt.show()

def plot_data(input, output):
  precip = input.filter(like = 'precip').mean(axis=1)
  tempr  = input.filter(like = 'tempr').mean(axis=1)
  PDSI   = input.filter(like = 'PDSI').mean(axis=1)
  SM     = input.filter(like = 'SM').mean(axis=1)
  Land   = input['Value']

  data = pd.concat([precip, tempr, PDSI, SM, Land, output], axis=1)
  data.columns = ['precip', 'tempr', 'PDSI', 'SM', 'Land', 'Crop Yield']

  f   = plt.figure()
  gs  = gridspec.GridSpec(3, 1, height_ratios=[3,1,1])
  ax1 = f.add_subplot(gs[0])
  ax2 = f.add_subplot(gs[0])
  ax3 = f.add_subplot(gs[1], sharex=ax1)
  ax4 = f.add_subplot(gs[2], sharex=ax1)

  p1 = ax1.plot(data.index, data['Crop Yield'], '-Dy', label='Corn Yield')
  for tl in ax1.get_yticklabels():
    tl.set_color('y')
  ax1.yaxis.label.set_color('y')
  ax1.set_title('Crop Yield Statistics (Avg. Over Growing Season)')

  ax2 = ax1.twinx()
  p2 = ax2.plot(data.index, data['Land'], '-g', label='Asset Value')
  for tl in ax2.get_yticklabels():
    tl.set_color('g')
  ax2.yaxis.label.set_color('green')

  ax1.set_ylabel('Corn Yield')
  ax2.set_ylabel('Farm Land Asset Value ($/Acre)')

  plt.setp([p1,p2], linewidth=2)

  lns = p1 + p2
  labs = [l.get_label() for l in lns]
  leg = plt.legend(lns, labs, loc='best')

# Precipitation plot
  ax3.bar(data.index, data['precip'], color='b')
  for tl in ax3.get_yticklabels():
    tl.set_color('b')
  ax3.set_yticks([0,3,6])
  ax3.set_ylabel('Avg. Precip')
  ax3.yaxis.label.set_color('blue')

# Average temperature plot
  ax4.bar(data.index, data['tempr'], color='r')
  for tl in ax4.get_yticklabels():
    tl.set_color('r')
  ax4.set_ylim([60, 70])
  ax4.set_ylabel('Avg. Temp')
  ax4.set_yticks([60,64,68])
  ax4.set_xticks(range(1950,2020,10))
  ax4.yaxis.label.set_color('red')
  ax4.set_xlabel('Year')
  ax4.set_xlim([1947, 2015])

  f.subplots_adjust(hspace=0.1)
  plt.setp([a.get_xticklabels() for a in f.axes[:-2]], visible=False)

  plt.savefig('Output/Actual_Stats.png', bbox_inches='tight')
  plt.show()
