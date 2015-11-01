import sys
import numpy as np
import pandas as pd
import Machine_Learning
import Make_Plot
import Correlation

def read_climdiv(filename, varname):
  df = pd.read_table("Data_Sets/"+filename,parse_dates=False,infer_datetime_format=False, sep = '   |  | ',  names=['DataSet','Jan_'+varname, 'Feb_'+varname, 'Mar_'+varname, 'Apr_'+varname, 'May_'+varname, 'Jun_'+varname, 'Jul_'+varname, 'Aug_'+varname, 'Sep_'+varname, 'Oct_'+varname, 'Nov_'+varname, 'Dec_'+varname], converters={0: str}, engine='python')
  
  state_vals = []
  div_vals   = []
  elem_vals  = []
  year_vals  = []

  for value in df['DataSet']:
    value_str = str(value)
    state_vals.append(value_str[0:3])
    div_vals.append(value_str[3:4])
    elem_vals.append(value_str[4:6])
    year_vals.append(int(value_str[-4:]))

  df['State Code']      = state_vals
  df['Division Number'] = div_vals
  df['Element Code']    = elem_vals
  df['Year']            = year_vals
  df.set_index('Year', inplace = True)

  df.drop('DataSet', axis=1, inplace=True)
  return df[['Jan_'+varname, 'Feb_'+varname, 'Mar_'+varname, 'Apr_'+varname, 'May_'+varname, 'Jun_'+varname, 'Jul_'+varname, 'Aug_'+varname, 'Sep_'+varname, 'Oct_'+varname, 'Nov_'+varname, 'Dec_'+varname]][df['State Code'] == '013']

def read_SM(filename, varname):
  df = pd.read_table("Data_Sets/"+filename, parse_dates=False, infer_datetime_format=False, sep='    	    ',  names=['Months', 'Avg SM'], engine='python')
  df = df.drop(df.index[:2])
  df['Months'] = df['Months'].astype(float)
  df['Date'] = (df['Months'] - 0.5)/12 + 1960
  df.drop('Months', axis=1, inplace=True)
  df['Year'] = df['Date'].astype(int)

  Jan = 'Jan_'+varname
  Feb = 'Feb_'+varname
  Mar = 'Mar_'+varname
  Apr = 'Apr_'+varname
  May = 'May_'+varname
  Jun = 'Jun_'+varname
  Jul = 'Jul_'+varname
  Aug = 'Aug_'+varname
  Sep = 'Sep_'+varname
  Oct = 'Oct_'+varname
  Nov = 'Nov_'+varname
  Dec = 'Dec_'+varname

  df['Month'] = 'Dec_'+varname
  pd.set_option('mode.chained_assignment',None)

  df['Month'][np.round((df['Date']-df['Year'])*12) == 0] = ['Jan_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 1] = ['Feb_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 2] = ['Mar_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 3] = ['Apr_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 4] = ['May_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 5] = ['Jun_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 6] = ['Jul_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 7] = ['Aug_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 8] = ['Sep_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 9] = ['Oct_'+varname]
  df['Month'][np.round((df['Date']-df['Year'])*12) == 10] = ['Nov_'+varname]
  df.drop('Date', axis=1, inplace=True)

  jan = pd.DataFrame()
  jan[[Jan, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Jan]
  jan.set_index('Year', inplace = True)
  feb = pd.DataFrame()
  feb[[Feb, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Feb]
  feb.set_index('Year', inplace = True)
  mar = pd.DataFrame()
  mar[[Mar, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Mar]
  mar.set_index('Year', inplace = True)
  apr = pd.DataFrame()
  apr[[Apr, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Apr]
  apr.set_index('Year', inplace = True)
  may = pd.DataFrame()
  may[[May, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == May]
  may.set_index('Year', inplace = True)
  jun = pd.DataFrame()
  jun[[Jun, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Jun]
  jun.set_index('Year', inplace = True)
  jul = pd.DataFrame()
  jul[[Jul, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Jul]
  jul.set_index('Year', inplace = True)
  aug = pd.DataFrame()
  aug[[Aug, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Aug]
  aug.set_index('Year', inplace = True)
  sep = pd.DataFrame()
  sep[[Sep, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Sep]
  sep.set_index('Year', inplace = True)
  oct = pd.DataFrame()
  oct[[Oct, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Oct]
  oct.set_index('Year', inplace = True)
  nov = pd.DataFrame()
  nov[[Nov, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Nov]
  nov.set_index('Year', inplace = True)
  dec = pd.DataFrame()
  dec[[Dec, 'Year']]  = df[['Avg SM', 'Year']][df['Month'] == Dec]
  dec.set_index('Year', inplace = True)
  
  df_sorted = pd.concat([jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec], axis=1)
  return df_sorted

def read_CY(filename):
  df1 = pd.DataFrame.from_csv("Data_Sets/"+filename+"_planted.csv", parse_dates=False, infer_datetime_format=False)
  df2 = pd.DataFrame.from_csv("Data_Sets/"+filename+"_production.csv", parse_dates=False, infer_datetime_format=False)

  df1.dropna()
  df2.dropna()

  df3 =  df1[['AREA HARVESTED in ACRES', 'AREA PLANTED in ACRES', 'PRODUCTION in TONS']]
  df4 =  df2[['AREA HARVESTED in ACRES', 'AREA PLANTED in ACRES', 'PRODUCTION in TONS']]
  df = df2[['AREA HARVESTED in ACRES', 'PRODUCTION in TONS']].join(df1['AREA PLANTED in ACRES'])
  return df

def read_land(filename):
  df = pd.read_csv('Data_Sets/'+filename, thousands=',')
  df2 = df[['Value', 'Year']][df['Data Item'] == 'AG LAND, INCL BUILDINGS - ASSET VALUE, MEASURED IN $ / ACRE']
  df2.set_index('Year', inplace = True)
  return df2

def read_crop_yield(filename):
  df = pd.DataFrame.from_csv("Data_Sets/"+filename, parse_dates=False, infer_datetime_format=False)
  df.set_index('Year', inplace = True)
  df1 = df[df['Period'] == 'YEAR']
  df1.sort_index(inplace=True)
  return df1['Value'][df1['Data Item'].str.contains('GRAIN') == True]

def main():
# Average Precipitation per month
  precip = read_climdiv('climdiv-pcpnst-v1.0.0-20150904.txt', 'precip')

# Average Temperature per month
  tempr  = read_climdiv('climdiv-tmpcst-v1.0.0-20150904.txt', 'tempr')

# Average Modified Palmer Drought Severity Index per month
  PDSI   = read_climdiv('climdiv-pmdist-v1.0.0-20151007.txt', 'PDSI')

# Average Soil Moisture per month
  SM     = read_SM('Avg_Soil_Moisture.tsv', 'SM')

# Land use: http://quickstats.nass.usda.gov/#BE906494-6004-377E-8974-9D3A9D0605B6
  Land   = read_land('8F11D825-02A1-3B3D-8470-EC931FA0D6B5.csv')

# Crop Yield per year
# Yield = read_CY('CORN-AcreageYieldandProductionIrrigatedNonIrrigated-2015-09-17')
  Yield = read_crop_yield('Corn_Yield.csv')
  Yield.columns = ['Crop Yield']

# Choose the range of years for the data avilable (limited by the Soil Mosture here)
  year_range = range(1948, 2015)
# Choose which months to use the data from (growing season)
  month_range = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
# month_range = ['Apr', 'May', 'Jun', 'Jul']

# Set the training data with the above prescribed limitations
  input1 = pd.concat([precip.ix[year_range], tempr.ix[year_range], PDSI.ix[year_range], SM.ix[year_range], Land.ix[year_range]], axis=1)
  input = pd.DataFrame(index=input1.index)
  for mnth in month_range:
    input = pd.concat([input, input1.filter(like=mnth)], axis=1)
  input = pd.concat([input, input1['Value']], axis=1)
# Set the ouput values
  output = Yield.ix[year_range]

  Make_Plot.plot_data(input, output)

# date_range = range(1959, 2015, 11)
  date_range = [1970, 1981, 1992]

  correlation = Correlation.pearson(input, output)
  print correlation

  Predicted_Output = Machine_Learning.Machine_Learning(input, output, date_range)
  Predicted_Output.to_csv('Predicted_Yield.csv')

  Make_Plot.error(Predicted_Output)
  Make_Plot.pred_yield(Predicted_Output)

if __name__ == '__main__':
  main()
