#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:02:00 2020

@author: xiejingyue
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import time

#Portfolio Analysis
#####Annual Returns
port_data = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/data.csv")
port_data = pd.DataFrame(port_data)
port_data.index = pd.to_datetime(port_data['Dates'])
port_data = port_data.drop(['Dates'], axis=1)
port_data['year'] = port_data.index.year

rep_ret_y= port_data['logret_rep'].groupby(port_data['year']).sum()
dem_ret_y = port_data['logret_dem'].groupby(port_data['year']).sum()

fig1 = plt.figure(figsize =(6,4))
plt.title('Annual Return')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('Return',fontsize=13,fontweight='bold')
plt.plot(rep_ret_y,color='black', label = 'rep')
plt.plot(dem_ret_y,color='blue', label = 'demo')
plt.legend()
plt.show()

#####Portfolio Variance 
rep_var_y= port_data['logret_rep'].groupby(port_data['year']).var()
dem_var_y = port_data['logret_dem'].groupby(port_data['year']).var()
fig2 = plt.figure(figsize =(6,4))
plt.title('Annual Return')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('variance',fontsize=13,fontweight='bold')
plt.plot(rep_var_y,color='black', label = 'rep')
plt.plot(dem_var_y,color='blue', label = 'demo')
plt.legend()
plt.show()


 


import seaborn as sns
#Rep Analysis
rep_data  = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/rep_data.csv")
rep_data = pd.DataFrame(rep_data)
rep_data.index = pd.to_datetime(rep_data['Dates'])
rep_data = rep_data.drop(['Dates'], axis=1)
rep_df = rep_data.drop(['Party'], axis=1)
rep_cov = rep_df.cov()

plt.figure(figsize =(10,5))
sns.set(font_scale=0.6)
sns.heatmap(rep_cov, xticklabels = rep_df.columns,yticklabels = rep_df.columns,annot=True)
plt.title('Covariance Matrix of rep portfolio',fontsize=15,fontweight='bold')


#window_size = 90
#def corr(window_size,data):
#    cor = {}
#    
#def correlation(window_size,data):
#    cor = {}
#    s1 = pd.Series(ETF_data['SPY'])
#    for t in tickers:
#        s = pd.Series(ETF_data[t])
#        cor[t] = s1.rolling(window_size).corr(s)
#    return cor
#df=pd.DataFrame.from_dict(correlation(window_size,ETF_data))
#rolling_correlation =  df.dropna() 
#rolling_correlation.plot(figsize=(18,8))
#plt.title('A rolling 90-day correlation of each sector ETF with the S&P index',fontsize=15,fontweight='bold')
#plt.xlabel('Year')
#plt.ylabel('Corr')
#
#
#for col in data.columns: 
#    print(col) 


#Demo Anlysis
dem_data  = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/demo_data.csv")
dem_data = pd.DataFrame(dem_data)
dem_data.index = pd.to_datetime(dem_data['Dates'])
dem_data = dem_data.drop(['Dates'], axis=1)
dem_df = dem_data.drop(['Party'], axis=1)
dem_cov = dem_df.cov()

plt.figure(figsize =(10,5))
sns.set(font_scale=0.6)
sns.heatmap(dem_cov, xticklabels = dem_df.columns,yticklabels = dem_df.columns,annot=True)
plt.title('Covariance Matrix of dem portfolio',fontsize=15,fontweight='bold')


############################################################################################################
##Risk Free Rate
rf_1 = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/F-F_Research_Data_Factors_daily.csv")
rf_2 = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/riskfree.csv")

rf_1['Date'] = pd.to_datetime(rf_1['Date'], format='%Y%m%d')
rf_1['Date'] = pd.to_datetime(rf_1['Date'])
rf_1.index = pd.to_datetime(rf_1['Date'])
rf_1 = pd.DataFrame(rf_1)
#rf_1= rf_1.drop(['Mkt-RF','SMB','HML'],axis=1)
rf_1= rf_1.drop(['Date','Mkt-RF','SMB','HML'],axis=1)
#rf_1['year'] = rf_1.index.year

rf_2 = pd.DataFrame(rf_2)
rf_2['Date'] =  pd.to_datetime(rf_2['Date'])
rf_2.index = pd.to_datetime(rf_2['Date'])
#rf_2 = rf_2.drop(['Div Yld','Grwth Rate','Div Pay Ratio','Mkt Return','Premium'], axis=1)
rf_2 = rf_2.drop(['Date','Div Yld','Grwth Rate','Div Pay Ratio','Mkt Return','Premium'], axis=1)
#port_data['year'] = port_data.index.year

port_data = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/data.csv")
port_data = pd.DataFrame(port_data)
port_data.index = pd.to_datetime(port_data['Dates'])
port_data = port_data.drop(['Dates'], axis=1)
port_data['year'] = port_data.index.year


import math
joint_data = port_data.join(rf_1)
joint_data = joint_data.join(rf_2)
joint_data['rf'] = np.zeros(len(joint_data['RF']))
for i in range(len(joint_data['RF'])):
    if math.isnan(joint_data['RF Rate'][i]):
        joint_data['rf'][i] = joint_data['RF'][i]
    else:
        joint_data['rf'][i] = joint_data['RF Rate'][i]
joint_data = joint_data.drop(['RF','RF Rate'],axis=1)
joint_data = joint_data.dropna() 


rep_ret_y= joint_data['logret_rep'].groupby(joint_data['year']).sum()
dem_ret_y = joint_data['logret_dem'].groupby(joint_data['year']).sum()

fig1 = plt.figure(figsize =(6,4))
plt.title('Annual Return')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('Return',fontsize=13,fontweight='bold')
plt.plot(rep_ret_y,color='black', label = 'rep')
plt.plot(dem_ret_y,color='blue', label = 'demo')
plt.legend()
plt.show()


rep_std_y= joint_data['logret_rep'].groupby(joint_data['year']).std()
dem_std_y = joint_data['logret_dem'].groupby(joint_data['year']).std()
fig2 = plt.figure(figsize =(6,4))
plt.title('Annual Return')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('Volatility',fontsize=13,fontweight='bold')
plt.plot(rep_std_y,color='black', label = 'rep')
plt.plot(dem_std_y,color='blue', label = 'demo')
plt.legend()
plt.show()    


rf_y = joint_data['rf'].groupby(joint_data['year']).mean()
rep_sharpe = (rep_ret_y - rf_y)/rep_std_y
dem_sharpe = (dem_ret_y - rf_y)/dem_std_y
fig3 = plt.figure(figsize =(6,4))
plt.title('Sharpe Ratio')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('Sharpe Ratio',fontsize=13,fontweight='bold')
plt.plot(dem_sharpe,color='black', label = 'demo')
plt.plot(rep_sharpe,color='blue', label = 'rep')
plt.legend()
plt.show()  





#### Information Ratio(IR)
#Set the benchmark as S&P500
sp_y = joint_data['SPX Index'].groupby(joint_data['year']).sum()
rep_ir = (rep_ret_y - sp_y)/np.std(rep_ret_y - sp_y)
dem_ir = (dem_ret_y - sp_y)/np.std(dem_ret_y - sp_y)
fig4 = plt.figure(figsize =(6,4))
plt.title('Information Ratio')
plt.xlabel('Yr',fontsize=13,fontweight='bold')
plt.ylabel('Information Ratio',fontsize=13,fontweight='bold')
plt.plot(dem_ir,color='black', label = 'demo')
plt.plot(rep_ir,color='blue', label = 'rep')
plt.legend()
plt.show() 



##Beta
rep_beta = np.cov(rep_ret_y,sp_y)[0,1]/np.var(sp_y)
dem_beta = np.cov(dem_ret_y,sp_y)[0,1]/np.var(sp_y)


####### Cumulative return
rep_cum = np.cumsum(joint_data['logret_rep'])
dem_cum = np.cumsum(joint_data['logret_dem'])
fig5 = plt.figure(figsize =(6,4))
plt.title('Cumulative  Return')
plt.xlabel('time',fontsize=13,fontweight='bold')
plt.ylabel('Cum_ret',fontsize=13,fontweight='bold')
plt.plot(dem_cum,color='black', label = 'demo')
plt.plot(rep_cum,color='blue', label = 'rep')
plt.legend()
plt.show() 


######Normality
import statsmodels.api as sm
import scipy.stats as stats
import pylab



plt.hist(joint_data['logret_rep'],bins = 500)
plt.title("Histogram of rep portfolio log return",fontsize=13,fontweight='bold')
plt.xlabel("return")
plt.ylabel("times")
plt.show()
#Normality test
sm.qqplot(joint_data['logret_rep'],stats.t, fit=True, line='45')
print (stats.kstest(joint_data['logret_rep'],'norm',args=(np.mean(joint_data['logret_rep']),np.std(joint_data['logret_rep']))))
#Reject null hypothesis


plt.hist(joint_data['logret_dem'],bins = 500)
plt.title("Histogram of dem portfolio log return",fontsize=13,fontweight='bold')
plt.xlabel("return")
plt.ylabel("times")
plt.show()
#Normality test
sm.qqplot(joint_data['logret_dem'],stats.t, fit=True, line='45')
print (stats.kstest(joint_data['logret_dem'],'norm',args=(np.mean(joint_data['logret_dem']),np.std(joint_data['logret_dem']))))
#Reject null hypothesis





######Skewness
from scipy.stats import skew
rep_skew =  skew(joint_data['logret_rep'])
joint_data.groupby('year').skew()['logret_rep']
rep_skew
dem_skew =  skew(joint_data['logret_dem'])
joint_data.groupby('year').skew()['logret_dem']
dem_skew

######Kurtosis
from scipy.stats import kurtosis 
rep_kur =  kurtosis(joint_data['logret_rep'])
joint_data.groupby('year').apply(pd.DataFrame.kurt)['logret_rep']
rep_kur
dem_kur =  kurtosis(joint_data['logret_dem'])
joint_data.groupby('year').kurt()['logret_dem']
dem_kur

x = np.linspace(-0.1, 0.1, 100)
ax = plt.subplot()
y = joint_data['logret_rep'].pdf(x)






#####Density plot
sns.distplot(joint_data['logret_rep'], hist = False, kde = True,
             kde_kws = {'shade':True,'linewidth': 2},
             label = 'rep',color = 'red')
sns.distplot(joint_data['logret_dem'], hist = False, kde = True,
             kde_kws = {'shade':True,'linewidth': 2},
             label = 'demo',color = 'blue')
x = np.random.normal(np.mean(joint_data['logret_dem']), np.std(joint_data['logret_dem']), size=1000)
sns.distplot(x, hist = False, kde = True,
             kde_kws = {'shade':True,'linewidth': 2},
             label = 'normal',color = 'green')
plt.title("Distribution of log return",fontsize=13,fontweight='bold')
plt.xlabel("log-return")
plt.ylabel("frequency")
plt.show()

    
# Plot formatting
plt.legend()
#plt.legend(prop={'size': 16}, title = 'Log return')
plt.title('Density Plot with log return of two portfolios')
plt.xlabel('log-return')
plt.ylabel('Density')

#plot(density(joint_data['logret_rep'], bw=0.5))



####################################################################
import statsmodels.api as sm
def capm(p_ret, mkt_ret):
    '''Calculate the alpha and beta of the portfolio.'''
    y = p_ret
    x = mkt_ret
    # Add the intercept
    x = sm.add_constant(x)
#    regr = regression.linear_model.OLS(p_rtn,mkt_rtn).fit()
    #model= regression.linear_model.OLS(y,x).fit()
    model = sm.OLS(y,x)
    results = model.fit()
    alpha = np.array(results.params)[0]
    beta = np.array(results.params)[1]
    return (alpha,beta)


# CAPM
rep_alpha,rep_beta = capm(rep_ret_y - rf_y ,sp_y - rf_y)
dem_alpha,dem_beta = capm(dem_ret_y - rf_y,sp_y - rf_y)


####################################################################
ffm_data = pd.read_csv(r"/Users/Sofia/Desktop/IAQF/bmg_data/F-F_Research_Data_Factors_daily.csv")
ffm_data['Date'] = pd.to_datetime(ffm_data['Date'], format='%Y%m%d')
ffm_data['Date'] = pd.to_datetime(ffm_data['Date'])
ffm_data.index = pd.to_datetime(ffm_data['Date'])
ffm_data = pd.DataFrame(ffm_data)
#rf_1= rf_1.drop(['Mkt-RF','SMB','HML'],axis=1)
ffm_data= ffm_data.drop(['Date'],axis=1)

import math
ffm_data = ffm_data.dropna() 

ffm_data = pd.merge(port_data,ffm_data,left_index = True, right_index = True)
ffm_data = ffm_data.drop(columns=['Party','SPX Index','year'])

y = np.exp(ffm_data.iloc[:,1])-1 - ffm_data['RF']
x = ffm_data[['Mkt-RF','SMB','HML']]
x = sm.add_constant(x)
rep_model = sm.OLS(y,x).fit()
rep_model.params

y = np.exp(ffm_data.iloc[:,2])-1 - ffm_data['RF']
x = ffm_data[['Mkt-RF','SMB','HML']]
x = sm.add_constant(x)
dem_model = sm.OLS(y,x).fit()
dem_model.params


############################################################################


















