# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:32:52 2023

@author: shduffy
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset' ,'-sf')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

git_dir = "C:\\Users\\shduffy\\OneDrive - Stanford\\Documents\\CaliforniaMunicipalStabilityScores"

def sort_cols(df, explicit_cols , end_cols = []):
    othercols = [col for col in df.columns if not col in explicit_cols and not col in end_cols ]
    newdf = df[explicit_cols + othercols + end_cols].copy()
    return newdf

kpi_data = pd.read_excel(f"{git_dir}/int/scoring_parameters.xlsx")
kpi_pars = {}
for index, row in kpi_data.iterrows():
    entry_name = row['index']
    entry_data = {
        'min_cutoff': row['min_cutoff'],
        'min_score': row['min_score'],
        'max_cutoff': row['max_cutoff'],
        'max_score': row['max_score']
    }
    kpi_pars[entry_name] = entry_data

def set_score(ratio, parameters):
    global score
    slope = (parameters["max_score"]-parameters["min_score"])/(parameters["max_cutoff"]-parameters["min_cutoff"])
    intercept=parameters["min_score"]-slope*parameters["min_cutoff"]
    if ratio != "nan":
        score = max(parameters["max_score"],parameters["min_score"])
    if ratio >= parameters["min_cutoff"] and ratio <= parameters["max_cutoff"]:
        score=slope*ratio+intercept
    elif ratio < parameters["min_cutoff"]:
        score=parameters["min_score"]
    elif ratio > parameters["max_cutoff"]:
        score=parameters["max_score"]
    return score

diff_pars = {"0_1_months": -0.5, "1_2_months": -1, "2_3_months": -1.5, "3_4_months": -2, "4_5_months": -3, "5_6_months": -4, "6_7_months": -4.5, "7_8_months": -5, "8_9_months": -5.5}

def set_diff(ratio, indicator, parameters):
    global diff
    time = indicator*12
    if time<0 or time>9:
        diff = 0
    elif time>0 and time<=1:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["0_1_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>1 and time<=2:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["1_2_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>2 and time<=3:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["2_3_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>3 and time<=4:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["3_4_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>4 and time<=5:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["4_5_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>5 and time<=6:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["5_6_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>6 and time<=7:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["6_7_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>7 and time<=8:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["7_8_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    elif time>8 and time<=9:
        temp_pars = {"min_cutoff" :  -0.2, "min_score" : parameters["8_9_months"], "max_cutoff": .2, "max_score" : 0}
        diff = set_score(ratio, temp_pars)
    else:
        diff = 0 
    return diff

def find_change(old,new,length):
    if old>0 and new>0:
        change = ((new/old) ** (1/length))-1
    elif old<0 and new>0:
        shift = abs(old) + .2 * ((abs(old)*new) ** (1/2))
        old = old + shift
        new = new + shift
        change = ((new/old) ** (1/length))-1
    elif old>0 and new<0:
        shift = abs(new) + .2 * ((abs(new)*old) ** (1/2))
        old = old + shift
        new = new + shift
        change = ((new/old) ** (1/length))-1
    elif old<0 and new<0:
        change = -(((new/old) ** (1/length))-1)
    elif old==0 and new!=0:
        change = abs(new)/new
    elif old!=0 and new==0:
        change = abs(old)/old
    else:
        change = np.nan
    return change

# =============================================================================
# LOAD DATA    
# =============================================================================
 
data = pd.read_excel(f"{git_dir}/int/cal_sample.xlsx", dtype = {'state_fips': str, 'county_fips': str, 'place_fips': str})

# =============================================================================
# KPI
# =============================================================================


kpi_df = data[['state_fips', 'county_fips', 'place_fips', 'year']].copy()

    # GF RESERVES #

kpi_dict = {"gfreserves" : ['year', 'state_fips', 'county_fips', 'place_fips',
                             "gf_reserves", "gf_transfers",
                             'gf_tot_exp']}

selection = data[kpi_dict["gfreserves"]].copy()
selection['kpi_gfreserves'] = selection['gf_reserves'] / (selection["gf_tot_exp"]-selection["gf_transfers"])

selection.sort_values(by = ['state_fips', 'county_fips', 'place_fips', 'year'], inplace = True)
selection["unreserved_fund_3"] = selection.groupby(['state_fips', 'county_fips', 'place_fips'])["gf_reserves"].shift(3)
selection["unreserved_fund_2"] = selection.groupby(['state_fips', 'county_fips', 'place_fips'])["gf_reserves"].shift(2)
selection["unreserved_fund_1"] = selection.groupby(['state_fips', 'county_fips', 'place_fips'])["gf_reserves"].shift(1)
selection["d_unreserved_fund_3"] = selection.apply(lambda x: find_change(x['unreserved_fund_3'],x['gf_reserves'],3), axis=1)
selection["d_unreserved_fund_3"].replace([np.inf, -np.inf], np.nan, inplace=True)
selection["d_unreserved_fund_2"] = selection.apply(lambda x: find_change(x['unreserved_fund_2'],x['gf_reserves'],2), axis=1)
selection["d_unreserved_fund_2"].replace([np.inf, -np.inf], np.nan, inplace=True)
selection["d_unreserved_fund_1"] = selection.apply(lambda x: find_change(x['unreserved_fund_1'],x['gf_reserves'],1), axis=1)
selection["d_unreserved_fund_1"].replace([np.inf, -np.inf], np.nan, inplace=True)
selection.loc[selection["d_unreserved_fund_3"].isna(), "d_unreserved_fund_3"] = selection.loc[selection["d_unreserved_fund_3"].isna(), "d_unreserved_fund_2"]
selection.loc[selection["d_unreserved_fund_3"].isna(), "d_unreserved_fund_3"] = selection.loc[selection["d_unreserved_fund_3"].isna(), "d_unreserved_fund_1"]
selection = selection[['state_fips', 'county_fips', 'place_fips','year', "d_unreserved_fund_3",'kpi_gfreserves']].copy()
selection.rename(columns = {"d_unreserved_fund_3": "kpi_unrsrvdgrwth"}, inplace=True)

selection.loc[(abs(selection['kpi_gfreserves']) < 0),'kpi_gfreserves'] = 0
selection.loc[(abs(selection['kpi_gfreserves']) > 5),'kpi_gfreserves'] = np.nan
selection.loc[(abs(selection['kpi_unrsrvdgrwth']) > 5),'kpi_unrsrvdgrwth'] = np.nan

selection.loc[(selection['kpi_gfreserves'].notna()), "main"] = selection['kpi_gfreserves'].apply(lambda x: set_score(x, kpi_pars["gfreserves"]))
selection.loc[(selection['kpi_unrsrvdgrwth'].notna()) & (selection['kpi_gfreserves'].notna()), "diff"] = selection.apply(lambda x: set_diff(x['kpi_unrsrvdgrwth'], x['kpi_gfreserves'],diff_pars), axis=1)
selection.loc[(selection['main'].notna()) & (selection['diff'].notna()), 'gfreserves_score'] = np.maximum((selection['main'] + selection['diff']), 0)
selection.loc[(selection['main'].notna()) & (selection['diff'].isna()), 'gfreserves_score'] = np.maximum((selection['main']), 0)

selection["kpi_gfreserves"].replace([np.inf, -np.inf], np.nan, inplace=True)
selection["kpi_unrsrvdgrwth"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['gfreserves_score'].isna(), "year"].value_counts())

if pd.isna(selection['kpi_unrsrvdgrwth']).all():
    fig, axes = plt.subplots(1,2)
    axes[0].hist(selection['kpi_gfreserves'], bins = 30, edgecolor='black', linewidth=1)
    axes[0].set_title("Reserves over exp.")
    axes[1].hist(selection['gfreserves_score'], bins = 30, edgecolor='black', linewidth=1)
    axes[1].set_title("Score (higher is better)")
    fig.suptitle("GF Reserves")
    plt.savefig(f"{git_dir}/tables_figures/kpi_hist_gfreserves_state.png")
else:
    fig, axes = plt.subplots(1,3)
    axes[0].hist(selection['kpi_gfreserves'], bins = 30, edgecolor='black', linewidth=1)
    axes[0].set_title("Reserves over exp.")
    axes[1].hist(selection['kpi_unrsrvdgrwth'], bins = 30, edgecolor='black', linewidth=1)
    axes[1].set_title("Reserves growth")
    axes[2].hist(selection['gfreserves_score'], bins = 30, edgecolor='black', linewidth=1)
    axes[2].set_title("Score (higher is better)")
    fig.suptitle("GF Reserves")
    plt.savefig(f"{git_dir}/tables_figures/kpi_hist_gfreserves_state.png")

selection.loc[(selection['gfreserves_score'].notna()), 'gfreserves_tot'] = max(kpi_pars["gfreserves"]["max_score"],kpi_pars["gfreserves"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips' ,'year', 'kpi_gfreserves', 'kpi_unrsrvdgrwth', 'gfreserves_score','gfreserves_tot']], on = ['state_fips', 'county_fips', 'place_fips' ,'year'], how = 'left')

    # DEBT BURDEN #

kpi_dict.update({"debtburden" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                 'gov_lt_debt', 'gov_tot_rev']})
selection = data[ kpi_dict["debtburden"]].copy()
selection["kpi_debtburden"] = (selection['gov_lt_debt']) / (selection['gov_tot_rev'])
selection['debtburden_score'] = np.nan
selection["debtburden_score"] = selection['kpi_debtburden'].apply(lambda x: set_score(x, kpi_pars["debtburden"]))

selection["kpi_debtburden"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['debtburden_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_debtburden'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("Debt over total gov rev.")
axes[1].hist(selection['debtburden_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Debt Burden")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_debtburden_state.png")

selection.loc[(selection['debtburden_score'].notna()), 'debtburden_tot'] = max(kpi_pars["debtburden"]["max_score"],kpi_pars["debtburden"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips' ,'year', 'kpi_debtburden','debtburden_score','debtburden_tot']], on = ['state_fips', 'county_fips', 'place_fips' ,'year'], how = 'left')

    # LIQUIDITY #
    
kpi_dict.update({"liquidity" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                'gf_cash', 'gf_invest', 'gf_tot_liab']})

selection = data[ kpi_dict["liquidity"]].copy()
selection["kpi_liquidity"] = selection[['gf_cash','gf_invest']].sum(axis=1) / selection['gf_tot_liab']
selection['liquidity_score'] = np.nan
selection["liquidity_score"] = selection['kpi_liquidity'].apply(lambda x: set_score(x, kpi_pars["liquidity"]))

selection["kpi_liquidity"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['liquidity_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_liquidity'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("Cash over liabilities")
axes[1].hist(selection['liquidity_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Liquidity")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_liquidity_state.png")

selection.loc[(selection['liquidity_score'].notna()), 'liquidity_tot'] = max(kpi_pars["liquidity"]["max_score"],kpi_pars["liquidity"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips' ,'year', 'kpi_liquidity', 'liquidity_score', 'liquidity_tot']], on = ['state_fips', 'county_fips', 'place_fips' ,'year'], how = 'left')

    # REVENUE TRENDS #

kpi_dict.update({"revenuetrend" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                   "gf_tot_rev"]})
selection = data[ kpi_dict["revenuetrend"]].copy()
selection["ln_gf_tot_rev"] = np.log(selection["gf_tot_rev"])
selection["ln_gf_tot_rev"].replace([np.inf, -np.inf], np.nan, inplace=True)
selection.sort_values(by = ['state_fips', 'county_fips', 'place_fips', 'year'], inplace = True)

selection["d3ln_gf_tot_rev"] = np.exp((selection['ln_gf_tot_rev'] - selection.groupby(['state_fips', 'county_fips', 'place_fips'])["ln_gf_tot_rev"].shift(3))/3) - 1
selection["d2ln_gf_tot_rev"] = np.exp((selection['ln_gf_tot_rev'] - selection.groupby(['state_fips', 'county_fips', 'place_fips'])["ln_gf_tot_rev"].shift(2))/2) - 1
selection["d1ln_gf_tot_rev"] = np.exp((selection['ln_gf_tot_rev'] - selection.groupby(['state_fips', 'county_fips', 'place_fips'])["ln_gf_tot_rev"].shift(1))/1) - 1
selection.loc[selection["d3ln_gf_tot_rev"].isna(), "d3ln_gf_tot_rev"] = selection.loc[selection["d3ln_gf_tot_rev"].isna(), "d2ln_gf_tot_rev"]
selection.loc[selection["d3ln_gf_tot_rev"].isna(), "d3ln_gf_tot_rev"] = selection.loc[selection["d3ln_gf_tot_rev"].isna(), "d1ln_gf_tot_rev"]
selection = selection[['state_fips', 'county_fips', 'place_fips','year', "d3ln_gf_tot_rev"]].copy()
selection.rename(columns = {"d3ln_gf_tot_rev": "kpi_revenuegrowth"}, inplace=True)
selection.loc[(abs(selection['kpi_revenuegrowth']) > .5),'kpi_revenuegrowth'] = np.nan
selection['revenuegrowth_score'] = np.nan
selection.loc[selection['kpi_revenuegrowth'].notna(), "revenuegrowth_score"] = selection['kpi_revenuegrowth'].apply(lambda x: set_score(x, kpi_pars["revenuegrowth"]))

selection["kpi_revenuegrowth"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['revenuegrowth_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_revenuegrowth'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("Revenue Growth")
axes[1].hist(selection['revenuegrowth_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Rev Growth")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_revgrowth_state.png")

selection.loc[(selection['revenuegrowth_score'].notna()), 'revenuegrowth_tot'] = max(kpi_pars["revenuegrowth"]["max_score"],kpi_pars["revenuegrowth"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips' ,'year', 'kpi_revenuegrowth','revenuegrowth_score', 'revenuegrowth_tot']], on = ['state_fips', 'county_fips', 'place_fips' ,'year'], how = 'left')

    # PENSION OBLIGATIONS #

kpi_dict.update({"pensionobligations" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                         'pension_npl','gov_tot_rev']})
selection = data[ kpi_dict["pensionobligations"]].copy()

selection["kpi_pensionobligations"] =  - selection["pension_npl"] / (selection["gov_tot_rev"])
selection.loc[(abs(selection['kpi_pensionobligations']) > 10),'kpi_pensionobligations'] = np.nan
selection['pensionobligations_score'] = np.nan
selection["pensionobligations_score"] = selection['kpi_pensionobligations'].apply(lambda x: set_score(x, kpi_pars["pensionobligations"]))

selection['kpi_pensionobligations'].describe()

selection["kpi_pensionobligations"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['pensionobligations_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_pensionobligations'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("NPL over total gov rev.")
axes[1].hist(selection['pensionobligations_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Pension Ob")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_pensionob_state.png")

selection.loc[(selection['pensionobligations_score'].notna()), 'pensionobligations_tot'] = max(kpi_pars["pensionobligations"]["max_score"],kpi_pars["pensionobligations"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips', 'year', 'kpi_pensionobligations', 'pensionobligations_score','pensionobligations_tot']], on = ['state_fips', 'county_fips', 'place_fips', 'year'], how = 'left')

    # PENSION FUNDING #

kpi_dict.update({"pensionfunding" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                     'pension_tpl', 'pension_fnp']})
selection = data[kpi_dict["pensionfunding"]].copy()
selection["kpi_pensionfunding"] = selection['pension_fnp'] / selection['pension_tpl']
selection['pensionfunding_score'] = np.nan
selection["pensionfunding_score"] = selection['kpi_pensionfunding'].apply(lambda x: set_score(x, kpi_pars["pensionfunding"]))

selection["kpi_pensionfunding"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['pensionfunding_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_pensionfunding'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("FNP/TPL")
axes[1].hist(selection['pensionfunding_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Pension Funding")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_pensionfund_state.png")

selection.loc[(selection['pensionfunding_score'].notna()), 'pensionfunding_tot'] = max(kpi_pars["pensionfunding"]["max_score"],kpi_pars["pensionfunding"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips', 'year', 'kpi_pensionfunding','pensionfunding_score', 'pensionfunding_tot']], on = ['state_fips', 'county_fips', 'place_fips', 'year'], how = 'left')

    # PENSION COST #

kpi_dict.update({"pensioncost" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                  'pension_act_req_cont','gov_tot_rev']})
selection = data[kpi_dict["pensioncost"]].copy()
selection["kpi_pensionadc"] = selection["pension_act_req_cont"] / (selection["gov_tot_rev"])
selection.loc[(abs(selection['kpi_pensionadc']) > 1),'kpi_pensionadc'] = np.nan
selection['pensionadc_score'] = np.nan
selection["pensionadc_score"] = selection['kpi_pensionadc'].apply(lambda x: set_score(x, kpi_pars["pensionadc"]))

selection['kpi_pensionadc'].describe()

selection["kpi_pensionadc"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['pensionadc_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_pensionadc'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("ADC over total gov rev.")
axes[1].hist(selection['pensionadc_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Pension Cost")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_pensionadc_state.png")

selection.loc[(selection['pensionadc_score'].notna()), 'pensionadc_tot'] = max(kpi_pars["pensionadc"]["max_score"],kpi_pars["pensionadc"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips', 'year', 'kpi_pensionadc','pensionadc_score', 'pensionadc_tot']], on = ['state_fips', 'county_fips', 'place_fips', 'year'], how = 'left')

    # OPEB OBLIGATIONS AND OPEB FUNDING #

kpi_dict.update({"opebobligation" : ['year', 'state_fips', 'county_fips', 'place_fips',
                                     'opeb_fnp', 'opeb_tot_liab', 'gov_tot_rev']})

selection = data[kpi_dict["opebobligation"]].copy()
selection["netopebobligation"] = - (selection["opeb_fnp"] - selection["opeb_tot_liab"])
selection.loc[selection["netopebobligation"] == 0, "netopebobligation"] = np.nan
selection["kpi_opebobligation"] = selection["netopebobligation"] / (selection["gov_tot_rev"])
selection['opebobligation_score'] = np.nan
selection["opebobligation_score"] = selection['kpi_opebobligation'].apply(lambda x: set_score(x, kpi_pars["opebobligation"]))

selection["kpi_opebobligation"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['opebobligation_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_opebobligation'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("OPEB liability over total gov rev.")
axes[1].hist(selection['opebobligation_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("OPEB ob")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_opebob_state.png")

selection.loc[(selection['opebobligation_score'].notna()), 'opebobligation_tot'] = max(kpi_pars["opebobligation"]["max_score"],kpi_pars["opebobligation"]["min_score"])

selection["kpi_opebfunding"] = selection["opeb_fnp"] / selection["opeb_tot_liab"]
selection['opebfunding_score'] = np.nan
selection["opebfunding_score"] = selection['kpi_opebfunding'].apply(lambda x: set_score(x, kpi_pars["opebfunding"]))

selection["kpi_opebfunding"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['opebfunding_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_opebfunding'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("OPEB asset over OPEB liability")
axes[1].hist(selection['opebfunding_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("OPEB funding")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_opebfund_state.png")

selection.loc[(selection['opebfunding_score'].notna()), 'opebfunding_tot'] = max(kpi_pars["opebfunding"]["max_score"],kpi_pars["opebfunding"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips', 'year', 'kpi_opebobligation','opebobligation_score', 'opebobligation_tot','kpi_opebfunding','opebfunding_score', 'opebfunding_tot']], on = ['state_fips', 'county_fips', 'place_fips', 'year'], how = 'left')    

    # UNRESTRICTED NET POSITION #

kpi_dict.update({"networth" : ['year', 'state_fips', 'county_fips', 'place_fips',
                               'gov_unrestricted_net_position','gov_tot_rev']})

selection = data[kpi_dict["networth"]].copy()
selection["kpi_unrestrictednetassets"] = selection["gov_unrestricted_net_position"] / (selection["gov_tot_rev"])
selection['unrestrictednetassets_score'] = np.nan
selection["unrestrictednetassets_score"] = selection['kpi_unrestrictednetassets'].apply(lambda x: set_score(x, kpi_pars["unrestrictednetassets"]))

selection["kpi_unrestrictednetassets"].replace([np.inf, -np.inf], np.nan, inplace=True)

print(selection.loc[~selection['unrestrictednetassets_score'].isna(), "year"].value_counts())

fig, axes = plt.subplots(1,2)
axes[0].hist(selection['kpi_unrestrictednetassets'], bins = 30, edgecolor='black', linewidth=1)
axes[0].set_title("UNA over governmentwide revenue")
axes[1].hist(selection['unrestrictednetassets_score'], bins = 30, edgecolor='black', linewidth=1)
axes[1].set_title("Score (higher is better)")
fig.suptitle("Unrestricted Net Assets")
plt.savefig(f"{git_dir}/tables_figures/kpi_hist_unrestrictednetassets_state.png")

selection.loc[(selection['unrestrictednetassets_score'].notna()), 'unrestrictednetassets_tot'] = max(kpi_pars["unrestrictednetassets"]["max_score"],kpi_pars["unrestrictednetassets"]["min_score"])

kpi_df = kpi_df.merge(selection[['state_fips', 'county_fips', 'place_fips' ,'year', 'kpi_unrestrictednetassets', 'unrestrictednetassets_score','unrestrictednetassets_tot']], on = ['state_fips', 'county_fips', 'place_fips' ,'year'], how = 'left')

    # RESCALE SCORES, CREATE FINAL SCORE #

kpi_df['scale_factor_pension'] = (max(kpi_pars["pensionobligations"]["max_score"],kpi_pars["pensionobligations"]["min_score"]) + 
                                  max(kpi_pars["pensionfunding"]["max_score"],kpi_pars["pensionfunding"]["min_score"]) + 
                                  max(kpi_pars["pensionadc"]["max_score"],kpi_pars["pensionadc"]["min_score"])) / (kpi_df[['pensionobligations_tot','pensionfunding_tot','pensionadc_tot']].sum(axis=1))
kpi_df.loc[kpi_df['scale_factor_pension'] == np.inf, 'scale_factor_pension'] = np.nan
kpi_df['scale_factor_opeb'] = (max(kpi_pars["opebobligation"]["max_score"],kpi_pars["opebobligation"]["min_score"]) + 
                               max(kpi_pars["opebfunding"]["max_score"],kpi_pars["opebfunding"]["min_score"])) / (kpi_df[['opebobligation_tot','opebfunding_tot']].sum(axis=1))
kpi_df.loc[kpi_df['scale_factor_opeb'] == np.inf, 'scale_factor_opeb'] = np.nan

kpi_df['scale_pensionobligations_tot'] = kpi_df['pensionobligations_tot'] * kpi_df['scale_factor_pension']
kpi_df['scale_pensionfunding_tot'] = kpi_df['pensionfunding_tot'] * kpi_df['scale_factor_pension']
kpi_df['scale_pensionadc_tot'] = kpi_df['pensionadc_tot'] * kpi_df['scale_factor_pension']
kpi_df['scale_opebobligation_tot'] = kpi_df['opebobligation_tot'] * kpi_df['scale_factor_opeb']
kpi_df['scale_opebfunding_tot'] = kpi_df['opebfunding_tot'] * kpi_df['scale_factor_opeb']

kpi_df['d_gfreserves'] = np.where(pd.isnull(kpi_df['gfreserves_tot']), 0, 1)
kpi_df['d_debtburden'] = np.where(pd.isnull(kpi_df['debtburden_tot']), 0, 1)
kpi_df['d_liquidity'] = np.where(pd.isnull(kpi_df['liquidity_tot']), 0, 1)
kpi_df['d_revenuegrowth'] = np.where(pd.isnull(kpi_df['revenuegrowth_tot']), 0, 1)
kpi_df['d_pension'] = np.where(pd.isnull(kpi_df['scale_factor_pension']), 0, 1)
kpi_df['d_opeb'] = np.where(pd.isnull(kpi_df['scale_factor_opeb']), 0, 1)
kpi_df['d_unrestrictednetassets'] = np.where(pd.isnull(kpi_df['unrestrictednetassets_tot']), 0, 1)
kpi_df['d_tot'] = kpi_df[['d_gfreserves','d_debtburden','d_liquidity','d_revenuegrowth','d_pension','d_opeb','d_unrestrictednetassets']].sum(axis=1)

kpi_df['scale_factor_tot'] = 100 / (kpi_df[['gfreserves_tot','debtburden_tot','liquidity_tot','revenuegrowth_tot','scale_pensionobligations_tot','scale_pensionfunding_tot',
                                            'scale_pensionadc_tot','scale_opebobligation_tot','scale_opebfunding_tot','unrestrictednetassets_tot']].sum(axis=1))
kpi_df.loc[kpi_df['scale_factor_tot'] == np.inf, 'scale_factor_tot'] = np.nan

kpi_df['scale_gfreserves_score'] = kpi_df['gfreserves_score'] * kpi_df['scale_factor_tot']
kpi_df['scale_debtburden_score'] = kpi_df['debtburden_score'] * kpi_df['scale_factor_tot']
kpi_df['scale_liquidity_score'] = kpi_df['liquidity_score'] * kpi_df['scale_factor_tot']
kpi_df['scale_revenuegrowth_score'] = kpi_df['revenuegrowth_score'] * kpi_df['scale_factor_tot']
kpi_df['scale_pensionobligations_score'] = kpi_df['pensionobligations_score'] * kpi_df['scale_factor_pension'] * kpi_df['scale_factor_tot']
kpi_df['scale_pensionfunding_score'] = kpi_df['pensionfunding_score'] * kpi_df['scale_factor_pension'] * kpi_df['scale_factor_tot']
kpi_df['scale_pensionadc_score'] = kpi_df['pensionadc_score'] * kpi_df['scale_factor_pension'] * kpi_df['scale_factor_tot']
kpi_df['scale_opebobligation_score'] = kpi_df['opebobligation_score'] * kpi_df['scale_factor_opeb'] * kpi_df['scale_factor_tot']
kpi_df['scale_opebfunding_score'] = kpi_df['opebfunding_score'] * kpi_df['scale_factor_opeb'] * kpi_df['scale_factor_tot']
kpi_df['scale_unrestrictednetassets_score'] = kpi_df['unrestrictednetassets_score'] * kpi_df['scale_factor_tot']

kpi_df['final_score'] = (kpi_df[['scale_gfreserves_score','scale_debtburden_score','scale_liquidity_score','scale_revenuegrowth_score','scale_pensionobligations_score',
                                 'scale_pensionfunding_score','scale_pensionadc_score','scale_opebobligation_score','scale_opebfunding_score','scale_unrestrictednetassets_score']].sum(axis=1))
kpi_df.loc[kpi_df['d_tot'] < 5, 'final_score'] = np.nan

kpi_df['temp1'] = kpi_df['year'].where(kpi_df['final_score'].notnull())
kpi_df['temp2'] = kpi_df.groupby(['state_fips', 'county_fips', 'place_fips'])['temp1'].transform('min')
kpi_df.loc[kpi_df['year'] == kpi_df['temp2'], 'final_score'] = np.nan

kpi_df.drop(columns = ['gfreserves_tot','debtburden_tot','liquidity_tot','revenuegrowth_tot','pensionobligations_tot','scale_pensionobligations_tot',
                                        'pensionfunding_tot','scale_pensionfunding_tot','pensionadc_tot','scale_pensionadc_tot','opebobligation_tot',
                                        'scale_opebobligation_tot','opebfunding_tot','scale_opebfunding_tot','unrestrictednetassets_tot',
                                        'scale_factor_pension','scale_factor_opeb','scale_factor_tot','d_gfreserves','d_debtburden','d_liquidity','d_revenuegrowth',
                                        'd_pension','d_opeb','d_unrestrictednetassets','d_tot','temp1','temp2'], inplace=True)

data = data.merge(kpi_df, on = ['state_fips', 'county_fips', 'place_fips', "year"], how = "left", indicator = True)
data["_merge"].value_counts()
data.drop(columns = ["_merge"], inplace=True)

# =============================================================================
# EXPORT DATA
# =============================================================================

identifiers = ['year','name','state_fips','county_fips','place_fips','state_code']

gov_kpis = ['kpi_gfreserves', 'kpi_unrsrvdgrwth', 'kpi_debtburden',
            'kpi_liquidity','kpi_revenuegrowth', 'kpi_pensionobligations',
            'kpi_pensionfunding', 'kpi_pensionadc', 'kpi_opebobligation',
            'kpi_opebfunding', 'kpi_unrestrictednetassets']

scale_gov_score = ['scale_gfreserves_score','scale_debtburden_score','scale_liquidity_score',
             'scale_revenuegrowth_score','scale_pensionobligations_score','scale_pensionfunding_score',
             'scale_pensionadc_score','scale_opebobligation_score','scale_opebfunding_score',
             'scale_unrestrictednetassets_score','final_score']

reg_gov_score = ['gfreserves_score','debtburden_score','liquidity_score','revenuegrowth_score',
                 'pensionobligations_score','pensionfunding_score','pensionadc_score','opebobligation_score',
                 'opebfunding_score','unrestrictednetassets_score']

input_data = ['gf_reserves','gf_transfers','gf_tot_exp','gov_lt_debt','gov_tot_rev','gf_tot_rev',
              'gf_cash','gf_invest','gf_tot_liab','pension_npl','pension_tpl','pension_fnp',
              'pension_act_req_cont','opeb_fnp','opeb_tot_liab','gov_unrestricted_net_position']

data = data[identifiers + input_data + gov_kpis + reg_gov_score + scale_gov_score].copy()

data = sort_cols(data, identifiers + input_data + gov_kpis + reg_gov_score, scale_gov_score)

data.to_pickle(f"{git_dir}/final/final_alldata_cal_sample.pkl")
data.to_excel(f"{git_dir}/final/final_alldata_cal_sample.xlsx", index=False)