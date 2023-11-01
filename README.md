# CaliforniaMunicipalStabilityScores

## DESCRIPTION

Code, data to replicate the financial rankings of 25 California cities, as reported by https://municipalfinance.stanford.edu/

## INSTRUCTIONS


1. Update the data in the ```/int/cal_sample.xlsx```, making sure to include state_fips, county_fips, and place_fips for each city in the sample. Make sure that all fields in the file are covered, and expressed in millions USD (see description for ```/int/cal_sample.xlsx``` for more details)
2. (Optional) Change scoring, weighting parameters in ```/int/scoring_parameters.xlsx``` as desired
3. Add your local directory to the ```/src/sample_code_for_calpolicy.py```
4. Run the code and collect the output in either ```/final/final_alldata_cal_sample.xlsx``` or ```/final/final_alldata_cal_sample.pkl```

## GUIDE TO FILES

### ```/final```

#### ```final_alldata_cal_sample.xlsx```
Excel spreadsheet containing input and output data for 25 California cities between 2016 and 2022. Output data includes key performance indicators (KPIs), raw category scores, scaled category scores, and the final score. See ```/int/cal_sample.xlsx``` for more details regarding input data.

#### ```final_alldata_cal_sample.pkl```
Pickle dataset containing input and output data for 25 California cities between 2016 and 2022. Output data includes key performance indicators (KPIs), raw category scores, scaled category scores, and the final score. See ```/int/cal_sample.xlsx``` for more details regarding input data.

### ```/int```

#### ```cal_sample.xlsx```
Dataset containing input data for 25 California cities between 2016 and 2022. All data is expressed in millions USD (nominal). The following describe the sixteen input fields:

##### gf_reserves
General fund reserves

##### gf_transfers
General fund transfers to other government funds (expressed as a negative value)

##### gf_tot_exp
Total general fund expenditures

##### gov_lt_debt
Government-wide long-term obligations unrelated to pension obligations or other post-employment benefits (OPEB)

##### gov_tot_rev
Government-wide total revenue

##### gf_cash
General fund cash, cash equivalents on hand

##### gf_invest
General fund investments

##### gf_tot_liab
Total general fund liabilities

##### pension_npl
Net pension obligations

##### pension_tpl
Total pension obligations

##### pension_fnp
Pension fiduciary net position

##### pension_act_req_cont
Pension actuarially required contributions

##### opeb_fnp
OPEB fiduciary net position

##### opeb_tot_liab
OPEB total liability

##### gov_unrestricted_net_position
Government-wide unrestricted net position

#### ```scoring_parameters.xlsx```
The minimum and maximum scores for each category of ranking, and the cutoffs, in terms of the relevant KPI, to which the minimums and maximums apply. For more details, consult https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4565350

### ```/src```

#### ```sample_code_for_calpolicy.py```
A python code which takes in the scoring parameters and input data for 25 California cities (see ```/int```), processes these inputs to calculate the relevant financial scores, and outputs the result as both a STATA dataset and a Pickle dataset (see ```/final```)

### ```/tables_figures```
A set of ten figures which display the distribution of both the category scores and the KPIs used to calculate them across the 25 California cities in the sample.
