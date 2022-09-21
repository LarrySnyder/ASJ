# Simulation of PredPol predictive policing system.
#
# Larry Snyder, Lehigh University
#
# Main ideas in this module are adapted from Lum and Isaac, "To Predict and Serve?,"
# Significance Magazine, 14-19, October 2016 (https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1740-9713.2016.00960.x)
# and their associated GitHub repo at https://github.com/arun-ramamurthy/pred-pol.

# Imports.
import pandas as pd
import numpy as np
import copy


# --- PREDPOL CALCULATIONS --- #

# TODO: clean up

def lower_tri_ij(dim):
    ''' tril_indices makes lower-triangular indexing easy '''
    ii, jj = np.tril_indices(dim)
    for i, j in zip(ii, jj):
        yield i, j


def lambdafun(t, mu, theta, omega):
    days_til_last = [(t[-1] - ti).days for ti in t[:-1]]
    epart = sum([np.exp(-omega * d) for d in days_til_last])
    return mu + theta * omega * epart


def calc_tij(data):
    assert all([len(n) > 0 for n in data.values()])
    tij = dict()
    for n, events in data.items():
        tij_len = len(events) - 1
        tij[n] = np.zeros((tij_len, tij_len))
        for i, j in lower_tri_ij(tij_len):
            td = (events[i + 1] - events[j]).days
            tij[n][i, j] = td
    return tij


def calc_pij(tij, theta, omega):
    pij = dict()
    for n in tij:
        pij[n] = np.zeros(tij[n].shape)
        for i, j in lower_tri_ij(tij[n].shape[0]):
            if tij[n][i, j] > 0:
                e_part = np.exp(-omega * tij[n][i, j])
                pij[n][i, j] = e_part * theta * omega
    return pij


def estep(data, mu, theta, omega, tij):
    # t these asserts are fast and document the common structure
    assert data.keys() == mu.keys()
    assert data.keys() == tij.keys()
    pj = dict()
    pij = calc_pij(tij, theta, omega)
    for n in data:
        # should possibly append a 1 to the front of this
        denom = mu[n] + pij[n].sum(axis=1)
        pj[n] = mu[n] / denom
        pij[n] = pij[n] / denom
        pj[n] = np.append(pj[n], 1)
    return pij, pj


# t iterating over a dict means over keys unless otherwise specified
def mstep(pij, pj, tij, data, T):
    # t these asserts are fast and document the common structure
    assert data.keys() == pij.keys()
    assert data.keys() == tij.keys()
    assert data.keys() == pj.keys()
    total_events = sum([len(v) for v in data.values()])
    # double sum: arrays over bins; np.sum whole array at once
    sum_pijs = sum([np.sum(pij[n]) for n in pij])
    denom = sum([np.sum(pij[n] * tij[n]) for n in pij])
    omega = sum_pijs / denom
    theta = sum_pijs / total_events
    mu = dict((n, sum(pj[n]) / T) for n in pj)
    # mu = np.ones(num_bins)*sum(mu)  #this is what the paper says to do but
    # this forces the "background rate" to be the same everywhere,
    return omega, theta, mu


def runEM(data, T, pred_date, k=20,
          theta_init=1, omega_init=1, mu_init=1,
          tol1=.00001, tol2=.00001, tol3=.0001):
    num_bins = len(data)
    theta = theta_init
    mu = dict((key, mu_init) for key in data)
    omega = omega_init
    omega_last = 10 + omega
    theta_last = 10 + theta
    mu_last = dict((key, mu_init + 10) for key in data)
    k = min(num_bins, k)
    tij = calc_tij(data)

    while(abs(omega - omega_last) > tol1 and
          abs(theta - theta_last) > tol2 and
          sum([abs(mu[n] - mu_last[n]) for n in mu]) > tol3):
        omega_last = omega
        theta_last = theta
        mu_last = copy.deepcopy(mu)
        pij, pj = estep(data, mu, theta, omega, tij)
        omega, theta, mu = mstep(pij, pj, tij, data, T)

        # deprecate?
        # I did this when I was debugging so that it wouldn't run away
        # if omega > T * 1000:
        #    omega = omega_last
        assert omega < T * 1000

    # get conditional intensity for selected parameters
    # need to add on latest date
    for n in data:
        data[n] = np.append(data[n], pred_date)

    # todo(KL): everything else is a dict[n], shouldn't rates be a dict?
    rates = [lambdafun(data[n], mu[n], theta, omega) for n in data]
    sorted_keys = [key for (rate, key)
                   in sorted(zip(rates, data), reverse=True)]
    return rates, sorted_keys[0:k], omega, theta


# --- SIMULATION --- #

def prepare_data_for_predpol(data, start_date, end_date):
	"""Prepare data for PredPol simulation."""

	# Remove records that are not in the desired time window.
	data = data[(data.DateTime >= start_date) & (data.DateTime < end_date)]

	# Build output dict. (Keys are map bins, values are data values.)
	pp_dict = dict((int(i), []) for i in set(data.bin))
	for i in data.index:
		pp_dict[int(data.bin[i])].append(data['DateTime'].loc[i])

	# Drop empty rows.
	keys_to_drop = []
	pp_dict = {n: sorted(v) for n, v in pp_dict.items()}
	for n in pp_dict.keys():
		if len(pp_dict[n]) < 1:
			keys_to_drop.append(n)
	for key in keys_to_drop:
		pp_dict.pop(key, None)

	return pp_dict


def load_predpol_data(options):
	"""Load data from CSV file."""

	# Read data.
	data = pd.read_csv(options["drug_crimes_with_bins"])
	# Specify column names.
	data.columns = ['rownum', 'bin', 'OCCURRED', 'LAG']
	# Remove data with null bins.
	data = data[pd.notnull(data['bin'])]
	# Format dates.
	data['DateTime'] = pd.to_datetime(data.OCCURRED, format = '%m/%d/%y')

	return data


def run_predpol_simulation(data, options):
	"""Run the PredPol simulation."""

	# Adjust output filenames.
	if options["add_crimes_logical"]:
		options["predictions"] += '_add_' + str(int(options["percent_increase*100"])) + 'percent.csv'
		options["observed"] += '_add_' + str(int(options["percent_increase*100"])) + 'percent.csv'
	else:
		options["predictions"] += '.csv'
		options["observed"] += '.csv'

	# Determine number of bins needed for dataframe.
	max_bin = int(max(data.bin))
	print(f"max_bin = {max_bin}")

	# Remove records that are not in the desired time window.
	# TODO: don't we do this in prepare_data_for_predpol() too?
	print(f"len(data) = {len(data)} before removing records by date")
	data = data[(data.DateTime >= options["global_start"]) & (data.DateTime < options["global_end"])]
	print(f"len(data) = {len(data)} after removing records by date")

	# Determine number of predictions (= # days in data minus predpol window).
	num_predictions = (options["global_end"] - options["global_start"]).days - options["predpol_window"]

	# Initialize results_rates dataframe (for PredPol predictions).
	results_rates = pd.DataFrame()
	results_rates['bin'] = range(1, max_bin + 1)
	results_rates = results_rates.set_index(['bin'])

	# Initialize results_num_crimes dataframe (for crimes observed).
	results_num_crimes = pd.DataFrame()
	results_num_crimes['bin'] = range(1, max_bin + 1)
	results_num_crimes = results_num_crimes.set_index(['bin'])

	# Main loop.
	for i in range(num_predictions):

		print(f"Simulating day {i}")

		# Determine dates of PredPol window.
		start_date = options["global_start"] + pd.DateOffset(i)
		end_date = options["global_start"] + pd.DateOffset(i + options["predpol_window"])

		# Prepare data for PredPol.
		pp_dict = prepare_data_for_predpol(data, start_date, end_date)

		# Run PredPol calculations.
		rates, o, omega, theta = runEM(pp_dict, options["predpol_window"], end_date)

		# Save rates.
		str_date = str(end_date).split(' ')[0]
		results_rates[str_date] = 0
		keys = list(pp_dict.keys())
		results_rates.loc[keys, str_date] = rates

		# Add crimes, if that's what we're doing.
		# TODO

		# Record total number of crimes on this day.
		crime_today = pd.value_counts(data[data.DateTime==end_date].bin)
		results_num_crimes[str_date] = 0
		results_num_crimes.loc[crime_today.index, str_date] = list(crime_today)

		# Remove old data to speed things up.
		data = data[(data.DateTime >= start_date)]

	# Write results.
	results_rates.to_csv(options["predictions"])
	results_num_crimes.to_csv(options["observed"])


# --- CODE TO RUN THE SIMULATION --- #

# Set up simulation options.
options = {}
options["drug_crimes_with_bins"] = "predpol/drug_crimes_with_bins.csv"	# path to input file
options["global_start"] = pd.to_datetime("2010/07/01")					# simulation start date
options["global_end"] = pd.to_datetime("2011/12/31")					# simulation end date
options["observed"] = "predpol/predpol_drug_observed"					# path to "observed crimes" output file
options["predictions"] = "predpol/predpol_drug_predictions"				# path to "predicted crimes" output file
options["predpol_window"] = 180											# prediction window for PredPol
options["begin_predpol"] = 0											# day to begin adding crimes due to increased policing
options["add_crimes_logical"] = False									# add crimes due to increased policing?
options["percent_increase"] = 0.0										# % increase in crimes due to increased policing (as a fraction)

# Load data.
data = load_predpol_data(options)

# Run simulation.
run_predpol_simulation(data, options)
