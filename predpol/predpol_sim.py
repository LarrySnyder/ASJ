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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


NUM_FLAGS = 20	# number of bins to flag per day
NUM_COLS = 20	# number of columns in heatmap


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


def calc_tij(crime_data):
	assert all([len(n) > 0 for n in crime_data.values()])
	tij = dict()
	for n, events in crime_data.items():
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


def estep(crime_data, mu, theta, omega, tij):
	# t these asserts are fast and document the common structure
	assert crime_data.keys() == mu.keys()
	assert crime_data.keys() == tij.keys()
	pj = dict()
	pij = calc_pij(tij, theta, omega)
	for n in crime_data:
		# should possibly append a 1 to the front of this
		denom = mu[n] + pij[n].sum(axis=1)
		pj[n] = mu[n] / denom
		pij[n] = pij[n] / denom
		pj[n] = np.append(pj[n], 1)
	return pij, pj


# t iterating over a dict means over keys unless otherwise specified
def mstep(pij, pj, tij, crime_data, T):
	# t these asserts are fast and document the common structure
	assert crime_data.keys() == pij.keys()
	assert crime_data.keys() == tij.keys()
	assert crime_data.keys() == pj.keys()
	total_events = sum([len(v) for v in crime_data.values()])
	# double sum: arrays over bins; np.sum whole array at once
	sum_pijs = sum([np.sum(pij[n]) for n in pij])
	denom = sum([np.sum(pij[n] * tij[n]) for n in pij])
	omega = sum_pijs / denom
	theta = sum_pijs / total_events
	mu = dict((n, sum(pj[n]) / T) for n in pj)
	# mu = np.ones(num_bins)*sum(mu)  #this is what the paper says to do but
	# this forces the "background rate" to be the same everywhere,
	return omega, theta, mu


def runEM(crime_data, T, pred_date, k=20,
		  theta_init=1, omega_init=1, mu_init=1,
		  tol1=.00001, tol2=.00001, tol3=.0001):
	num_bins = len(crime_data)
	theta = theta_init
	mu = dict((key, mu_init) for key in crime_data)
	omega = omega_init
	omega_last = 10 + omega
	theta_last = 10 + theta
	mu_last = dict((key, mu_init + 10) for key in crime_data)
	k = min(num_bins, k)
	tij = calc_tij(crime_data)

	while(abs(omega - omega_last) > tol1 and
		  abs(theta - theta_last) > tol2 and
		  sum([abs(mu[n] - mu_last[n]) for n in mu]) > tol3):
		omega_last = omega
		theta_last = theta
		mu_last = copy.deepcopy(mu)
		pij, pj = estep(crime_data, mu, theta, omega, tij)
		omega, theta, mu = mstep(pij, pj, tij, crime_data, T)

		# deprecate?
		# I did this when I was debugging so that it wouldn't run away
		# if omega > T * 1000:
		#	omega = omega_last
		assert omega < T * 1000

	# get conditional intensity for selected parameters
	# need to add on latest date
	for n in crime_data:
		crime_data[n] = np.append(crime_data[n], pred_date)

	# Calculate rate for each bin.
	rates = {n: lambdafun(crime_data[n], mu[n], theta, omega) for n in crime_data}

	return rates, omega, theta

	# todo(KL): everything else is a dict[n], shouldn't rates be a dict?
	# rates = [lambdafun(crime_data[n], mu[n], theta, omega) for n in crime_data]
	# # Determine which bins to send police to.
	# # TODO: put this in a separate function, that can be customized
	# sorted_bins = [key for (rate, key)
	# 			   in sorted(zip(rates, crime_data), reverse=True)]
	# sorted_rates = [rates[k] for k in range(len(sorted_bins))]

	# return rates, sorted_bins, sorted_rates, omega, theta


# --- PLOTTING --- #

def plot_heatmaps(baseline_filepath, observed_filepath, flagged_filepath, output_pdf_filepath=None):

	# Read CSV files.
	baseline_df = pd.read_csv(baseline_filepath)
	observed_df = pd.read_csv(observed_filepath, index_col='bin')
	flagged_df = pd.read_csv(flagged_filepath, index_col='bin')

	# Get # of crimes in each bin (baseline and observed).
	baseline_crimes = pd.value_counts(baseline_df.bin)
	baseline_crimes_array = np.zeros(len(observed_df))
	for i in baseline_crimes.index:
		baseline_crimes_array[int(i) - 1] = baseline_crimes[int(i)]
	observed_crimes = observed_df.sum(axis=1)
	observed_crimes_array = np.array(list(observed_crimes))

	# Get # of times each bin is flagged.
	flags = flagged_df.sum(axis=1)
	flags_array = np.array(list(flags))

	# Add 0s so that size of arrays are multiples of NUM_COLS.
	while len(baseline_crimes_array) % NUM_COLS != 0:
		baseline_crimes_array = np.append(baseline_crimes_array, [0])
	while len(observed_crimes_array) % NUM_COLS != 0:
		observed_crimes_array = np.append(observed_crimes_array, [0])
	while len(flags_array) % NUM_COLS != 0:
		flags_array = np.append(flags_array, [0])

	# Convert to 10-column numpy arrays.
	baseline_crimes_array = baseline_crimes_array.reshape(-1, NUM_COLS)
	observed_crimes_array = observed_crimes_array.reshape(-1, NUM_COLS)
	flags_array = flags_array.reshape(-1, NUM_COLS)

	# Build heatmaps.
	fig, axes = plt.subplots(1, 3)
	fig.set_figheight(5)
	fig.set_figwidth(17)
	sns.heatmap(baseline_crimes_array, cmap='flare', ax=axes[0])
	axes[0].title.set_text('Baseline Crimes')
	sns.heatmap(observed_crimes_array, cmap='flare', ax=axes[1])
	axes[1].title.set_text('Observed Crimes')
	sns.heatmap(flags_array, cmap='flare', ax=axes[2])
	axes[2].title.set_text('# Times Flagged')

	# Title.
	fig.suptitle(f"(total crimes observed: {observed_df.sum().sum()})")

	if output_pdf_filepath:
		fig.savefig(output_pdf_filepath, bbox_inches='tight')

	plt.show()


def plot_heatmaps_from_dataframes(baseline_df, observed_df, flagged_df, fig=None):

	# Get # of crimes in each bin (baseline and observed).
	baseline_crimes = pd.value_counts(baseline_df.bin)
	baseline_crimes_array = np.zeros(len(observed_df))
	for i in baseline_crimes.index:
		baseline_crimes_array[int(i) - 1] = baseline_crimes[int(i)]
	observed_crimes = observed_df.sum(axis=1)
	observed_crimes_array = np.array(list(observed_crimes))

	# Get # of times each bin is flagged.
	flags = flagged_df.sum(axis=1)
	flags_array = np.array(list(flags))

	# Add 0s so that size of arrays are multiples of NUM_COLS.
	while len(baseline_crimes_array) % NUM_COLS != 0:
		baseline_crimes_array = np.append(baseline_crimes_array, [0])
	while len(observed_crimes_array) % NUM_COLS != 0:
		observed_crimes_array = np.append(observed_crimes_array, [0])
	while len(flags_array) % NUM_COLS != 0:
		flags_array = np.append(flags_array, [0])

	# Convert to 10-column numpy arrays.
	baseline_crimes_array = baseline_crimes_array.reshape(-1, NUM_COLS)
	observed_crimes_array = observed_crimes_array.reshape(-1, NUM_COLS)
	flags_array = flags_array.reshape(-1, NUM_COLS)

	# Initialize.
	first_iter = fig is None
	if fig is None:
		fig, axes = plt.subplots(1, 3)
		fig.set_figheight(5)
		fig.set_figwidth(17)
		axes[0].title.set_text('Baseline Crimes')
		axes[1].title.set_text('Observed Crimes')
		axes[2].title.set_text('# Times Flagged')
	else:
		axes = fig.axes

	# Get cbar axes. (fig.axes contains 6 axes; first 3 are heatmaps and last 3 are colorbars.)
	# (If we don't do this, new colorbar will be drawn at each iteration and take up new space.)
	cbar_ax_baseline = None if first_iter else fig.axes[3]
	cbar_ax_observed = None if first_iter else fig.axes[4]
	cbar_ax_flags = None if first_iter else fig.axes[5]

	# Build heatmaps.
	sns.heatmap(baseline_crimes_array, cmap='flare', ax=axes[0], cbar_ax=cbar_ax_baseline)
	sns.heatmap(observed_crimes_array, cmap='flare', ax=axes[1], cbar_ax=cbar_ax_observed)
	sns.heatmap(flags_array, cmap='flare', ax=axes[2], cbar_ax=cbar_ax_flags)

	# Title.
	fig.suptitle(f"Iteration {observed_df.shape[1]} (total crimes observed: {observed_df.sum().sum()})")

	# if show:
	# 	plt.show()
	# else:
	plt.pause(0.0001)

	return fig



# --- SIMULATION --- #

def prepare_data_for_predpol(crime_data, start_date, end_date):
	"""Prepare crime_data for PredPol simulation."""

	# Remove records that are not in the desired time window.
	crime_data = crime_data[(crime_data.DateTime >= start_date) & (crime_data.DateTime < end_date)]

	# Build output dict. (Keys are map bins, values are crime_data values.)
	pp_dict = dict((int(i), []) for i in set(crime_data.bin))
	for i in crime_data.index:
		pp_dict[int(crime_data.bin[i])].append(crime_data['DateTime'].loc[i])

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
	"""Load crime_data from CSV file."""

	# Read crime_data.
	crime_data = pd.read_csv(options["drug_crimes_with_bins"])
	# Specify column names.
	crime_data.columns = ['rownum', 'bin', 'OCCURRED', 'LAG']
	# Remove crime_data with null bins.
	crime_data = crime_data[pd.notnull(crime_data['bin'])]
	# Format dates.
	crime_data['DateTime'] = pd.to_datetime(crime_data.OCCURRED, format = '%m/%d/%y')

	return crime_data


def get_empty_bin_dataframe(max_bin):
	"""Build an empty dataframe with index = 'bin' and indices from 1, ..., max_bin. """

	df = pd.DataFrame()
	df['bin'] = range(1, max_bin + 1)
	df = df.set_index(['bin'])

	return df


def add_column_to_bin_dataframe(df, bins, values, str_date):
	"""Add a column (in place) to `df`, with header `str_date`. For each bin in `bins`, the function
	sets its value equal to the corresponding value in `values`; all other values are set to 0. 
	(`bins` and `values` must be lists of the same length.)
	This function takes care of a few annoying aspects of adding the column, e.g., 
	handling first column vs. subsequent ones."""
	temp_series = pd.DataFrame(0, index=df.index, columns=[str_date])
	temp_series.loc[bins, str_date] = values
	if df.empty:
		df[str_date] = list(temp_series[str_date])
	else:
		df = pd.concat([df, temp_series], axis=1)
	return df


def run_predpol(crime_data_df, window_start, window_end, options):

	# Prepare data for PredPol calculations.
	pp_dict = prepare_data_for_predpol(crime_data_df, window_start, window_end)

	# Run PredPol calculations.
	rates, omega, theta = runEM(pp_dict, options["predpol_window"], window_end)

	return rates, omega, theta


def choose_flagged_bins(rates, observed_crimes_df):
	"""Choose which bins to flag, given the bins and their rates."""

	# Sort rates.
	sorted_bins = sorted(rates, key=lambda x: rates[x], reverse=True)
	sorted_rates = [rates[b] for b in sorted_bins]

	# Choose the NUM_FLAGS bins with the largest rates.
	flagged_bins = sorted_bins[0:NUM_FLAGS]

	return flagged_bins


def run_predpol_simulation(crime_data, options):
	"""Run the PredPol simulation."""

	# Determine number of bins needed for dataframe.
	max_bin = int(max(crime_data.bin))
	print(f"max_bin = {max_bin}")

	# Remove records that are not in the desired time window.
	# TODO: don't we do this in prepare_data_for_predpol() too?
	print(f"len(crime_data) = {len(crime_data)} before removing records by date")
	crime_data = crime_data[(crime_data.DateTime >= options["global_start"]) & (crime_data.DateTime < options["global_end"])]
	print(f"len(crime_data) = {len(crime_data)} after removing records by date")

	# Determine number of predictions (= # days in crime_data minus predpol window).
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

		# Prepare crime_data for PredPol.
		pp_dict = prepare_data_for_predpol(crime_data, start_date, end_date)

		# Run PredPol calculations.
		rates, omega, theta = runEM(pp_dict, options["predpol_window"], end_date)
#		rates, sorted_bins, sorted_rates, omega, theta = runEM(pp_dict, options["predpol_window"], end_date)

		# Sort rates.
		sorted_bins = sorted(rates, key=lambda x: rates[x], reverse=True)
		sorted_rates = [rates[b] for b in sorted_bins]

		# Choose which bins to send police to.
		flagged_bins = choose_flagged_bins(sorted_bins, sorted_rates)

		# Save rates.
		str_date = str(end_date).split(' ')[0]
		rates_series = pd.DataFrame(0, index=results_rates.index, columns=[str_date])
		keys = list(pp_dict.keys())
		rates_series.loc[keys, str_date] = rates
		if results_rates.empty:
			results_rates[str_date] = list(rates_series[str_date])
		else:
			results_rates = pd.concat([results_rates, rates_series], axis=1)
		# results_rates[str_date] = 0
		# keys = list(pp_dict.keys())
		# results_rates.loc[keys, str_date] = rates

		# Add crimes, if that's what we're doing.
		if options["add_crimes_logical"] and i >= options["begin_predpol"]:
			# Get today's baseline crimes.
			crime_today = pd.value_counts(crime_data[crime_data.DateTime==end_date].bin)
			# Filter for the bins we will send police to.
			crime_today_predicted = crime_today[[b for b in flagged_bins if b in crime_today.index]]
			crime_today_predicted[pd.isnull(crime_today_predicted)] = 0

			# Generate new crimes randomly, with binomial distribution based on predicted rates.
			add_crimes = np.random.binomial(list(crime_today_predicted + 1), options["percent_increase"])

			# Create dataframe for new crimes (will be appended to existing crime_data).
			new_crimes = pd.DataFrame()
			new_crimes['bin'] = np.repeat(list(crime_today_predicted.index), add_crimes)
			new_crimes['DateTime'] = end_date
			new_crimes['OCCURRED'] = end_date
			new_crimes['LAG'] = 0
			new_crimes.index = range(1 + max(crime_data.index), 1 + max(crime_data.index) + sum(add_crimes))
			crime_data = pd.concat([crime_data, new_crimes])
			#print(f"  len(crime_data) now = {len(crime_data)}")

		# Record total number of crimes on this day.
		crime_today = pd.value_counts(crime_data[crime_data.DateTime==end_date].bin)
		num_crimes_series = pd.DataFrame(0, index=results_num_crimes.index, columns=[str_date])
		keys = list(pp_dict.keys())
		num_crimes_series.loc[keys, str_date] = rates
		if results_num_crimes.empty:
			results_num_crimes[str_date] = num_crimes_series[str_date]
		else:
			results_num_crimes = pd.concat([results_num_crimes, num_crimes_series], axis=1)
		# results_num_crimes[str_date] = 0
		# results_num_crimes.loc[crime_today.index, str_date] = list(crime_today)

		# Remove old crime_data to speed things up.
		crime_data = crime_data[(crime_data.DateTime >= start_date)]

	# Write results.
	results_rates.to_csv(options["predictions"])
	results_num_crimes.to_csv(options["observed"])


def run_predpol_simulation2(crime_data_df, options):
	"""Run the PredPol simulation."""

	# Determine number of bins needed for dataframe.
	max_bin = int(max(crime_data_df.bin))

	# Remove data that are not in the desired time window.
	crime_data_df = crime_data_df[(crime_data_df.DateTime >= options["global_start"]) & (crime_data_df.DateTime < options["global_end"])]

	# Make a copy to serve as basline (just for record-keeping).
	baseline_crimes_df = crime_data_df.copy(deep=True) 

	# Determine number of days to simulate.
	num_days = (options["global_end"] - options["global_start"]).days - options["predpol_window"]
	
	# Initialize dataframe for PredPol rates, observed crimes, and whether a bin was flagged.
	predpol_rates_df = get_empty_bin_dataframe(max_bin)
	observed_crimes_df = get_empty_bin_dataframe(max_bin)
	flagged_df = get_empty_bin_dataframe(max_bin)

	# Initialize heatmap figure.
	fig = None

	# Initialize progress bar.
	pbar = tqdm(total=num_days)

	# Loop through days in simulation.
	for i in range(num_days):

		# Update progress bar.
		pbar.update()

		# Determine dates of PredPol window.
		window_start = options["global_start"] + pd.DateOffset(i)
		window_end = options["global_start"] + pd.DateOffset(i + options["predpol_window"])

		# Run PredPol.
		rates, omega, theta = run_predpol(crime_data_df, window_start, window_end, options)

		# Choose which bins to send police to.
		flagged_bins = choose_flagged_bins(rates, observed_crimes_df)

		# Get today's baseline crimes.
		crime_today = pd.value_counts(crime_data_df[crime_data_df.DateTime==window_end].bin)

		# Add crimes, if that's what we're doing.
		if options["add_crimes_logical"] and i >= options["begin_predpol"]:

			# Filter for the bins we will send police to.
			crime_today_flagged_bins = crime_today[[b for b in flagged_bins if b in crime_today.index]]
			crime_today_flagged_bins[pd.isnull(crime_today_flagged_bins)] = 0

			# Generate new crimes randomly, with binomial distribution based on PredPol rates.
			addl_crimes = np.random.binomial(list(crime_today_flagged_bins + 1), options["percent_increase"])

			# Create dataframe for new crimes (will be appended to existing df).
			# TODO: make this more compact
			new_crimes = pd.DataFrame()
			new_crimes['bin'] = np.repeat(list(crime_today_flagged_bins.index), addl_crimes)
			new_crimes['DateTime'] = window_end
			new_crimes['OCCURRED'] = window_end
			new_crimes['LAG'] = 0
			new_crimes.index = range(1 + max(crime_data_df.index), 1 + max(crime_data_df.index) + sum(addl_crimes))
			crime_data_df = pd.concat([crime_data_df, new_crimes])

			# Add new crimes to crime_today.
			for k in range(len(new_crimes)):
				crime_today.loc[[new_crimes.iloc[k]['bin']]] += 1

		# Save today's data.
		str_date = str(window_end).split(' ')[0]
		predpol_rates_df = add_column_to_bin_dataframe(predpol_rates_df, bins=list(rates.keys()), values=list(rates.values()), str_date=str_date)
		flagged_df = add_column_to_bin_dataframe(flagged_df, bins=flagged_bins, values=[1] * len(flagged_bins), str_date=str_date)
		observed_crimes_df = add_column_to_bin_dataframe(observed_crimes_df, bins=crime_today.index, values=list(crime_today), str_date=str_date)

		# Display heatmap.
		if options["heatmap_display_interval"] > 0 and i % options["heatmap_display_interval"] == 0:
			fig = plot_heatmaps_from_dataframes(baseline_crimes_df, observed_crimes_df, flagged_df, fig)

		# Remove old data from crime_data_df to speed things up.
		crime_data_df = crime_data_df[(crime_data_df.DateTime >= window_start)]

	# Close progress bar.
	pbar.close()

	# Write results to CSV.
	predpol_rates_df.to_csv(options["rates_filename"])
	flagged_df.to_csv(options["flagged_filename"])
	baseline_crimes_df.to_csv(options["baseline_crime_filename"])
	observed_crimes_df.to_csv(options["observed_crime_filename"])


# --- CODE TO RUN THE SIMULATION --- #

def set_options():
	# Set up simulation options.
	options = {}
	options["drug_crimes_with_bins"] = "predpol/input/drug_crimes_with_bins.csv"	# path to input file
	options["global_start"] = pd.to_datetime("2010/07/01")							# simulation start date
#	options["global_end"] = pd.to_datetime("2010/12/31")							# simulation end date
#	options["global_end"] = pd.to_datetime("2011/1/31")								# simulation end date
	options["global_end"] = pd.to_datetime("2011/12/31")							# simulation end date
	options["baseline_crime_filename"] = "predpol/output/predpol_drug_baseline"		# path to "baseline crimes" output file
	options["observed_crime_filename"] = "predpol/output/predpol_drug_observed"		# path to "observed crimes" output file
	options["rates_filename"] = "predpol/output/predpol_drug_rates"					# path to "rates" output file
	options["flagged_filename"] = "predpol/output/predpol_flagged"					# path to "flagged" output file
	options["predpol_window"] = 180													# prediction window for PredPol
	options["begin_predpol"] = 0													# day to begin adding crimes due to increased policing
	options["add_crimes_logical"] = False											# add crimes due to increased policing?
	options["percent_increase"] = 0.2												# % increase in crimes due to increased policing (as a fraction)
	options["heatmap_display_interval"] = 0										# display heatmaps every this many iterations (if 0, don't display heatmaps until end)
	options["heatmap_filename"] = "predpol/output/heatmap"							# path to save heatmap PDF

	return options

def generate_odds_figure():
	"""Generate Figure 3."""

	options = set_options()

	# TODO: just read files, don't rerun sim

	# Load crime_data.
	crime_data = load_predpol_data(options)

	# Run without adding crimes.
	options["baseline_crime_filename"] = "predpol/output/temp_baseline.csv"
	options["observed_crime_filename"] = "predpol/output/temp_observed.csv"
	options["rates_filename"] = "predpol/output/temp_rates.csv"
	options["flagged_filename"] = "predpol/output/temp_flagged.csv"
	options["add_crimes_logical"] = False
	run_predpol_simulation2(crime_data, options)

	# Run with adding crimes.
	options["baseline_crime_filename"] = "predpol/output/temp_baseline_add_20percent.csv"
	options["observed_crime_filename"] = "predpol/output/temp_observed_add_20percent.csv"
	options["rates_filename"] = "predpol/output/temp_rates_add_20percent.csv"
	options["flagged_filename"] = "predpol/output/temp_flagged_add_20percent.csv"
	options["add_crimes_logical"] = True
	run_predpol_simulation2(crime_data, options)

	# Read CSV files.
	rates_no_addl_df = pd.read_csv("predpol/output/temp_rates.csv")
	rates_addl_df = pd.read_csv("predpol/output/temp_rates_add_20percent.csv")
	flagged_no_addl_df = pd.read_csv("predpol/output/temp_flagged.csv")
	flagged_addl_df = pd.read_csv("predpol/output/temp_flagged_add_20percent.csv")

	# Calculate ratio of rates for flagged vs. non-flagged bins for case with no additional crimes.
	odds_no_addl = []
	for date in rates_no_addl_df:
		rates_flagged_no_addl = []
		rates_not_flagged_no_addl = []
		for bin in rates_no_addl_df.index:
			if flagged_no_addl_df[date][bin] == 1:
				rates_flagged_no_addl.append(rates_no_addl_df[date][bin])
			else:
				rates_not_flagged_no_addl.append(rates_no_addl_df[date][bin])
		odds_no_addl.append(np.average(rates_flagged_no_addl) / np.average(rates_not_flagged_no_addl))

	# Calculate ratio of rates for flagged vs. non-flagged bins for case with additional crimes.
	odds_addl = []
	for date in rates_addl_df:
		rates_flagged_addl = []
		rates_not_flagged_addl = []
		for bin in rates_addl_df.index:
			if flagged_addl_df[date][bin] == 1:
				rates_flagged_addl.append(rates_addl_df[date][bin])
			else:
				rates_not_flagged_addl.append(rates_addl_df[date][bin])
		odds_addl.append(np.average(rates_flagged_addl) / np.average(rates_not_flagged_addl))

	fig = plt.figure()
	plt.plot(odds_no_addl[1:])
	plt.plot(odds_addl[1:])
	plt.show()

	fig.savefig("figure3", bbox_inches='tight')


def main(options):

	options = set_options()

	# Load crime_data.
	crime_data = load_predpol_data(options)

	# Adjust output filenames.
	if options["add_crimes_logical"]:
		options["rates_filename"] += '_add_' + str(int(options["percent_increase"]*100)) + 'percent.csv'
		options["baseline_crime_filename"] += '_add_' + str(int(options["percent_increase"]*100)) + 'percent.csv'
		options["observed_crime_filename"] += '_add_' + str(int(options["percent_increase"]*100)) + 'percent.csv'
		options["flagged_filename"] += '_add_' + str(int(options["percent_increase"]*100)) + 'percent.csv'
		options["heatmap_filename"] += '_add_' + str(int(options["percent_increase"]*100)) + 'percent.pdf'
	else:
		options["rates_filename"] += '.csv'
		options["baseline_crime_filename"] += '.csv'
		options["observed_crime_filename"] += '.csv'
		options["flagged_filename"] += '.csv'
		options["heatmap_filename"] += '.pdf'

	# Run simulation.
#	run_predpol_simulation(crime_data, options)
	run_predpol_simulation2(crime_data, options)

	# Draw plots.
	plot_heatmaps(
		baseline_filepath=options["baseline_crime_filename"],
		observed_filepath=options["observed_crime_filename"],
		flagged_filepath=options["flagged_filename"],
		output_pdf_filepath=options["heatmap_filename"]
	)


if __name__ == '__main__':
	main()
#	generate_odds_figure()
