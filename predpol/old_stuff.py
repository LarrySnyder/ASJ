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

			# Add new crimes to model increased police presence.
			crime_data = add_new_crimes(crime_data, end_date, flagged_bins, options["percent_increase"])

			# # Get today's baseline crimes.
			# crime_today = pd.value_counts(crime_data[crime_data.DateTime==end_date].bin)
			# # Filter for the bins we will send police to.
			# crime_today_predicted = crime_today[[b for b in flagged_bins if b in crime_today.index]]
			# crime_today_predicted[pd.isnull(crime_today_predicted)] = 0

			# # Generate new crimes randomly, with binomial distribution based on predicted rates.
			# add_crimes = np.random.binomial(list(crime_today_predicted + 1), options["percent_increase"])

			# # Create dataframe for new crimes (will be appended to existing crime_data).
			# new_crimes = pd.DataFrame()
			# new_crimes['bin'] = np.repeat(list(crime_today_predicted.index), add_crimes)
			# new_crimes['DateTime'] = end_date
			# new_crimes['OCCURRED'] = end_date
			# new_crimes['LAG'] = 0
			# new_crimes.index = range(1 + max(crime_data.index), 1 + max(crime_data.index) + sum(add_crimes))
			# crime_data = pd.concat([crime_data, new_crimes])
			# #print(f"  len(crime_data) now = {len(crime_data)}")

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
