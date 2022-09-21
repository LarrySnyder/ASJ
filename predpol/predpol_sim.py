# Simulation of PredPol predictive policing system.
#
# Larry Snyder, Lehigh University
#
# Main ideas in this module are adapted from Lum and Isaac, "To Predict and Serve?,"
# Significance Magazine, 14-19, October 2016 (https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1740-9713.2016.00960.x)
# and their associated GitHub repo at https://github.com/arun-ramamurthy/pred-pol.

# Set up simulation options.
options = {}
options["drug_crimes_with_bins"] = "drug_crimes_with_bins.csv"	# path to input file
options["global_start"] = "2010/07/01"							# simulation start date
options["global_end"] = "2011/12/31"							# simulation end date
options["observed"] = "predpol_drug_observed"					# path to "observed crimes" output file
options["predictions"] = "predpol_drug_predictions"				# path to "predicted crimes" output file
options["predpol_window"] = 180									# prediction window for PredPol
options["begin_predpol"] = 0									# day to begin adding crimes due to increased policing
options["add_crimes_logical"] = False							# add crimes due to increased policing?
options["percent_increase"] = 0.0								# % increase in crimes due to increased policing

def prepare_data_for_predpol(data, start_time, end_time):
	"""Prepare data for PredPol simulation."""

	# Remove records that are not in the desired time window.
	data = data[(data.DateTime >= start_time) & (data.DateTime < end_time)]

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


