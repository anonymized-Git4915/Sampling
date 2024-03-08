import csv
import time

import sampling_RP_and_AB as sampling_n
from ConfigManager import Config_Manager as config
import pandas as pd
import external_sampling.CminSampler as CminSampler


def sampling(base_log, sampling_algo):
    """
    This function calls the set of sampling algorithms. The Algorithms are set in the Config Manager
    :param base_log: dataframe, the non sampled log
    :param sampling_algo: string, specifies the sampling algorithm
    :return: dictionary, dict of the sampling, the key is the sample-ratio
    """

    # start the sample algorithms depending on which is called
    if sampling_algo.startswith("allbehaviour_Sampling"):
        return allbehaviour_config(base_log)
    elif sampling_algo.startswith("remainder_plus_sampling"):
        return remainder_plus_config(base_log)
    elif sampling_algo.startswith("random_Sampling"):
        return random_config(base_log)
    elif sampling_algo.startswith("stratified_sampling"):
        return stratified_sampling(base_log)
    elif sampling_algo.startswith("c_min_sampling"):
        return c_min_sampling(base_log)
    else:
        raise ValueError("No valid sampling algorythm selected")


def remainder_plus_config(base_log):
    """
    This function calls the remainder plus sampling algorithm for every specified sample ratio. Ratios are set in the Config Manager.
    :param base_log: dataframe, the non sampled log
    :return:
       """

    # init dict for samples
    samples = {}

    counter = 1

    # dict for collecting the processing times of each sampleing run
    processing_times = {}

    # iterates over sample ratios and starts algo
    for ratio in config.ab_sample_ratio:
        start_time = time.time()

        # call the remainder plus sampling
        samples[ratio] = sampling_n.remainder_plus_sampling(base_log, format=config.ab_format, sample_ratio=ratio,
                                                            case_id_column=config.ab_case_id_column,
                                                            activity_column=config.ab_activity_column,
                                                            timestamp_column=config.ab_timestamp_column,
                                                            format_timestamp_conversion_to_datetime_obj=config.ab_format_timestamp_conversion_to_datetime_obj,
                                                            min_columns=config.ab_min_columns)

        # record runtime and save in dict
        end_time = time.time()
        processing_times[ratio] = end_time - start_time

        print("Sampling " + str(counter) + " of " + str(len(config.ab_sample_ratio)) + " completed (Remainder Plus on " + config.source_path + ")")
        counter = counter + 1

    # save the runtimes
    with open(config.output_path + '/remainder_plus_sampling_runtimes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(processing_times.keys())
        writer.writerow(processing_times.values())

    return samples


def allbehaviour_config(base_log):
    """
    This function calls the allbehaviour sampling algorithm for every specified sample ratio. Ratios are set in the Config Manager.
    :param base_log: dataframe, the non sampled log
    :return:
    """

    # init dict for samples
    samples = {}

    counter = 1

    # dict for collecting the processing times of each sampleing run
    processing_times = {}

    # iterates over sample ratios and starts algo
    for ratio in config.ab_sample_ratio:
        start_time = time.time()

        # call the remainder plus sampling
        samples[ratio] = sampling_n.allbehaviour_Sampling(base_log, format=config.ab_format, sample_ratio=ratio,
                                                          case_id_column=config.ab_case_id_column,
                                                          activity_column=config.ab_activity_column,
                                                          timestamp_column=config.ab_timestamp_column,
                                                          format_timestamp_conversion_to_datetime_obj=config.ab_format_timestamp_conversion_to_datetime_obj,
                                                          min_columns=config.ab_min_columns)

        # record runtime and save in dict
        end_time = time.time()
        processing_times[ratio] = end_time - start_time

        print("Sampling " + str(counter) + " of " + str(len(config.ab_sample_ratio)) + " completed (Allbehaviour Sampling on " + config.source_path + ")")
        counter = counter + 1

    # save the runtimes
    with open(config.output_path + '/allbehaviour_Sampling_runtimes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(processing_times.keys())
        writer.writerow(processing_times.values())

    return samples


def random_config(base_log):
    """
    This function uses random sampling.
    :param base_log: dataframe, the non sampled log
    :return:
    """

    # init dict for samples
    samples = {}

    counter = 1
    case_name = config.ab_case_id_column
    processing_times = {}

    # iterates over sample ratios and starts algo
    for ratio in config.ab_sample_ratio:
        start_time = time.time()

        # Group dataframe by cases and choose random cases in size of the ratio
        case_ids = base_log[case_name].unique()
        selected_case_ids = pd.Series(case_ids).sample(frac=ratio)

        # Filter the log by these Case_IDs selected
        sampled_log = base_log[base_log[case_name].isin(selected_case_ids)]

        # save the sample in the dict
        samples[ratio] = sampled_log

        # record runtime and save in dict
        end_time = time.time()
        processing_times[ratio] = end_time - start_time
        print("Sampling " + str(counter) + " of " + str(len(config.ab_sample_ratio)) + " completed (Random Sampling on " + config.source_path + ")")

        counter = counter + 1

    # save the runtimes
    with open(config.output_path + '/random_Sampling_runtimes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(processing_times.keys())
        writer.writerow(processing_times.values())

    return samples


def stratified_sampling(base_log):
    """
    This function uses stratified sampling.
    :param base_log: dataframe, the non sampled log
    :return:
    """

    # init dict for samples
    samples = {}

    counter = 1
    case_name = config.ab_case_id_column
    processing_times = {}

    # iterates over sample ratios and starts algo
    for ratio in config.ab_sample_ratio:
        start_time = time.time()

        # Group the dataframe by cases and write the trace (activity order) in a string, that defines the case.
        grouped_df = base_log.groupby(config.ab_case_id_column)[config.ab_activity_column].apply(
            lambda x: ', '.join(x)).reset_index()

        # Group the cases by same traces
        grouped_cases = grouped_df.groupby(config.ab_activity_column, group_keys=False)

        # Apply sampling on each trace group
        sampled_cases = grouped_cases.apply(lambda x: x.sample(frac=ratio))

        # Create a list of the selected cases
        selected_case_ids = sampled_cases[config.ab_case_id_column]

        # Filter the log by these Case_IDs selected
        sampled_log = base_log[base_log[case_name].isin(selected_case_ids)]
        samples[ratio] = sampled_log

        # record runtime and save in dict
        end_time = time.time()
        processing_times[ratio] = end_time - start_time
        print("Sampling " + str(counter) + " of " + str(len(config.ab_sample_ratio)) + " completed (Stratified Sampling on " + config.source_path + ")")
        counter = counter + 1

        # save the runtimes
    with open(config.output_path + '/stratified_sampling_runtimes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(processing_times.keys())
        writer.writerow(processing_times.values())

    return samples


def c_min_sampling(base_log):
    """
    This function uses the c min sampling from the literature.
    https://ieeexplore.ieee.org/document/9576679
    :param base_log: dataframe, the non sampled log
    :return:
    """

    # init dict for samples
    samples = {}

    counter = 1
    processing_times = {}
    case_name = config.ab_case_id_column

    # iterates over sample ratios and starts algo
    for ratio in config.ab_sample_ratio:
        start_time = time.time()

        # get number of trace variants
        num_of_cases = base_log[config.ab_case_id_column].nunique()
        num_f_traces_in_sl = round(ratio * num_of_cases)

        # start the c-min sampling
        sampler = CminSampler.CminSampler(num_f_traces_in_sl)
        sel_cases = sampler.load_df(base_log, config.ab_case_id_column, config.ab_activity_column)
        sample_unformated = sampler.sample(output="case")

        # Filter the log by these Case_IDs selected
        sampled_log = base_log[base_log[case_name].isin(sample_unformated)]
        samples[ratio] = sampled_log

        # save the runtimes
        end_time = time.time()
        processing_times[ratio] = end_time - start_time
        print("Sampling " + str(counter) + " of " + str(len(config.ab_sample_ratio)) + " completed (C-Min Sampling on " + config.source_path + ")")
        counter = counter + 1

    # save the runtimes
    with open(config.output_path + '/c_min_sampling_runtimes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(processing_times.keys())
        writer.writerow(processing_times.values()),

    return samples

