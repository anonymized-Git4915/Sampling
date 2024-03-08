import copy
import os
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
import ConfigManager
from ConfigManager import Config_Manager as config
import inspect
import pm4py
import pandas as pd


def import_raw_data():
    """
    Import file with raw data from spotter buoy
    :return: Pandas dataframe of the csv file
    """
    data = pd.read_csv(config.source_path, index_col=False)
    print("data import of " + config.source_path + " complete")
    return data


def import_data():
    """
    Import file with data of type .csv or .xes
    :return: Pandas dataframe of the csv or xes file
    """
    data_type = config.source_path[-3:]
    config.data_type = data_type
    if data_type == "csv":
        return pd.read_csv(config.source_path, sep=config.seperator_in_log)
    elif data_type == "xes":
        # set the column name in the config mananer
        config.ab_case_id_column = 'case:concept:name'
        config.ab_activity_column = 'concept:name'
        config.ab_timestamp_column = 'time:timestamp'

        # import the xes with pm4py
        event_log = pm4py.read_xes(config.source_path)
        df = pm4py.convert_to_dataframe(event_log)
        config.ab_format = "DF"
        return df


def time_calculation(start_time, start_time_sampling, start_time_pm_and_eval, end_time_sampling, end_time_pm_and_eval,
                     end_time):
    """
    Calculates and prints the calculation times
    :param start_time:
    :param start_time_sampling:
    :param start_time_pm_and_eval:
    :param end_time_sampling:
    :param end_time_pm_and_eval:
    :param end_time:
    :return:
    """
    # calculate runtime of sampling
    execution_time_sampling = end_time_sampling - start_time_sampling
    print("runtime sampling:", execution_time_sampling, "seconds")

    # calculate runtime of process mining and evaluation
    execution_time_pm_and_eval = end_time_pm_and_eval - start_time_pm_and_eval
    print("runtime process mining and evaluation:", execution_time_pm_and_eval, "seconds")

    # calculate full runtime of the program
    execution_time = end_time - start_time
    print("runtime complete:", execution_time, "seconds")

    # generate text file of runtime as output
    if config.output:
        with open(config.output_path + 'runtimes.txt', 'w') as f:
            f.write("runtime sampling:" + str(execution_time_sampling) + "seconds \n")
            f.write("runtime process mining and evaluation:" + str(execution_time_pm_and_eval) + "seconds \n")
            f.write("runtime complete:" + str(execution_time) + "seconds")


def save_info_file():
    """
    this function saves the Config-Manager as a file in the folder of the iteration
    :return:
    """
    if config.output:
        with open(config.output_path + "ConfigManager.txt", "w") as file:
            file.write(inspect.getsource(ConfigManager.Config_Manager))


def save_logs(base_log, samples):
    """
    saving the base log and the sampled logs as a file in the output directory
    :param base_log: the base log as a dataframe
    :param samples: a dict with the sampled logs as a dataframe
    :return:
    """
    if config.output:
        os.mkdir(config.output_path + "logs/")

        # save the samples of each sampling run
        if config.log_output_format == "csv":
            for key in samples:
                samples[key].to_csv(config.output_path + "logs/" + str(key) + ".csv")
            base_log.to_csv(config.output_path + "logs/base_log.csv")
        elif config.log_output_format == "xes":
            for key in samples:
                pm4py.write_xes(samples[key], config.output_path + "logs/" + str(key) + ".xes")
            pm4py.write_xes(base_log, config.output_path + "logs/base_log.xes")

    return None


def time_calculation_multi_algo(start_time, end_time):
    """
    this function calculate the runtime of multiple algorithms combined
    :param start_time: time when the algorithms started
    :param end_time: time when the algorithms ended
    :return:
    """
    # calculate runtime
    execution_time = end_time - start_time
    print("runtime complete:", execution_time, "seconds")
    if config.output:
        # generate text file of runtime as output
        with open(config.output_path + 'runtimes.txt', 'w') as f:
            f.write("runtime complete:" + str(execution_time) + "seconds")

def add_algos_to_repeat():
    """
    This function repeats the algorithm until the amount in the config is reached
    """
    for algo_to_repeat in config.algos_to_average:
        if algo_to_repeat in config.sampling_algo:
            counter = 2
            while counter <= config.algos_to_average_repeats:
                config.sampling_algo.append(algo_to_repeat + str(counter))
                counter += 1


def get_log_info():
    import pm4py
    from pm4py.objects.log.importer.xes import importer as xes_importer

    # Load the XES event log
    log_path = "data/Eventlogs/CoSeLoG WABO 3/CoSeLoG WABO 3.xes"
    # log = xes_importer.apply(log_path)
    log = pm4py.read_xes(log_path)

    # Calculate the number of unique traces
    num_unique_traces = len(pm4py.statistics.traces.generic.log.case_statistics.get_variant_statistics(log))
    # Calculate the number of cases (number of traces)
    num_cases = log['case:concept:name'].nunique()
    # Calculate the number of events
    num_events = len(log)

    num_unique_events = log['concept:name'].nunique()

    # Print the results
    print(log_path)
    print("Number of Cases:", num_cases)
    print("Number of Unique Traces:", num_unique_traces)
    print("Number of Events:", num_events)
    print("Number of Unique Events:", num_unique_events)