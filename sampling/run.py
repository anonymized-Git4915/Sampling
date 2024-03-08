import os
import sampling_management as sampling
import sampling_eval
import sampling_utils
from ConfigManager import Config_Manager as config
from datetime import datetime
import time

def sampling_and_eval(event_log_path=""):
    """
    Funktion to manage the sampling and evaluation.
    :param event_log_path:
    :return:
    """
    # non-deterministic algorithm are multiple times added to the list of algos to execute
    sampling_utils.add_algos_to_repeat()

    # only relevant for call from multiprocessing. Here the source path is set individually in each process instance
    if event_log_path != "":
        config.source_path = event_log_path

    # track start time
    start_time = time.time()

    # get the name of the source file
    stripped_event_log_name = config.source_path[config.source_path.rfind("/") + 1:config.source_path.rfind(".")]

    # set the output path for the iteration
    config.output_path = os.getcwd() + '/data/output/sampling_model_eval ' + str(
        datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_" + stripped_event_log_name + '/')
    print("Preparation finished")

    # ------- IMPORT -------
    print("Start data import")

    # load the base log for sampling and evaluation. Path is set in the config manager
    base_log = sampling_utils.import_data()
    print("Data import of " + str(config.source_path) + " successful")

    # ------- PRE-PROCESSING -------

    # sampling
    print("Start sampling")

    # init dict to collect the quality metrics of the models for each sampling configuration
    quality_metrics = {}

    # save original path to be accessible in all iterations
    original_path = config.output_path

    # create the output folder
    if config.output:
        os.makedirs(config.output_path + '/graphs')

    # dicts for collecting the results
    samples_metrics_results = {}
    dfrs = {}

    # iterate though all selected sampling algos
    for algo in config.sampling_algo:
        print("Start sampling with " + algo + " for " + config.source_path)

        # create output path for this specific algo
        config.output_path = original_path + algo
        os.mkdir(config.output_path)

        # sample the event logs
        samples = sampling.sampling(base_log.copy(), algo)

        # calculate the quality metrics for the samples of the current "algo"
        if config.with_evaluation:
            samples_metrics_results[algo], dfrs[algo] = sampling_eval.samples_eval(samples, base_log, algo)

        # save the results as file
        if config.with_evaluation:
            sampling_eval.save_results(samples_metrics_results[algo], dfrs[algo], samples)

    # reset the output path
    config.output_path = original_path

    # average the results from non-deterministic algos
    if config.with_evaluation:
        sampling_eval.average_sampling_metrics(samples_metrics_results)

    # save the overall results as file
    if config.with_evaluation:
        sampling_eval.save_overall_results(samples_metrics_results)

    # save the overalls plots  as file
    if config.with_evaluation:
        sampling_eval.vis_results(samples_metrics_results)

    # track the end time of the proces mining and eval
    end_time = time.time()

    # calculate and print the processing times
    sampling_utils.time_calculation_multi_algo(start_time, end_time)

    # save a copy of the configuration from the ConfigManager
    sampling_utils.save_info_file()
    print("Finished successfully")
