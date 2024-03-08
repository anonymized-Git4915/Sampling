import run
from multiprocessing import Process, Queue

# starts the program with multiple Event logs

# All event logs for stating a process for each event log
sources_all = ["data/Eventlogs/DomesticDeclarations/DomesticDeclarations.xes",
                 "data/Eventlogs/BPIC 2012/BPI_Challenge_2012.xes",
                 "data/Eventlogs/InternationalDeclarations/InternationalDeclarations.xes",
                 "data/Eventlogs/PrepaidTravelCost/PrepaidTravelCost.xes",
                 "data/Eventlogs/RequestForPayment/RequestForPayment.xes",
                 "data/Eventlogs/PermitLog/PermitLog.xes",
                 "data/Eventlogs/BPIC 2015/BPIC15_1.xes",
                 "data/Eventlogs/sepsis/Sepsis Cases - Event Log.xes",
                 "data/Eventlogs/CoSeLoG WABO 2/CoSeLoG WABO 2.xes",
                 "data/Eventlogs/CoSeLoG WABO 3/CoSeLoG WABO 3.xes",
                 ]

# Event logs split into two lists, for two separate runs
sources_1 = ["data/Eventlogs/DomesticDeclarations/DomesticDeclarations.xes",
                 "data/Eventlogs/BPIC 2012/BPI_Challenge_2012.xes",
                 "data/Eventlogs/InternationalDeclarations/InternationalDeclarations.xes",
                 "data/Eventlogs/PrepaidTravelCost/PrepaidTravelCost.xes",
                 "data/Eventlogs/RequestForPayment/RequestForPayment.xes"]

sources_2 = ["data/Eventlogs/PermitLog/PermitLog.xes",
                 "data/Eventlogs/BPIC 2015/BPIC15_1.xes",
                 "data/Eventlogs/sepsis/Sepsis Cases - Event Log.xes",
                 "data/Eventlogs/CoSeLoG WABO 2/CoSeLoG WABO 2.xes",
                 "data/Eventlogs/CoSeLoG WABO 3/CoSeLoG WABO 3.xes",
                 ]


def rand_num(queue, string=""):
    run.sampling_and_eval(string)
    queue.put(string)


# start the multiprocessing
if __name__ == "__main__":
    queue = Queue()

    # init one process for each event log
    processes = [Process(target=rand_num, args=(queue, x)) for x in sources_1]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # results are only collected for finishing print
    results = [queue.get() for p in processes]

    print(results)
    print("All processes are finished")

