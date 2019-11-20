# resultParser.py

import numpy as np 
import re
import pandas as pd


# regex dictionary
rx_dict = {
    'nprocs': re.compile(r'Timing for nProcs = (?P<nprocs>.*)\n'),
    'time': re.compile(r'\^\^\^Time to complete 1 multigrid vcycle\(s\)\: (?P<time>.*)\n'),
    'error': re.compile(r'\^\^\^relative error multigrid \(cycle 1\)\: (?P<error>.*)\n')
}


def _parse_line(line):
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    return None, None


def parse_file(filepath, nruns):

    nprocsList = []
    tempTimingList = []
    timingList = []
    tempErrorList = []
    errorList = []

    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            key, match = _parse_line(line)

            if key == 'nprocs':
                nprocsList.append(int(match.group('nprocs')))
                #data.append(nprocs)
            if key == 'time':
                tempTimingList.append(float(match.group('time')))
            if key == 'error':
                tempErrorList.append(float(match.group('error')))

            line = file_object.readline()

    # sum up the timings
    while tempTimingList:
        avgt = sum(tempTimingList[0:nruns]) /nruns
        tempTimingList = tempTimingList[nruns:] 
        avge = sum(tempErrorList[0:nruns]) /nruns
        tempErrorList = tempErrorList[nruns:] 
        timingList.append(avgt)
        errorList.append(avge)

    return (nprocsList, timingList, errorList)