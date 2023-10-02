import os, sys
import numpy as np
import pandas as pd

from qchsleep.datasleep_50 import *
from qchsleep.events_to_csv import get_files_list

def test_name_converter():
    l1 = ['2018 - Cycle 4_QC2',
          '2019 - Cycle 4_QC2',
          '2022 - Cycle 3_QC2']
    l2 = ['QS002_Q2CP_16072012',
          'Q2CP_Q2CP_14032018-2',
          'QC2P_27102023']
    for k,v in enumerate(l1):
        assert name_converter(v) == l2[k]
    for k,v in enumerate(l2):
        assert name_converter(v) == l1[k]

def test_read_v2_hypnograms():
    files = ['data/u-sleep2/U-Sleep 2.0 EEG/C4/Q1CP_01102015_-_C4_hypnogram.txt',
            'data/u-sleep2/U-Sleep 2.0/O2/QS002_Q2CP_16072012_-_O2_hypnogram.txt']
    all_events = read_v2_hypnograms(files)
    assert len(all_events) == 2
    assert len(all_events[0]) == 201
    assert len(all_events[1]) == 201
    assert all_events[0][0] == 'Wake'

def test_merge_summs():
    merge_summs()

def test_merge_heatmaps():
    merge_heatmaps()

def test_add_v2():
    files = get_files_list("data/u-sleep2")[0]

    gens = []
    models = []
    leads = []
    edf_names = []
    for file in files:
        _, gen,model,lead,edf_name = file.split('/')
        edf_name = edf_name.split('_-_')[0]
        gens.append(gen)
        models.append(model)
        leads.append(lead)
        edf_names.append(edf_name)

    data = read_v2_hypnograms(files)
    print(len(edf_names), len(data))
    for i, j in zip(edf_names, data):
        print(i,len(j))
    sys.exit()

def test_gen_stacked_bars():
    gen_stacked_bars()
