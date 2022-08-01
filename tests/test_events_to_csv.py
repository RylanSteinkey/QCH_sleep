import pytest
import os, sys
from datetime import datetime

from qchsleep import events_to_csv

def test_get_files_list():
    path = "data/U-Sleep_predictions/"
    try:
        assert os.path.isdir(path)
    except:
        pytest.skip("no data for this test, skipping")
    files = events_to_csv.get_files_list(path)

    # we expect 5 clusters with 3 files each
    assert len(files) == 5
    for f in files:
        assert len(f) == 3

def test_read_time():
    test = "START=2018/10/09-14:19:10 (UTC)"
    readable = events_to_csv.read_time(test)
    assert readable == datetime(2018,10,9,14,19,10)
    test = ['7/02/2022', '11:30:09 PM']
    readable = events_to_csv.read_time(test)
    assert readable == datetime(2022,2,7,23,30,9)

def test_RemLogic_to_pandas():
    path = "data/gold_labels/2022 CONCORDANCE-1-Events 7-feb-2022.txt"
    stime = datetime(2022,2,7,23,30,9)
    l,t = events_to_csv.RemLogic_to_pandas(path)
    assert len(l) > 10
    assert stime == t
