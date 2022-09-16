from qchsleep import research_comp
from qchsleep import events_to_csv
import pytest
import os, sys

def test_read_job_title_yaml():
    try:
        assert os.path.isfile("data/staff.yaml")
    except:
        pytest.skip("no data for this test, skipping")

    titles = research_comp.read_job_title_yaml()
    assert len(titles.keys())==3

def test_scores_to_df():
    files = events_to_csv.get_files_list("data/researcher_reports")

def test_all():
    outu = 'data/researcher_predictions.xlsx'
    print(dfs)
