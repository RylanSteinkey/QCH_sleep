import pytest
import pandas as pd
import numpy as np

from qchsleep import stats


def test_grade():
    data = "data/U-Sleep-v-Gold.xlsx"
    xlsx = pd.ExcelFile(data)
    df = pd.read_excel(xlsx, '7-feb-2022')

    res = stats.grade(df)

    print(res)
    #assert False

def test_converter():
    vals = ["SLEEP-REM","SLEEP-REM","SLEEP-REM","SLEEP-S2","SLEEP-S2"]
    inv_vals = ["REM","REM","REM","N2","N2"]

    inv = stats.converter(vals)

    for i,j in zip(inv, inv_vals):
        assert i==j

    double_inv = stats.converter(inv)

    for i,j in zip(vals, double_inv):
        assert i==j

def test_ck_rep():
    lc = ["SLEEP-REM","SLEEP-REM","SLEEP-REM","SLEEP-S2","SLEEP-S2"]
    rc = ["REM","REM","N1","N1","N2"]

    fake_rc = stats.ck_prep(lc,rc,'non_REM_merge')

    print(fake_rc)

    for i,j in zip(fake_rc, ['SLEEP-REM', 'SLEEP-REM', 'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S2']):
        assert i == j
