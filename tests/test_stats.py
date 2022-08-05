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

def test_is_equal():
    # pure
    assert stats.is_equal("WAKE","SLEEP-S0","pure")
    assert stats.is_equal("N1","SLEEP-S1","pure")
    assert stats.is_equal("N2","SLEEP-S2","pure")
    assert stats.is_equal("N3","SLEEP-S3","pure")
    assert stats.is_equal("REM","SLEEP-REM","pure")

    # merge nrem
    assert not stats.is_equal("REM","SLEEP-S3","non_REM_merge")
    assert not stats.is_equal("WAKE","SLEEP-S1","non_REM_merge")
    assert stats.is_equal("N1","SLEEP-S1","non_REM_merge")
    assert stats.is_equal("N1","SLEEP-S2","non_REM_merge")
    assert stats.is_equal("N1","SLEEP-S3","non_REM_merge")
    assert not stats.is_equal("N1","SLEEP-S0","non_REM_merge")
    assert not stats.is_equal("N1","SLEEP-REM","non_REM_merge")

    # wake_sleep
    assert stats.is_equal("WAKE","SLEEP-S0","wake_sleep")
    assert not stats.is_equal("N1","SLEEP-S0","wake_sleep")
    assert stats.is_equal("N1","SLEEP-REM","wake_sleep")


def test_grade():
    df = pd.DataFrame()
    df['C4'] = ['WAKE','WAKE',"N1","N2","N3","REM","WAKE","WAKE"]
    df['F4'] = ['WAKE','WAKE',"N1","N1","REM","REM","WAKE","WAKE"]
    df['O2'] = ['WAKE','WAKE',"WAKE","N2","N2","N2","WAKE","WAKE"]
    df['gold_std'] = ['','','SLEEP-S1','SLEEP-S1','SLEEP-S2','SLEEP-REM','SLEEP-S0','']

    sim, ck = stats.grade(df)
    assert values
