import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import os, sys

from qchsleep.research_comp import read_scorings_for_figs
from qchsleep.research_comp import grade_scorings_for_figs

date_groups = ['30-may-2018','7-feb-2022','9-oct-2018','2-may-2018','18-feb-2021','3-mar-2015']

def staff_classifier(name, staff_dict):
    """
    Takes a name and a staff dict
    returns what type of staff they are
    """
    docs = [i.split(' ')[0].upper() for i in staff_dict['doctors']]
    nurses = [i.split(' ')[0].upper() for i in staff_dict['nurses']]
    scientists = [i.split(' ')[0].upper() for i in staff_dict['scientists']]
    AI = [i.split(' ')[0].upper() for i in staff_dict['USleep']]

    for rem_sym in [' ','-','_']:
        name = name.split(rem_sym)[0]
    name = name.upper()

    if name == 'Gold':
        raise Exception("Asked to classify gold in staff classifier")
    elif name in docs:
        return "Doctor"
    elif name in nurses:
        return "Nurse"
    elif name in scientists:
        return "Scientist"
    elif name in AI:
        return "USleep Lead"
    else:
        raise Exception("classifier doesnt know who {} is".format(name))

def multifacet_research_compare():
    """
    Multifacet staff comparison.
    Compares Nurses to scientists to doctors to AI;
    all against the gold standard
    """

    stages = read_scorings_for_figs()
    grading = grade_scorings_for_figs(stages)

    with open('data/staff.yaml') as staff:
        staff_classes = yaml.safe_load(staff)
    staff_classes['USleep'] = ['O2','F4','C4']

    all_data = []

    # want where 6 columns are 6 test params
    # i.e. [pure, rem, wake/sleep]*[sim, ck]
    # 7th column that denotes which staff type they are

    grading = grading[0] #delete this, make it a loop, only doing 1 concordance for now

    for emp in grading.keys():
        if emp not in ['Gold','Gold_1']:
            row = []
            data = grading[emp]
            for i in data[0]:
                row.append(i)
            for j in data[1]:
                row.append(j)
            row.append(staff_classifier(emp,staff_classes))
            all_data.append(row)
    data = pd.DataFrame(data=all_data,
                        columns=['Similarity\n(All Stages)','Similarity\n(REM VS NREM VS Wake)','Similarity\n(Sleep VS Wake)',
                                 "Cohen's Kappa\n(All Stages)","Cohen's Kappa\n(REM VS NREM VS Wake)","Cohen's Kappa\n(Sleep VS Wake)","Job Title"])

    data = pd.melt(data,'Job Title',var_name='test')
    sns.set_theme(style='whitegrid')
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    sns.stripplot(data = data, x='value',y='test',hue = "Job Title",
                  dodge=True, alpha=0.7, zorder=1) #, legend=False
    sns.pointplot(data = data, x='value',y='test',hue = "Job Title",
                  join=False, dodge=.8-.8/4, palette='dark',
                  markers='d',scale=0.75,errorbar=('ci',0),errwidth=0)

    plt.show()


def run_all_figures():
    multifacet_research_compare()

if __name__ == "__main__":
    run_all_figures()
