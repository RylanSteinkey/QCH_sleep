import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import os, sys

from qchsleep.research_comp import read_scorings_for_figs
from qchsleep.research_comp import grade_scorings_for_figs

date_groups = ['30-may-2018','7-feb-2022','9-oct-2018','2-may-2018','18-feb-2021','3-mar-2015']
proper_dates = ['May 30th, 2018','February 7th, 2022','October 9th, 2018','May 2nd, 2018','February 18th, 2021','March 3rd, 2015']
ordered = ['March 3rd, 2015', 'May 2nd, 2018', 'May 30th, 2018', 'October 9th, 2018', 'February 18th, 2021', 'February 7th, 2022']


def staff_classifier(name, staff_dict):
    """
    Takes a name and a staff dict
    returns what type of staff they are
    """
    docs = [i.split(' ')[0].upper() for i in staff_dict['doctors']]
    nurses = [i.split(' ')[0].upper() for i in staff_dict['nurses']]
    scientists = [i.split(' ')[0].upper() for i in staff_dict['scientists']]
    AI = [i.split(' ')[0].upper() for i in staff_dict['USleep']]
    goldc = [i.split(' ')[0].upper() for i in staff_dict['goldc']]

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
    elif name in goldc:
        return "Gold Contributor"
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

    print(data)

    data = pd.melt(data,'Job Title',var_name='test')
    print(data)
    sys.exit()
    sns.set_theme(style='whitegrid')
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    sns.stripplot(data = data, x='value',y='test',hue = "Job Title",
                  dodge=True, alpha=0.7, zorder=1) #, legend=False
    sns.pointplot(data = data, x='value',y='test',hue = "Job Title",
                  join=False, dodge=.8-.8/4, palette='dark',
                  markers='d',scale=0.75,errorbar=('ci',0),errwidth=0)

    plt.show()

def u_sleep_fig1():
    """
    Multifacet staff comparison.
    Compares Nurses to scientists to doctors to AI;
    all against the gold standard

    6 columns, each is a different test type
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

    #grading = grading[1] #delete this, make it a loop, only doing 1 concordance for now

    for date_group, conc_data in zip(proper_dates, grading):
        for emp in conc_data.keys():
            if emp.upper() not in ['GOLD','GOLD_1','GOLD_2']:
                row = []
                data = conc_data[emp]
                for i in data[0]:
                    row.append(i)
                for j in data[1]:
                    row.append(j)
                row.append(staff_classifier(emp,staff_classes))
                row.append(date_group)
                all_data.append(row)
    data = pd.DataFrame(data=all_data,
                        columns=['Similarity\n(All Stages)','Similarity\n(REM VS NREM VS Wake)','Similarity\n(Sleep VS Wake)',
                                 "Cohen's Kappa\n(All Stages)","Cohen's Kappa\n(REM VS NREM VS Wake)","Cohen's Kappa\n(Sleep VS Wake)",
                                 "Sleep Stager","Concordance Date"])


    """
    Data needs 3: job title, test_type, and value of test result
    """
    new_data = []
    for date_group in ordered:
        for job in ["Gold Contributor","Scientist",'Nurse','Doctor','USleep Lead']:
            data_df = data[(data['Concordance Date']==date_group) & (data['Sleep Stager']==job)]
            row = [data_df[i].mean() for i in data_df.columns[:6]]
            row.append(job)
            new_data.append(row)

    new_data = pd.DataFrame(data = new_data, columns=[
            'Similarity\n(All Stages)','Similarity\n(REM VS NREM VS Wake)','Similarity\n(Sleep VS Wake)',
             "Cohen's Kappa\n(All Stages)","Cohen's Kappa\n(REM VS NREM VS Wake)",
             "Cohen's Kappa\n(Sleep VS Wake)","Sleep Stager"])

    #data = data[data['Concordance Date']=='May 30th, 2018'] #delete this
    #data = data.drop(columns=['Concordance Date'])

    data = pd.melt(new_data,'Sleep Stager',var_name='Metric')

    data = data.rename(columns={'value':"Accuracy (Percent Similarity or Cohen's Kappa)"})
    sns.set_theme(style='whitegrid')
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    sns.stripplot(data = data, y="Accuracy (Percent Similarity or Cohen's Kappa)",x='Metric',hue = "Sleep Stager",
                  dodge=True, alpha=0.7, zorder=1, orient='v') #, legend=False


    # seaborn will ignore labels with a _ preceding, them, dont want to double
    # up on the legend
    data['Sleep Stager'] = ['_'+i for i in data['Sleep Stager']]
    sns.pointplot(data = data, y="Accuracy (Percent Similarity or Cohen's Kappa)",x='Metric',hue = "Sleep Stager",
                  join=False, dodge=.8-.8/4, palette='dark', orient='v',
                  markers='d',scale=0.75,errorbar=('sd'),errwidth=0.25)

    #plt.legend(bbox_to_anchor=(1.0,1),loc=2)
    sns.move_legend(ax, 'lower right')

    # BLACK BARS
    #for x in range(0, len(data['test'].unique()) - 1):
        #plt.plot([x + 0.5, x + 0.5], [0.5, data['value'].max()],c='black')

    # REPLACE Concordance Date with 'test'
    for x in range(0, len(data['Metric'].unique())):
        plt.axvspan(x - 0.5, x + 0.5, facecolor='black', alpha=[0.2 if x%2 == 0 else 0.05][0])

    #plt.savefig('figures/sp_pp.png')
    plt.ylim(0,1)
    plt.show()

def u_sleep_fig2and3(test_type):
    """
    Compares Nurses to scientists to doctors to AI;
    all against the gold standard

    6 columns, each is a different concordance
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

    #grading = grading[1] #delete this, make it a loop, only doing 1 concordance for now

    for date_group, conc_data in zip(proper_dates, grading):
        for emp in conc_data.keys():
            if emp.upper() not in ['GOLD','GOLD_1','GOLD_2']:
                row = []
                data = conc_data[emp]
                """for i in data[0]:
                    row.append(i)
                for j in data[1]:
                    row.append(j)"""
                if test_type == "Cohen's Kappa\n(All Stages)":
                    row.append(data[1][0])
                elif test_type == "Cohen's Kappa\n(REM VS NREM VS Wake)":
                    row.append(data[1][1])
                elif test_type == 'Similarity\n(All Stages)':
                    row.append(data[0][0])
                else:
                    raise Exception("rules for test {} not written yet".format(test_type))
                row.append(staff_classifier(emp,staff_classes))
                row.append(date_group)
                all_data.append(row)
    """data = pd.DataFrame(data=all_data,
                        columns=['Similarity\n(All Stages)','Similarity\n(REM VS NREM VS Wake)','Similarity\n(Sleep VS Wake)',
                                 "Cohen's Kappa\n(All Stages)","Cohen's Kappa\n(REM VS NREM VS Wake)","Cohen's Kappa\n(Sleep VS Wake)",
                                 "Sleep Stager","Concordance Date"])"""
    data = pd.DataFrame(data=all_data,
                        columns=[test_type, 'Sleep Stager', "Concordance Date"])

    """
    Data needs 3: job title, conc date, and value of test result
    """
    #print(data)
    #data = pd.melt(data,'Sleep Stager',var_name='Concordance Date')
    #data = data.rename(columns={'value':"Cohen's Kappa"})
    #print(data)
    sns.set_theme(style='whitegrid')
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    """sns.stripplot(data = data, y='value',x='test',hue = "Sleep Stager",
                  dodge=True, alpha=0.7, zorder=1, orient='v') #, legend=False"""
    sns.stripplot(data = data, y=test_type,x='Concordance Date',hue = "Sleep Stager",
                  dodge=True, alpha=0.7, zorder=1, orient='v', order = ordered)

    # seaborn will ignore labels with a _ preceding, them, dont want to double
    # up on the legend
    data['Sleep Stager'] = ['_'+i for i in data['Sleep Stager']]
    sns.pointplot(data = data, y=test_type,x='Concordance Date',hue = "Sleep Stager",
                  join=False, dodge=.8-.8/4, palette='dark', orient='v',
                  markers='d',scale=0.75,errorbar=('sd'),errwidth=0.25, order = ordered)

    #plt.legend(bbox_to_anchor=(1.0,1),loc=2)
    sns.move_legend(ax, 'lower right')

    # BLACK BARS
    #for x in range(0, len(data['test'].unique()) - 1):
        #plt.plot([x + 0.5, x + 0.5], [0.5, data['value'].max()],c='black')

    # REPLACE Concordance Date with 'test'
    for x in range(0, len(data['Concordance Date'].unique())):
        plt.axvspan(x - 0.5, x + 0.5, facecolor='black', alpha=[0.2 if x%2 == 0 else 0.05][0])

    #plt.savefig('figures/sp_pp.png')
    plt.ylim(0,1)
    plt.show()

def run_all_figures():
    #multifacet_research_compare()
    # 6 columns, each is a test. each plot a concordance average
    # i.e. a single dot under nurses is the average score of the nurses for that plot
    #u_sleep_fig1()

    #each concordance is a column, shows cohens cohen_kappa_score for all stages
    u_sleep_fig2and3("Cohen's Kappa\n(All Stages)")
    #u_sleep_fig2and3('Similarity\n(All Stages)')
    #each concordance is a column, shows cohens cohen_kappa_score for when all nrem stages are merged
    #u_sleep_fig2and3("Cohen's Kappa\n(REM VS NREM VS Wake)")

    # extra


if __name__ == "__main__":
    run_all_figures()
