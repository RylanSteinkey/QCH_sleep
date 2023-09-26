import pandas as pd
import numpy as np
import os, sys

from statistics import mode
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from qchsleep.events_to_csv import get_files_list


import seaborn as sns
import matplotlib.pyplot as plt

def is_equal(l,r,rule):
    """
    Determines if 2 strings are equal,
    Acceptable: W,N1,N2,N3,R
    """
    if rule == 'pure':
        if l==r:
            return True
        else:
            return False
    elif rule == 'non_REM_merge':
        if l==r or (l in ['N1','N2','N3'] and r in ['N1','N2','N3']):
            return True
        else:
            return False
    elif rule == 'wake_sleep':
        if l==r or (l in ['N1','N2','N3','R'] and r in ['N1','N2','N3','R']):
            return True
        else:
            return False
    else:
        raise Exception("Rule {} not supported".format(rule))
def load_excel(file):
    """
    Load file into 2 df's; metadata and predictions
    """
    # df is dict(sheet_name, sheet_data)
    df = pd.read_excel(file, sheet_name = None)

    sheets = list(df.keys())
    for sheet in sheets:
        df[sheet] = [df[sheet].iloc[:10], df[sheet].iloc[10:]]
    return sheets, df

def add_gold_stds(sheets, dfs):
    """
    Calculates the gold standards, adds it as a new row
    """
    for sheet in sheets:
        train_status = dfs[sheet][0].values[1]
        trained_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if j=='Trained']

        preds = dfs[sheet][1]
        trained_only_preds = preds[trained_cols].values

        gold_column = [mode(i) for i in trained_only_preds]
        dfs[sheet][1]['gold_std'] = gold_column

    return dfs

def get_perc_ck_per_class(dfs,sheets):
    """
    Calculates % simularity and cohens kappa for each predictor
    as well as means across predictor classes (trained/untrained)

    Returns 2xN array
                {scorer1..scorer13}, Trained, Untrained, C4, F4, O2
    % Simular
    Cohens_k
    """
    rules = ['pure','non_REM_merge','wake_sleep']
    sum_ars = {}
    for rule in rules:
        sum_ars[rule] = {}

    for sheet in sheets:
        train_status = dfs[sheet][0].values[1]
        trained_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if j=='Trained']

        for rule in rules:
            # row0 = % sim, row1 = ck
            sums = pd.DataFrame()
            gold_stds = dfs[sheet][1]['gold_std']
            for col_name in dfs[sheet][1].columns:
                if col_name == 'Sleep' or col_name == 'Unnamed: 23':
                    continue
                #compare column to gold std, get %sim and ck

                col = dfs[sheet][1][col_name]
                perc = np.sum([i==j for i,j in zip(col,gold_stds)])/len(gold_stds)

                #ck_fake_left = [is_equal(i,j,rule) for i,j in zip(col,gold_stds)]
                #ck = cohen_kappa_score(ck_fake_left,[True for i in ck_fake_left]) #might not be quite accurate, using below method instead

                ck = cohen_kappa_score(gold_stds,[i if is_equal(i,j,rule) else j for i,j in zip(gold_stds,col)])
                sums[col_name] = pd.Series([perc,ck])

            #average categories into new column # TODO NEXT
            for group in ['Trained', 'In Training']:
                mean_perc = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j==group and 'U-Sleep' not in i)]].values[0])
                mean_ck = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j==group and 'U-Sleep' not in i)]].values[1])
                sums[group+" Average"] = pd.Series([mean_perc,mean_ck])

            mean_perc = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j in (['Trained', 'In Training']) and 'U-Sleep' not in i)]].values[0])
            mean_ck = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j in (['Trained', 'In Training']) and 'U-Sleep' not in i)]].values[1])
            sums["All Human Average"] = pd.Series([mean_perc,mean_ck])


            sum_ars[rule][sheet] = sums
    return sum_ars

def acc_by_title(summary_arrays, sheets):
    """
    Takes the summary arrays and in a single dataframe,
    returns the accuracy for each type of trainer

    rows = trained status
    cols = study number
    """
    sim_dfs = {}
    ck_dfs = {}
    types = ['Trained Average', 'In Training Average', 'U-Sleep_F4', 'U-Sleep_C4', 'U-Sleep_O2', 'All Human Average']
    for lead in ['C4','F4','O2']:
        for model in ['2.0_EEG_','2.0_']:
            types.append("U-Sleep_"+model+lead)
    for rule in ['pure','non_REM_merge','wake_sleep']:
        ck_df = pd.DataFrame(index=types)
        sim_df = pd.DataFrame(index=types)
        for sheet in sheets:
            sim_data = []
            ck_data = []
            sdf = summary_arrays[rule][sheet]
            for col_name in types:
                try:
                    sim_data.append(sdf[col_name].values[0])
                    ck_data.append(sdf[col_name].values[1])
                except:
                    print('Error loading {} for {}'.format(col_name, sheet))
                    raise

            ck_df[sheet] = pd.Series(ck_data,index=types)
            sim_df[sheet] = pd.Series(sim_data,index=types)


        sim_dfs[rule] = sim_df
        ck_dfs[rule] = ck_df
    return sim_dfs, ck_dfs

def name_converter(name):
    """
    Takes a name as either qsleep or edf format,
    converts the name to the other
    i.e. '2016 - Cycle 3_QC1' is renamed to 'Q1CP_01102015'
    and vice versa
    """
    files = get_files_list("data/u-sleep2")

    edf_names = []
    for file in files:
        data, gen,model,lead,edf_name = files[0][0].split('/')
        edf_name = edf_name.split('_-_')[0]
        edf_names.append(edf_name)

    sheets, dfs = load_excel("data/QSleep_U-Sleep_Data.xlsx")
    sheets = [i.split(')')[1][1:] for i in sheets]

    names_df = pd.read_excel('data/U-Sleep_Naming.xlsx')
    qsleeps = list(names_df['QSleepCycle'])
    edfs = list(names_df['EDF'])

    for i in edf_names:
        if i not in edfs:
            raise Exception('file name {} not found in edfs'.format(i))
    for i in sheets:
        if i not in qsleeps:
            raise Exception('file name {} not found in qsleeps'.format(i))

    if name in edfs:
        indx = edfs.index(name)
        return qsleeps[indx]
    elif name in qsleeps:
        indx = qsleeps.index(name)
        return edfs[indx]
    else:
        raise Exception("The name {} wasnt found in the naming sheet 'data/U-Sleep_Naming.xlsx'")

def read_v2_hypnograms(files):
    all_events = []
    for hypno in files:
        events = []
        with open(hypno) as file:
            try:
                for line_number, line in enumerate(file):
                    line = line.rstrip()
                    if line_number == 0:
                        try:
                            assert line == 'EPOCH=30.0s'
                        except:
                            raise Exception("Epoch not set to 30s")
                    elif line_number == 1:
                        #This line contains start time which isn't validated
                        continue
                    else:
                        if line == 'Wake':
                            events.append('W')
                        elif line == 'REM':
                            events.append('R')
                        else:
                            events.append(line)

            except:
                print("Error when reading file {}".format(file))
                raise
        all_events.append(events)

    return all_events

def add_v2(sheets, dfs):
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

    for data, gen, model, lead, edf in zip(data, gens, models, leads, edf_names):
        try:
            """ we found these
            missing_edf = ['Q1CP_09022015-1',
                           'Q1CP_09022015-2',
                           'Q2CP_01122016']
            if edf in missing_edf:
                continue
            """
            qsleep = name_converter(edf)
            qsleep = '(DataSleep)({})'.format(qsleep)
        except:
            print("edf {} not found in converter document".format(edf))
            raise

            #'(DataSleep)(2017 - Cycle 2_QC1)',
        missing = ['(DataSleep)(2017 - Cycle 4_QC1)',
                   '(DataSleep)(2017 - Cycle 2_QC2)',
                   '(DataSleep)(2017 - Cycle 3_QC2)',
                   '(DataSleep)(2017 - Cycle 2_QC1)']
        if qsleep in missing:
            continue
        name = "_".join(model.split())+'_'+lead
        qsleep_len = len(dfs[qsleep][1]['gold_std'])
        dfs[qsleep][1] = dfs[qsleep][1].reset_index(drop=True)
        dfs[qsleep][1][name] = pd.Series(data[:qsleep_len])

    """
    dfs['(DataSleep)(2016 - Cycle 1_QC1)'][1].to_csv('data/test_v2_del.csv')
    print(dfs['(DataSleep)(2016 - Cycle 1_QC1)'])
    print(dfs['(DataSleep)(2016 - Cycle 1_QC1)'][1].columns)
    sys.exit()
    """
    return dfs

def merge_summs():
    """
    reads saved summary sim and ck csv's
    prints summary stats about each model type
    """
    mdf = pd.DataFrame()
    inx = []
    for rule in ['pure','non_REM_merge','wake_sleep']:
        for stat in ['sim','ck']:
            summs = []
            df = pd.read_csv("data/{}_summ_{}.csv".format(stat,rule),index_col=0)
            inx = df.index
            for row in df.values:
                summs.append("{:.4f}".format(np.mean(row))+" (+/- {:.4f})".format(np.std(row)))
            mdf[stat+'_'+rule] = summs
    mdf.index = inx
    mdf.to_csv('data/summary_stats.csv')

def merge_heatmaps():
    """
    Merges all heatmaps into a single figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    heatmap_files = [f for f in os.listdir('figures/heatmaps') if f.endswith('.png')]
    heatmap_files.sort()

    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    for i, heatmap_file in enumerate(heatmap_files):
        row = i % 3
        col = i // 3


        if row == 0 and col == 1:
            col = 2
        elif row == 0 and col == 2:
            col = 1

        if col == 1:
            col = 3
        elif col == 3:
            col = 1

        heatmap_path = os.path.join('figures/heatmaps', heatmap_file)
        heatmap = mpimg.imread(heatmap_path)

        axs[row, col].imshow(heatmap)
        axs[row, col].axis('off')

    """
    for i in range(3):
        axs[i, 1], axs[i, 3] = axs[i, 3], axs[i, 1]
    """



    # Optionally, you can save the merged figure
    plt.savefig('figures/merged_heatmaps.png', bbox_inches='tight', dpi=400)


def print_figs(ck_sum):
    """
    Prelim figure generation for a 5x50 array
    """
    ax = sns.barplot(data=ck_sum.T, errorbar='ci')
    ax.set(xlabel = "Predictor", ylabel="Cohen's Kappa (n=50)")
    plt.show()

def gen_confusion_matrix(sheets, dfs, scorer):
    """
    Generates confusion matrices across all 50 studies
    scorer in ['Trained', 'In Training', 'U-Sleep_F4', 'U-Sleep_C4', 'U-Sleep_O2','All Human Average']
    """
    gold = []
    pred = []

    for sheet in sheets:
        #print(dfs[sheet])
        train_status = dfs[sheet][0].values[1]
        if 'U-Sleep' in scorer:
            to_keep_cols = [scorer]
        else:
            if scorer == 'All Human Average':
                to_keep_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if (j in (['Trained', 'In Training']) and 'U-Sleep' not in i)]
            else:
                to_keep_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if (j==scorer and 'U-Sleep' not in i)]
        pred_df = dfs[sheet][1]
        for col in to_keep_cols:
            try:
                for i in pred_df[col]:
                    pred.append(i)
                for i in pred_df['gold_std']:
                    gold.append(i)
            except:
                print("Error finding {} in sheet {}".format(col, sheet))
                raise

    labels = ['W','N1','N2','N3','R']
    # (y_true, y_pred)  == (gold, pred)
    conf_matrix = confusion_matrix(gold, pred, labels = labels)

    # Calculate accuracy percentages
    cm_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using Seaborn or Matplotlib
    plt.figure(figsize=(6, 4))

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Reds', cbar=False)

    # Add accuracy percentages to the plot
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell_color = conf_matrix[i, j] / np.max(conf_matrix)
            text_color = 'white' if cell_color > 0.5 else 'black'
            plt.text(j+0.5, i+0.25, f"{cm_percent[i, j]*100:.2f}%",
                 ha='center', va='center', color=text_color)


    plt.title("Confusion Matrix")
    plt.xlabel("Predicted by {}".format(scorer))
    plt.ylabel("Gold Standard")

    plt.xticks(np.arange(len(labels))+0.5, labels)
    plt.yticks(np.arange(len(labels))+0.5, labels)

    if not os.path.exists('figures/heatmaps/'):
         os.makedirs('figures/heatmaps/')
    plt.savefig("figures/heatmaps/{}.png".format(scorer))
    plt.clf()



def main():
    sheets, dfs = load_excel("data/QSleep_U-Sleep_Data.xlsx")
    #print(dfs[0].columns,dfs[1].columns)
    dfs = add_gold_stds(sheets, dfs)

    dfs = add_v2(sheets, dfs)

    summary_arrays = get_perc_ck_per_class(dfs,sheets)

    """
    rule = "pure"
    sheet = '(DataSleep)(2016 - Cycle 1_QC1)'
    summary_arrays[rule][sheet].to_csv("data/summ_arr_test_del_me.py")
    sys.exit()
    """

    sim_sum, ck_sum = acc_by_title(summary_arrays, sheets)
    for rule in ['pure','non_REM_merge','wake_sleep']:
        sim_sum[rule].to_csv("data/sim_summ_{}.csv".format(rule))
        ck_sum[rule].to_csv("data/ck_summ_{}.csv".format(rule))


    #print(dfs)
    #test dfs
    with pd.ExcelWriter('data/del_test.xlsx') as writer:
        for sheet in sheets:
            dfs[sheet][1].to_excel(writer, sheet_name=sheet)
    sys.exit()
    
    #print_figs(ck_sum['pure'])
    for scorer in ['Trained', 'In Training', 'U-Sleep_F4', 'U-Sleep_C4', 'U-Sleep_O2', 'All Human Average']:
        gen_confusion_matrix(sheets, dfs, scorer)
    for lead in ['C4','F4','O2']:
        for model in ['2.0_EEG_','2.0_']:
            gen_confusion_matrix(sheets, dfs, "U-Sleep_"+model+lead)

    merge_summs()


    #plot



if __name__ == "__main__":
    main()
