import glob
import json
import pandas as pd
import seaborn as sns

results_files = glob.glob('/Users/naveenmysore/Documents/plots/csdi/recent/*.json')
auroc_results = dict()
columns = ['trajectory_length', 'number_of_simulations', 'num_of_particles', 'data_size', 'tau', 'p_threshold', 'auroc']
auroc_results = {label: [] for label in columns}
for json_file in results_files:
    with open(json_file) as jfile:
        jresults = json.load(jfile)
        for col_name in jresults.keys():
            auroc_results[col_name].append(jresults[col_name])
        print(jresults)
df = pd.DataFrame(auroc_results)
print(df.head())