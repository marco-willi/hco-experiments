# Assess probability estimates of SWAP against true probabilities

# import modules
from swap.control import MetaDataControl, Control
from swap.mongo import DB
from swap.mongo import Query, Group
from swap.config import Config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# functions
# Extract labels
def getLabelReal(subs, score='score'):
    res = {'actual': [], 'predicted': [], 'prob': []}
    for sub in subs.values():
        res['actual'].append(sub['gold_label'])
        res['predicted'].append(sub['label'])
        res['prob'].append(sub[score])
    return res

# Check data in data base
db = DB()
cfg = Config()
classifications = db.classifications

# process SWAP
control = Control(0.1, 0.5)
control.process()
subs = control.getSWAP().exportSubjectData()
labels = getLabelReal(subs)

# plot reliability of estimates
prob_bins = [x/100 for x in range(0,105,1)]
prob_binned = pd.cut(labels['prob'], prob_bins)
labels['prob_binned'] = prob_binned.get_values()

# group by bins and calculate true reals
labels_df = pd.DataFrame.from_dict(labels)

# remove -1 labels
labels_df = labels_df.query('actual>= 0')

# calculate binned true probabilities
labels_grp = labels_df.groupby(by='prob_binned')['actual'].agg(['sum','count'])
labels_grp['prop'] = labels_grp['sum'] / labels_grp['count']
labels_grp.head

# plot
sns.set_style("whitegrid")
ax = sns.barplot(x='prob_binned', y="prop", data=labels_grp.reset_index(),
                 palette="Blues_d")

#plt.plot([0,1],[0,1],color = "red")
#plt.bar(prob_binned. , labels_grp.prop,align='center')
#ax2 = ax.twinx()
#ax2.plot(ax.get_xticks(),pd.DataFrame([[0,0],[1,1]]),marker='o')
#plt.plot([0, 0], [1, 1], linewidth=2)
ax.set(ylim=(0,1))
fig = ax.get_figure()
fig.set_size_inches(17,17)
fig.savefig("estimated_prob_vs_true_prob_1p.png")
