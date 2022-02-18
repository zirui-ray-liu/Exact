import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import pdb
import itertools
import seaborn as sns

FONTSIZE = 14
LABELSIZE = 15
LEGENDSIZE = 14
TEXTSIZE = 20
MARKERSIZE = 64

markers = 'ov*d'
colors = 'rbgm'

font_config = {'font.size': FONTSIZE, 'font.family': 'sans-serif'}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (6.4, 2.8)


fracs = [0.125, 0.25, 0.5, 1.0]
bit = 2
dataset = 'arxiv'
archs = ['gcn', 'sage', 'gcn2']
leg = []
throughout = {}
run_time = {}

for arch in archs:
    throughout[arch], run_time[arch] = [], []
    for frac in fracs:
        precision = f'int{bit}'
        log_path = f'./{dataset}_{arch}_{precision}_frac{frac}.txt'
        lines = open(log_path, 'r').readlines()
        try:
            th = float(lines[-1].split('epoch/s: ')[1][:-2])
            rt = float(lines[-2].split('s/epoch: ')[1][:-2])
            throughout[arch].append(th)
            run_time[arch].append(rt)
        except:
            print(arch, frac)
    bl_path = f'./{dataset}_{arch}_fp32_frac1.0.txt'
    lines = open(bl_path, 'r').readlines()
    th = float(lines[-1].split('epoch/s: ')[1][:-2])
    rt = float(lines[-2].split('s/epoch: ')[1][:-2])
    throughout[arch].append(th)
    run_time[arch].append(rt)
print(throughout)

X = np.arange(5)
seq = ['gcn', 'sage', 'gcn2']

for i in range(1):
    plt.bar(X, throughout[seq[i]])
pdb.set_trace()
plt.savefig(f'test.png', bbox_inches='tight')


# for frac, c, m in zip(fracs, colors, markers):
#     plt.plot(range(len(means[frac])), means[frac], f'-{m}k', color=c, markersize=6,  linewidth=2)
#     plt.xticks(np.arange(3), ['INT2', 'INT4', 'INT8'])
#     # if frac == 0.5:
#     plt.fill_between(range(len(means[frac])), 
#                     np.array(means[frac]) - np.array(stds[frac]), 
#                     np.array(means[frac]) + np.array(stds[frac]), color=c, alpha=0.1)

# plt.plot(range(3), [baselines[dataset]] * 3, color='k')
# leg = [f'D/R = {int(1/frac)}' for frac in fracs] + ['baseline']
# plt.legend(leg, prop={'size': 12}, bbox_to_anchor=(1, 1))
# # if arch == 'sage':
# #     plt.legend(leg, prop={'size': 12}, bbox_to_anchor=(1, 1))
# # if arch == 'sage':
# #     plt.yticks(np.arange(71.0, 72.1, 0.2))
# # elif arch == 'gcn':
# #     plt.yticks(np.arange(71.2, 72.2, 0.2))
# # else:
# #     raise NotImplemented
# filename = f'./{dataset}_{arch}_results_norm{norm}'
# if large_batch:
#     filename += '_large_batch'
# plt.savefig(f'{filename}.png', bbox_inches='tight')
