import matplotlib.pyplot as plt
import json
import os
import csv
import matplotlib
import bisect
from matplotlib import rcParams
import numpy as np


rcParams['font.size'] = 10
plt.rcParams.update({
    'xtick.labelsize': 10,
    'axes.labelsize': 10,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
})

legend_position = (0.45, -0.85)
figuresize = (4.6,4.6)
def value2rank(values,smaller_highter_rank=True):
    if isinstance(values,list):
        sorted_pairs = sorted(enumerate(values), key=lambda x: x[1], reverse=not smaller_highter_rank)

        # Prepare a list for ranks
        ranks = [0] * len(values)

        for rank, (idx, _) in enumerate(sorted_pairs, start=1):
            ranks[idx] = rank
    if isinstance(values,dict):
        sorted_pairs = sorted(values.items(), key=lambda x: x[1], reverse=not smaller_highter_rank)

        # Prepare a list for ranks
        ranks = {}
        for rank, (key, _) in enumerate(sorted_pairs, start=1):
            ranks[key] = rank

    return ranks

def find_next_indices(sorted_values, checkpoints):
    result = []
    for cp in checkpoints:
        idx = bisect.bisect_left(sorted_values, cp)
        if idx < len(sorted_values):
            result.append(idx)
        else:
            result.append(None)
    return result

def insert_points(x, y, x_new, y_new):
    for xi, yi in zip(x_new, y_new):
        insert_pos = next((i for i, val in enumerate(x) if xi < val), len(x))
        x.insert(insert_pos, xi)
        y.insert(insert_pos, yi)
    return x, y


def get_all_subfolders(folder_path,order_index=0):
    subfolders = [name for name in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, name))]
    dis_str = 'Order'+str(order_index)
    subfolders = [name for name in subfolders if dis_str in name or '_ST' in name]
    return subfolders


def build_fig(titles,minx,maxx):
    plt.cla()
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(len(titles), 1, figsize=figuresize, sharex=True,tight_layout=True)
    for i in range(len(titles)):
        if titles[i] == 'Ring':
            axes[i].set_ylabel(r'$T_{}$'.format(i+1) + '=R')
        elif titles[i] == 'Grid':
            axes[i].set_ylabel(r'$T_{}$'.format(i+1) +'=G')
        else:
            axes[i].set_ylabel(r'$T_{}$='.format(i+1) +titles[i])
        if only_final_model:
            axes[i].set_xlim([minx, maxx+5])
        else:
            axes[i].set_xlim([minx, maxx+5])
        # axes[i].grid(True)
        axes[i].grid(axis='x')

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    axes[-1].set_xlabel("Training Epoch")

    return fig, {titles[i]:axes[i] for i in range(len(titles))}




    

def load_json_files(file_paths,check_test_not_finish=False):
    if isinstance(file_paths,dict):
        data = {}
        for key,value in file_paths.items():
            if not os.path.exists(value):
                # print("test result file not exist: ",value)
                continue
            with open(value, "r") as file:
                # data.append(json.load(file))
                temp = json.load(file)
                data[key] = {}
                for epoch in temp.keys():
                    if int(epoch) < minx:
                        continue
                    if not check_test_not_finish or len(temp[epoch].keys()) == len(distribution_order):
                        data[key][epoch]=temp[epoch]
    elif isinstance(file_paths,str):
        if not os.path.exists(file_paths):
            # print("test result file not exist: ",file_paths)
            return {}
        with open(file_paths, "r") as file:
            # data.append(json.load(file))
            data = {}
            temp=json.load(file)
            if not check_test_not_finish:
                data = temp
            else:
                for epoch in temp.keys():
                    if int(epoch) < minx:
                        continue
                    if not check_test_not_finish or len(temp[epoch].keys()) == len(distribution_order):
                        data[epoch] = temp[epoch]
                    else:
                        # print('epoch {} not all distribution tested'.format(epoch))
                        pass
    else:
        raise ValueError('json file not exist: ', file_paths)
    return data

def compute_avg_forgetting(data,distribution_order,bounds,isKB=False):
    
    score_name = score_type
    # forgettings = []
    forgettings = {}
    for i in range(len(distribution_order)-1):
        end_epochs = [200 * (n) for n in range(len(distribution_order)) if n > i]
        distribution = distribution_order[i]

        if gap:
            obj_range = bounds[distribution]['min']
        else: # min max norm
            obj_range = bounds[distribution]['max'] -  bounds[distribution]['min']
        final_objs = [data[str(x)][distribution][score_name] for x in end_epochs]
        # best_obj = min(final_objs)
        best_obj = final_objs[0]
        end_obj = final_objs[-1]
        # forgettings.append(max(0,end_obj - best_obj)/obj_range)
        forgettings[distribution] = max(0,end_obj - best_obj)/obj_range
    return forgettings

def compute_max_forgetting(data,distribution_order,bounds,isKB=False):
    
    score_name = score_type
    # forgettings = []
    forgettings = {}
    for i in range(len(distribution_order)-1):

        end_epochs = [200 * (n) for n in range(len(distribution_order)) if n > i]
        distribution = distribution_order[i]

        if gap:
            obj_range = bounds[distribution]['min']
        else: # min max norm
            obj_range = bounds[distribution]['max'] -  bounds[distribution]['min']
        # obj_range = bounds[distribution]['max'] -  bounds[distribution]['min']
        final_objs = [data[str(x)][distribution][score_name] for x in end_epochs]
        # best_obj = min(final_objs)
        best_obj = final_objs[0]
        worst_ojb = max(final_objs)
        # forgettings.append(max(0,worst_ojb - best_obj)/obj_range)
        forgettings[distribution] = max(0,worst_ojb - best_obj)/obj_range
    return forgettings

def compute_plasticity(data,distribution_order,bounds,isKB=False):
    
    end_epochs = [200 * (n+1) for n in range(len(distribution_order))]
    score_name = score_type
    # forgettings = []
    learn_performance = {}
    for i in range(len(distribution_order)):
        distribution = distribution_order[i]
        min_obj = bounds[distribution]['min']
        max_obj = bounds[distribution]['max']
        current_end_epoch = end_epochs[i]
        
        if gap: # gap
            learn_performance[distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (min_obj)
        else:        # min max norm
            learn_performance[distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (max_obj-min_obj)
        # learn_performance[distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (max_obj-min_obj)
    return learn_performance


def compute_zero_shot_generalization(data,distribution_order,bounds,isKB=False):
    
    end_epochs = [200 * (n+1) for n in range(len(distribution_order))]
    score_name = score_type
    # forgettings = []
    generalization = {}
    for i in range(len(distribution_order)-1): # for each learned task
        single_task_generalization = []
        learned_distribution = distribution_order[i]
        current_end_epoch = end_epochs[i]
        # performance on the next tasks
        j = i + 1
        distribution = distribution_order[j]
        min_obj = bounds[distribution]['min']
        max_obj = bounds[distribution]['max']     
        # generalization[learned_distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (max_obj-min_obj)
        if gap: # gap
            generalization[learned_distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (min_obj)
        else:        # min max norm
            generalization[learned_distribution] = (data[str(current_end_epoch)][distribution][score_name] - min_obj) / (max_obj-min_obj)
        
    return generalization

def compute_avg_rank_and_normalized_value(data,folder,bounds):
    from collections import defaultdict

    dimensions = distribution_order
    for k,v in data.items():
        dimensions = v.keys()
        break
    ranks_by_name = defaultdict(list)
    norm_by_name = defaultdict(list)

    for dim in dimensions:
        values = [(name, data[name][dim][-1]) for name in data]
        values.sort(key=lambda x: x[1])
        current_rank = 1
        ranks = {}
        for i, (name, val) in enumerate(values):
            if i > 0 and val == values[i-1][1]:
                ranks[name] = current_rank
            else:
                current_rank = i + 1
                ranks[name] = current_rank

        raw_vals = [val for _, val in values]
        for name, raw_val in values:
            if bounds is None:
                ranks_by_name[name].append(ranks[name])
                norm_by_name[name].append(raw_val)
            else:
                vmin = bounds[dim]['min']
                vmax = bounds[dim]['max']
                ranks_by_name[name].append(ranks[name])
      
                if gap:
                    norm_val = (raw_val - vmin) / (vmin)
                else:
                    if vmax > vmin:
                        norm_val = (raw_val - vmin) / (vmax-vmin)
                    else:
                        norm_val = 0.0
                norm_by_name[name].append(norm_val)
            
    result_list = []
    for name in data:
        avg_rank = sum(ranks_by_name[name]) / len(ranks_by_name[name])
        avg_norm = sum(norm_by_name[name]) / len(norm_by_name[name])
        result_list.append((name, avg_rank, ranks_by_name[name], avg_norm, norm_by_name[name]))

 
    result_list.sort(key=lambda x: x[1])

    summary = {}

    print("AvgRank\tAvgGap\tRanks                \tGap              \t\t      \t\t\tName")

    for raw_name, avg_rank, rank_list, avg_norm, norm_list in result_list:
        name = raw_name
        rank_str = str(rank_list)
        norm_rounded = [round(v, 4) for v in norm_list]
        norm_str = str(norm_rounded)
        summary[name] = {
            'avg_rank': f"{avg_rank:.1f}",
            'avg_norm': f"{avg_norm:.4f}",
            'rank_list': rank_str,
            'norm_list': norm_str
        }
        print(f"{avg_rank:.1f}\t{avg_norm:.4f}\t{rank_str}\t{norm_str}            \t{name}")

    return summary


orders = [
    ['E', 'C', 'Grid', 'U', 'Ring', 'GM'],
    ['U', 'GM', 'E', 'Ring', 'Grid', 'C'],
    ['E', 'Grid', 'Ring', 'C', 'U', 'GM'],
    ['Grid', 'GM', 'E', 'U', 'Ring', 'C'],
    ['Grid', 'C', 'Ring', 'U', 'GM', 'E']
    ]

dfolder = './result/'
task_interval = 200
only_final_model = False
bsf_model = False
remove_unseen = True
gap = True # true: no normalization, false: normalization based on max-min
K=3

ablation = False

use_instance_augment = False
score_type = 'score'
cut_name = True
all_summary = {}
all_plasticity_summary = {}
all_avg_forget_summary = {}
all_max_forget_summary = {}
all_generalization_summary = {}

output_fodler = './'
if ablation:  output_fodler += 'figures_rebuttal_ablation/'
else: output_fodler += 'figures_rebuttal/'
if not os.path.exists(output_fodler):
    os.mkdir(output_fodler)


skip_list = []

if ablation:
    skip_list = []


skip_order = []


# -------  find min and max avg obj for each tested task -------------
bounds = {}
for order_index in range(len(orders)): 
    distribution_order = orders[order_index]
    order_str = distribution_order[0]
    for i in distribution_order[1:]:
        order_str +=  '->' + i
    all_runname = get_all_subfolders(dfolder,order_index)
    if cut_name:runnames = {name:'-'.join(('_'.join(name.split('_')[3:])).split('-')[:-1]) for name in all_runname}
    else: runnames = {name:name for name in all_runname}
    if len(runnames) == 0:
        print('no runname find')
        continue
    for runname in runnames:
        if runnames[runname] in skip_list:
            continue
        if os.path.exists(dfolder + runname+ '/quick_test_performance.json'):
            data = load_json_files(dfolder + runname+ '/quick_test_performance.json',check_test_not_finish=False)
            try:
                for distribution in distribution_order:
                    if use_instance_augment:
                        all_objs = [v['aug_score'] for dict in data.values() for k,v in dict.items() if k == distribution]
                    else:
                        all_objs = [v['score'] for dict in data.values() for k,v in dict.items() if k == distribution]
                    min_obj = min(all_objs)
                    max_obj = max(all_objs)
                    if not distribution in bounds:
                        bounds[distribution]={'min':min_obj,'max':max_obj}
                    else:
                        if min_obj < bounds[distribution]['min']:
                            bounds[distribution]['min'] = min_obj
                        if max_obj > bounds[distribution]['max']:
                            bounds[distribution]['max'] = max_obj
            except ValueError: pass
print(bounds)
# ---


if K<6:
    orders = [orders[i][:K] for i in range(len(orders))]
    print(orders)



if ablation: temp = [0]
else: temp = range(len(orders))
for order_index in temp:
    if order_index  in skip_order:
        continue
    distribution_order = orders[order_index]
    order_str = distribution_order[0]
    for i in distribution_order[1:]:
        order_str +=  '->' + i
    print(f'\n\n\n ---------------- order {order_index} ({order_str}) ---------------------')
    
    all_runname = get_all_subfolders(dfolder,order_index)
    if len(all_runname) == 0:
        print('no runname find')
        continue
    
    
    if cut_name:runnames = {name:'-'.join(('_'.join(name.split('_')[3:])).split('-')[:-1]) for name in all_runname}
    else: runnames = {name:name for name in all_runname}

    from collections import defaultdict
    value_to_keys = defaultdict(list)
    for k, v in runnames.items():
        value_to_keys[v].append(k)

    duplicates = {val: keys for val, keys in value_to_keys.items() if len(keys) > 1}
    if duplicates:
        print("!! duplicate run:")
        for val, keys in duplicates.items():
            print(f" {val} -> keys: {keys}")


    names = list(runnames.values())
    n = len(names)
    
    color_table = matplotlib.colormaps.get_cmap('tab20').resampled(n) 
    runnames = dict(sorted(runnames.items()))
    colors = {name: color_table(i) for i, name in enumerate(names)}


    minx = 199

        
    folder = output_fodler +  f'Order{order_index}/'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    fig,axes = build_fig(distribution_order,minx=minx,maxx=task_interval*len(distribution_order)+10)

    avg_forgetting_all = {}
    max_forgetting_all = {}
    plasticity_all = {}
    final_results = {}
    generalization_all = {}
    for runname in sorted(runnames.keys(), reverse=True):
        if runname in skip_list or runnames[runname] in skip_list:
            print('skip: ',runname,runnames[runname])
            continue

        if True:
            if not os.path.exists(dfolder + runname + '/quick_test_performance.json'):
                print(' test result file not exist: ',dfolder + runname + '/quick_test_performance.json')
                continue
            # print(runname)
            result = {}
            data = load_json_files(dfolder + runname + '/quick_test_performance.json')
            forgets = compute_avg_forgetting(data,distribution_order,bounds=bounds,isKB=False)
            avg_forgetting_all[runnames[runname]] = {k:[v] for k,v in forgets.items()}
            max_forgets = compute_max_forgetting(data,distribution_order,bounds=bounds,isKB=False)
            max_forgetting_all[runnames[runname]] = {k:[v] for k,v in max_forgets.items()}
            plasticity = compute_plasticity(data,distribution_order,bounds=bounds,isKB=False)
            plasticity_all[runnames[runname]] = {k:[v] for k,v in plasticity.items()}
            generalization = compute_zero_shot_generalization(data,distribution_order,bounds=bounds,isKB=False)
            generalization_all[runnames[runname]] = {k:[v] for k,v in generalization.items()}
            
            for i in range(len(distribution_order)):
                distribution = distribution_order[i]
                epoch_list = list(data.keys())
                epoch_list = sorted([int(x) for x in epoch_list])
                if only_final_model:
                    x_values = [x for x in epoch_list if (x) % task_interval == 0]
                else:
                    x_values = [x for x in epoch_list]
                if remove_unseen:
                    x_values = [x for x in x_values if  (x)/task_interval > i] 
                scores = [data[str(x)][distribution][score_type] for x in x_values]
                result[distribution] = scores

                if True:
                    axes[distribution].plot(
                        x_values,
                        scores,
                        marker=',',
                        label=runname,
                        color=colors[runnames[runname]] if runnames[runname] in colors.keys() else None,
                        linestyle='-'
                    )
                
            final_results[runnames[runname]]=result
    plt.legend()
    plt.tight_layout()    
    

    tempstr = ''
    if remove_unseen:
        tempstr+='_nounseen'
    plt.savefig(folder+str(K)+'_compare'+tempstr+f'_O{order_index}.pdf')
    
    print('     final model obj')
    summary = compute_avg_rank_and_normalized_value(final_results,folder,bounds)
    print('     avg forget')
    avg_forget_summary = compute_avg_rank_and_normalized_value(avg_forgetting_all,folder,None)
    print('     max forget')
    max_forget_summary = compute_avg_rank_and_normalized_value(max_forgetting_all,folder,None)
    print('     plasticity')
    plasticity_summary = compute_avg_rank_and_normalized_value(plasticity_all,folder,None)
    print('     generalization')
    generalization_summary = compute_avg_rank_and_normalized_value(generalization_all,folder,None)
    print()
    

    all_summary[order_index]=summary
    all_avg_forget_summary[order_index] = avg_forget_summary
    all_max_forget_summary[order_index] = max_forget_summary
    all_plasticity_summary[order_index] = plasticity_summary
    all_generalization_summary[order_index] = generalization_summary

    with open(folder+"/"+str(K)+"_summary.csv", mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Name", "AvgNorm",  "AvgFinalForget", "AvgMaxForget", "AvgPlasticity", "AvgGeneralization"])
        for name in summary.keys():
            writer.writerow([
                name,
                "{:.1f}".format(float(summary[name]['avg_norm'])* 1000),
                "{:.1f}".format(float(avg_forget_summary[name]['avg_norm'])* 1000),
                "{:.1f}".format(float(max_forget_summary[name]['avg_norm'])* 1000),
                "{:.1f}".format(float(plasticity_summary[name]['avg_norm'])* 1000),
                "{:.1f}".format(float(generalization_summary[name]['avg_norm'])* 1000),
            ])

with open(output_fodler+"/"+str(K)+"_overall_order_summary.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "AvgNorm",  "AvgFinalForget", "AvgMaxForget", "AvgPlasticity", "AvgGeneralization"])
    print(' =========== overall order summary ===========')
    print('AvgRank\tAvgNorm\tAvgAvgAvgForget\tAvgAvgForgetRank\tAvgAvgMaxForget\tAvgMaxForgetRank\tName')
    names = set(all_summary[0].keys())
    for i in range(1,len(all_summary)):
        if i in skip_order:
            continue
        names = names.union(set(all_summary[i].keys()))
        # print('-- ',names)
    for_rank = {}
    for_norm_obj = {}
    for name in names:
        for_rank[name] = [float(all_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        for_norm_obj[name] = [float(all_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
    names = sorted(for_norm_obj.keys(), key=lambda x: sum(for_norm_obj[x])/len(for_norm_obj[x]))

    for name in names:
        avgrank = [float(all_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        avgnorm = [float(all_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
        avg_avg_forgetting = [float(all_avg_forget_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
        rank_avg_forgetting = [float(all_avg_forget_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        avg_max_forgetting = [float(all_max_forget_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
        rank_max_forgetting = [float(all_max_forget_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        avg_plasticity = [float(all_plasticity_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
        rank_plasticity = [float(all_plasticity_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        avg_generalization = [float(all_generalization_summary[i][name]['avg_norm']) for i in all_summary.keys() if name in all_summary[i].keys()]
        rank_generalization = [float(all_generalization_summary[i][name]['avg_rank']) for i in all_summary.keys() if name in all_summary[i].keys()]
        
        writer.writerow([
            name,
            "{:.1f} ({:.1f})".format(sum(avgnorm)/len(avgnorm)* 1000, np.std(avgnorm, ddof=0)*1000),
            "{:.1f} ({:.1f})".format(sum(avg_avg_forgetting)/len(avg_avg_forgetting)* 1000, np.std(avg_avg_forgetting, ddof=0)*1000),
            "{:.1f} ({:.1f})".format(sum(avg_max_forgetting)/len(avg_max_forgetting)* 1000, np.std(avg_max_forgetting, ddof=0)*1000),
            "{:.1f} ({:.1f})".format(sum(avg_plasticity)/len(avg_plasticity)* 1000, np.std(avg_plasticity, ddof=0)*1000),
            "{:.1f} ({:.1f})".format(sum(avg_generalization)/len(avg_generalization)* 1000, np.std(avg_generalization, ddof=0)*1000),
        ])
        print(f"{sum(avgrank)/len(avgrank):.1f}\t{sum(avgnorm)/len(avgnorm):.4f}\t{sum(avg_avg_forgetting)/len(avg_avg_forgetting):.4f}\t\t\t{sum(rank_avg_forgetting)/len(rank_avg_forgetting):.4f}\t\t{sum(avg_max_forgetting)/len(avg_max_forgetting):.4f}\t\t{sum(rank_max_forgetting)/len(rank_max_forgetting):.4f}\t\t\t{name}\t{len(avgrank)}")