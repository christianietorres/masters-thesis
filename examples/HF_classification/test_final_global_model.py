'''
# test_final_global_model.py

from fl_main.lib.util.helpers import load_model_file
from .training_utils import compute_performance, prep_test_data, TrainingMetaData
from .training import DataManger
import numpy as np

def evaluate_final_global_model(agent_nr, model_file_name, rounds=10):
    path = f'./data/agents/{agent_nr}'
    name = model_file_name

    final_global_model, _ = load_model_file(path, name)
    DataManger.dm(agent_nr, th=1000)

    metrics = {key: [] for key in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auroc']}

    for _ in range(rounds):
        acc, sens, spec, prec, f1, auroc, _ = compute_performance(final_global_model, prep_test_data(), False, agent_nr)
        metrics['accuracy'].append(acc)
        metrics['sensitivity'].append(sens)
        metrics['specificity'].append(spec)
        metrics['precision'].append(prec)
        metrics['f1'].append(f1)
        metrics['auroc'].append(auroc)

    for metric_name, values in metrics.items():
        if TrainingMetaData.binary:
            avg = np.mean(values)
        else:
            avg = np.mean([np.mean(v) for v in values])
        print(f"Average {metric_name} for {agent_nr}: {avg:.4f}")

#if __name__ == "__main__":
agents = ['a1', 'a2']  # Add more agents if needed
for agent in agents:
    evaluate_final_global_model(agent, model_file_name='final_global_modelgms.binaryfile')

'''

from fl_main.lib.util.helpers  import load_model_file
from .classification_engine import compute_performance, prep_test_data, TrainingMetaData
from .training import DataManger
import statistics
import numpy as np

agent1 = 'a1'
agent2 = 'a2'
list_agents = [agent1, agent2]

for agent_nr in list_agents:
    path = f'./data/agents/{agent_nr}'
    name = 'final_global_modelgms.binaryfile'
    final_global_model, final_global_performance = load_model_file(path, name)

    dm = DataManger.dm(agent_nr, th=1000)
    global_model_accuracies = []
    global_model_sensitivities = []
    global_model_specificities = []
    global_precisions = []
    global_f1scores = []
    global_aurocs = []
    for i in range(10):
        global_acc, global_sens, global_spec, global_prec, global_f1, global_auroc, global_cm = compute_performance(final_global_model, prep_test_data(), False, agent_nr)
        
        global_model_accuracies.append(global_acc)
        global_model_sensitivities.append(global_sens)
        global_model_specificities.append(global_spec)
        global_precisions.append(global_prec)
        global_f1scores.append(global_f1)
        global_aurocs.append(global_auroc)

    perf_metric_list = [global_model_accuracies, global_model_sensitivities, global_model_specificities, global_precisions, global_f1scores, global_aurocs]
    
    for idx, per_metric in enumerate(perf_metric_list):
        if idx == 0: 
            #print(per_metric)
            avg = float(np.mean(per_metric))

            #avg = statistics.mean(per_metric)
            #print(avg)
        else:
            #print(per_metric)
            if TrainingMetaData.binary:
                #avg = statistics.mean(per_metric)
                print(per_metric)
                avg = float(np.mean(per_metric))
            else:
                avg_metric_list = []
                for m in per_metric:
                    #avg = statistics.mean(m)
                    avg = float(np.mean(m))
                    avg_metric_list.append(avg)
                #print(avg)
        if idx == 0:
            avg_global_acc = avg
            print(f'average global model accuracy is {avg_global_acc}')

        elif idx == 1:
            if TrainingMetaData.binary:
                avg_global_sens = avg
                print(f'average global model sensitivity is {avg_global_sens}')
            else:
                avg_here = statistics.mean(avg_metric_list)
                #avg_here = float(np.mean(avg_metric_list))
                avg_global_sens = avg_here
                print(f'average global model sensitivity is {avg_global_sens}')

        elif idx == 2:
            if TrainingMetaData.binary:
                avg_global_spec = avg
                print(f'average global model specificity is {avg_global_spec}')
            else:
                avg_here = statistics.mean(avg_metric_list)
                #avg_here = float(np.mean(avg_metric_list))
                avg_global_spec = avg_here
                print(f'average global model specificity is {avg_global_spec}')

        elif idx == 3:
            if TrainingMetaData.binary:
                avg_global_prec = avg
                print(f'average global model precision is {avg_global_prec}')
            else:
                avg_here = statistics.mean(avg_metric_list)
                #avg_here = float(np.mean(avg_metric_list))
                avg_global_prec = avg_here
                print(f'average global model precision is {avg_global_prec}')

        elif idx == 4:
            if TrainingMetaData.binary:
                avg_global_f1 = avg
                print(f'average global model f1 is {avg_global_f1}')
            else:
                avg_here = statistics.mean(avg_metric_list)
                #avg_here = float(np.mean(avg_metric_list))
                avg_global_f1 = avg_here
                print(f'average global model f1 is {avg_global_f1}')

        elif idx == 5:
            avg_global_auroc = avg
            print(f'average global model auroc is {avg_global_auroc}')
    print('compute_performance for agent i done')
    