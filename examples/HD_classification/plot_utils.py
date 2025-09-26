from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def average_metric_per_round(metric_list: List[Union[List[float], np.ndarray]]) -> List[float]:
    """
    Computes average value per round across a list of metrics.
    
    Args:
        metric_list (List[Union[List[float], np.ndarray]]): A list where each item is a list 
        or numpy array of floats representing metrics.

    Returns:
        List[float]: A list of float averages per round.
    """

    return [np.mean(m) if isinstance(m, (list, np.ndarray)) else m for m in metric_list]

def transpose(metric_list: List[List[float]]) -> List[List[float]]:
    """
    Transposes a 2D list (row <-> column).

    Args:
        metric_list (List[List[float]]): A 2D list of floats.

    Returns:
        List[List[float]]: A transposed 2D list.
    """

    return list(map(list, zip(*metric_list)))

def save_figure(fig_name: str, agent_nr: str, subfolder: str = "avg") -> None:
    """
    Saves the current Matplotlib figure to a categorized folder based on metric type.

    Args:
        fig_name (str): Base name of the figure (without extension).

        agent_nr (str): Identifier for the agent variant.
        
        subfolder (str): Decides which subfolder to save figures in ("binary", "avg", or "per_class").
    """
    DPI = 300

    path = f"examples/HD_classification/figures/{agent_nr}/{subfolder}/{fig_name}.png"
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.show()
    #plt.close() # Uncomment to prevent plot windows from opening

def plot_accuracy(rounds: List[int], global_accs: List[float], local_rounds: List[int] = [], local_accs: List[float] = [], agent_nr: str ='a1', subfolder: str = "avg") -> None:
    """
    Plots accuracy over rounds for global and optionally local models.
    
    Args:
        rounds (List[int]): Global training round numbers.
        
        global_accs (List[float]): Global model accuracy per round.

        local_rounds (List[int]): Local round numbers. Defaults to [].

        local_accs (List[float]): Local model accuracy per round. Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'.    

        subfolder (str): Decides which subfolder to save accuracy plot in ("binary", "avg", or "per_class").
    """

    FIG_SIZE_TUPLE = (9,5)
    FONT_SIZE_T = 14
    FONT_SIZE_X = FONT_SIZE_Y = 12
    Z_ORDER = 5 

    plt.figure(figsize = FIG_SIZE_TUPLE)
    plt.plot(rounds, global_accs, label="Global Model Accuracy", linestyle='-', marker='o')
    if agent_nr != 'CL':
        plt.scatter(0, global_accs[0], color='red', zorder=Z_ORDER, label='Initial Global (Unaggregated)')
        plt.plot(local_rounds, local_accs, label="Local Model Accuracy", linestyle='--', marker='x')
        plt.legend()

    plt.xlabel("Rounds", fontsize=FONT_SIZE_X)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE_Y)
    plt.title(f"Accuracy Over Communication Rounds for {agent_nr}", fontsize=FONT_SIZE_T)
    plt.grid(True)
    plt.tight_layout()
    save_figure("accuracy_plot", agent_nr, subfolder)


def plot_f1_score(rounds: List[int], global_f1s: List[float], local_rounds: List[int] = [], local_f1s: List[float] = [], agent_nr: str ='a1', subfolder: str = "avg") -> None:
    """
    Plots F1 scores over rounds for global and optionally local models.
    
    Args:
        rounds (List[int]): Global training round numbers.

        global_f1s (List[float]): Global model F1 scores per round.

        local_rounds (List[int]): Local round numbers. Defaults to [].

        local_f1s (List[float]): Local model F1 scores per round. Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'. 
        
        subfolder (str): Decides which subfolder to save f1 score 
        plot in ("binary", "avg", or "per_class").    
    """
    
    FIG_SIZE_TUPLE = (9,5)
    FONT_SIZE_T = 14
    FONT_SIZE_X = FONT_SIZE_Y = 12

    plt.figure(figsize=FIG_SIZE_TUPLE)
    plt.plot(rounds, global_f1s, label="Global", marker='o')
    if agent_nr != 'CL':
        plt.plot(local_rounds, local_f1s, label="Local", marker='x')
    plt.title(f"Average F1 Score Over Communication Rounds for {agent_nr}", fontsize=FONT_SIZE_T)
    plt.xlabel("Rounds", fontsize=FONT_SIZE_X)
    plt.ylabel("F1 Score",fontsize=FONT_SIZE_Y)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("f1_score", agent_nr, subfolder)

def plot_auroc(rounds: List[int], global_aurocs: List[float], local_rounds: List[int] = [], local_aurocs: List[float] = [], agent_nr: str ='a1', subfolder: str = "avg") -> None:
    
    """
    Plots AUROC over rounds for global and optionally local models.
    
    Args:
        rounds (List[int]): Global training round numbers.

        global_aurocs (List[float]): Global model AUROC values per round.

        local_rounds (List[int]): Local round numbers. Defaults to [].

        local_aurocs (List[float]): Local model AUROC values per round. Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'. 

        subfolder (str): Decides which subfolder to save auroc plot in ("binary", "avg", or "per_class"). 
    """
    
    FIG_SIZE_TUPLE = (9,5)
    FONT_SIZE_T = 14
    FONT_SIZE_X = FONT_SIZE_Y = 12

    plt.figure(figsize=FIG_SIZE_TUPLE)
    plt.plot(rounds, global_aurocs, label="Global", marker='o')
    if agent_nr != 'CL':
        plt.plot(local_rounds, local_aurocs, label="Local", marker='x')
    plt.title(f"Average AUROC Score Over Communication Rounds for {agent_nr}", fontsize=FONT_SIZE_T)
    plt.xlabel("Rounds", fontsize=FONT_SIZE_X)
    plt.ylabel("AUROC", fontsize=FONT_SIZE_Y)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("auroc", agent_nr, subfolder)



def plot_f1_score_per_class(rounds: List[int], global_f1_class: List[List[float]], local_rounds: List[int] = [], local_f1_class: List[List[float]] = [], agent_nr: str = 'a1', subfolder: str = "per_class") -> None:
    """
    Plots per-class F1 score over rounds for global and optionally local models.

    Args:
        rounds (List[int]): Global training round numbers.

        global_f1_class (List[List[float]]): Per-class F1 scores for global model.

        local_rounds (List[int]): Local round numbers. Defaults to [].

        local_f1_class (List[List[float]]): Per-class F1 scores for local model. Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'.

        subfolder (str): Decides which subfolder to save f1 plot in ("binary", "avg", or "per_class"). 
    """
    FIG_SIZE_TUPLE = (10,6)
    FONT_SIZE_T = 14
    FONT_SIZE_X = FONT_SIZE_Y = 12

    plt.figure(figsize=FIG_SIZE_TUPLE)
    for cls_idx, values in enumerate(global_f1_class):
        # Plot global F1 scores for each class
        plt.plot(rounds, values, label=f"Global Class {cls_idx}", linestyle='--', marker='o')
    if agent_nr != 'CL':
        for cls_idx, values in enumerate(local_f1_class):
            # Plot local F1 scores for each class (only if FL)
            plt.plot(local_rounds, values, label=f"Local Class {cls_idx}", linestyle='-', marker='x')
    plt.title(f"Per-Class F1 Score Over Communication Rounds for {agent_nr}", fontsize=FONT_SIZE_T)
    plt.xlabel("Rounds", fontsize=FONT_SIZE_X)
    plt.ylabel("F1 Score", fontsize=FONT_SIZE_Y)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("f1_score_per_class", agent_nr, subfolder)

def plot_auroc_per_class(rounds:List[int], global_aurocs: List[float], local_rounds: List[int] = [], local_aurocs: List[float] = [], agent_nr: str = 'a1', subfolder: str = "per_class") -> None:
    """
    Plots per-class AUROC over rounds for global and optionally local models.
    
    Args:
        rounds (List[int]): Global training round numbers.

        global_aurocs (List[float]): Per-class AUROC scores for the global model.

        local_rounds (List[int]): Local round numbers. Defaults to [].

        local_aurocs (List[float]): Per-class AUROC scores for the local model. 
        Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'.

        subfolder (str): Decides which subfolder to save auroc plot in ("binary", "avg", or "per_class"). 
    """
    FIG_SIZE_TUPLE = (10,6)
    FONT_SIZE_T = 14
    FONT_SIZE_X = FONT_SIZE_Y = 12

    plt.figure(figsize=FIG_SIZE_TUPLE)
    plt.plot(rounds, global_aurocs, label="Global", linestyle='--', marker='o')
    if agent_nr != 'CL':
        plt.plot(local_rounds, local_aurocs, label="Local", linestyle='-', marker='x')
    plt.title(f"AUROC Over Communication Rounds for {agent_nr}", fontsize=FONT_SIZE_T)
    plt.xlabel("Rounds", fontsize=FONT_SIZE_X)
    plt.ylabel("AUROC", fontsize=FONT_SIZE_Y)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure("auroc_per_class", agent_nr, subfolder)

def plot_performance_centralized(global_accs: List[float], global_f1s: Union[List[float], List[List[float]]], global_aurocs:List[float], agent_nr: str ='a1',
    is_binary: bool = True, plot_version: str = "avg") -> None: 
    
    """
    Generates centralized performance plots for accuracy, F1, and AUROC.
    
    Args:
        global_accs (List[float]): Global model accuracy per round.

        global_f1s (Union[List[float], List[List[float]]]): Global model F1 scores (average or per-class).

        global_aurocs (List[float]): Global model AUROC values per round.

        agent_nr (str): Agent identifier. Defaults to 'a1'.

        is_binary (bool): Whether the classification task is binary. 
        Defaults to True.

        plot_version (str): 'avg' for average metrics or 'per_class' for per-class metrics. 
        Defaults to 'avg'.
    """

    # Define global and local round ranges based on metric lengths
    rounds = list(range(len(global_accs)))
    

    if is_binary:
        # Plot single-value metrics
        plot_accuracy(rounds = rounds, global_accs = global_accs, agent_nr = agent_nr, subfolder = "binary")
        plot_f1_score(rounds = rounds, global_f1s = global_f1s, agent_nr = agent_nr, subfolder = "binary")
        plot_auroc(rounds = rounds, global_aurocs = global_aurocs, agent_nr = agent_nr,subfolder = "binary")

    else:
        if plot_version == "avg":
            # Plot average F1 across all classes
            avg_global_f1s = average_metric_per_round(global_f1s)
            avg_global_aurocs = global_aurocs 

            plot_accuracy(rounds = rounds, global_accs = global_accs, agent_nr = agent_nr, subfolder = plot_version)
            plot_f1_score(rounds = rounds, global_f1s = avg_global_f1s, agent_nr = agent_nr, subfolder = plot_version)
            plot_auroc(rounds = rounds, global_aurocs = avg_global_aurocs, agent_nr = agent_nr, subfolder = plot_version)
            

        elif plot_version == "per_class":
            # Plot per-class F1 using transposed data
            # Reorganize class-wise metrics: [round][class] → [class][round]
            global_f1_class = transpose(global_f1s)

            plot_accuracy(rounds = rounds, global_accs = global_accs, agent_nr = agent_nr, subfolder = plot_version)
            plot_f1_score_per_class(rounds = rounds, global_f1_class = global_f1_class, agent_nr = agent_nr, subfolder = plot_version) 
            plot_auroc_per_class(rounds = rounds, global_aurocs = global_aurocs, agent_nr = agent_nr, subfolder = plot_version) 

    
    

def plot_performance_federated(global_accs: List[float], global_f1s: Union[List[float], List[List[float]]], global_aurocs: List[float], 
    local_accs: List[float] = [], local_f1s: List[float] = [], local_aurocs: List[float] = [],agent_nr: str ='a1',
    is_binary: bool = True, plot_version: str = "avg") -> None:
    """
    Generates federated performance plots for accuracy, F1, and AUROC 
    for both global and local models.
    
    Args:
        global_accs (List[float]): Global model accuracy per round.

        global_f1s (Union[List[float], List[List[float]]]): Global model F1 scores (average or per-class).

        global_aurocs (List[float]): Global model AUROC values per round.

        local_accs (List[float]): Local model accuracy per round. Defaults to [].

        local_f1s (Union[List[float], List[List[float]]]): Local model F1 scores 
        (average or per-class). Defaults to [].

        local_aurocs (List[float]): Local model AUROC values per round. Defaults to [].

        agent_nr (str): Agent identifier. Defaults to 'a1'.

        is_binary (bool): Whether the classification task is binary. 
        Defaults to True.

        plot_version (str): 'avg' for average metrics or 'per_class' for per-class metrics. 
        Defaults to 'avg'.
    """

    # Define global and local round ranges based on metric lengths
    rounds = list(range(len(global_accs)))
    # Local rounds start at 1 (global round 0 is unaggregated initial model)
    local_rounds = list(range(1, len(local_accs) + 1)) 

    if is_binary:
        # Plot single-value metrics
        
        plot_accuracy(rounds, global_accs, local_rounds, local_accs, agent_nr, subfolder = "binary")
        plot_f1_score(rounds, global_f1s, local_rounds, local_f1s, agent_nr, subfolder = "binary")
        plot_auroc(rounds, global_aurocs, local_rounds, local_aurocs, agent_nr, subfolder = "binary")
            
    else:
        if plot_version == "avg":
            # Plot average F1 and AUROC across all classes
            avg_global_f1s = average_metric_per_round(global_f1s)
            avg_global_aurocs = global_aurocs

            avg_local_f1s = average_metric_per_round(local_f1s)
            avg_local_aurocs = local_aurocs

            plot_accuracy(rounds, global_accs, local_rounds, local_accs, agent_nr, subfolder = plot_version)
            plot_f1_score(rounds, avg_global_f1s, local_rounds, avg_local_f1s, agent_nr, subfolder = plot_version)
            plot_auroc(rounds, avg_global_aurocs, local_rounds, avg_local_aurocs, agent_nr, subfolder = plot_version)
            

        elif plot_version == "per_class":
            # Plot per-class F1 using transposed data
            # Reorganize class-wise metrics: [round][class] → [class][round]

            global_f1_class = transpose(global_f1s)
            local_f1_class = transpose(local_f1s)

            plot_accuracy(rounds, global_accs, local_rounds, local_accs, 
            agent_nr, subfolder = plot_version)

            plot_f1_score_per_class(rounds, global_f1_class,
        local_rounds, local_f1_class, agent_nr, subfolder = plot_version)

            plot_auroc_per_class(rounds, global_aurocs,
        local_rounds, local_aurocs, agent_nr, subfolder = plot_version) 
