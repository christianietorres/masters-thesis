import logging
logging.basicConfig(level=logging.INFO)

from typing import List, Tuple, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .MLP import MLPNet
from .conversion import Converter
from .training import DataManger, execute_th_training

from sklearn.metrics import f1_score, roc_auc_score
import warnings
import shap


class TrainingMetaData:
    """
    A simple configuration container for training parameters 
    used across the federated learning pipeline.

    Attributes:
        NUM_TRAINING_DATA (int): Number of samples 
        to be used for local training.
        Used to control the cutoff threshold and sample weighting.
        
        BINARY (bool): Whether the classification task is binary.

        NUM_CLASSES (int): number of clae of the data 

        LEARNING_RATE (int): the learning rate for the model training.

        MOMENTUM (int): the momentum for the model training.

        DIVIDOR (int): used to calculate the threshold for training 
        due to the batch size
    """

    NUM_TRAINING_DATA = 8000
    BINARY = True
    
    NUM_CLASSES = 5 
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    DIVIDOR = 4


def init_models() -> Dict[str,np.array]:
    """
    Return the templates of models (in a dict) to tell the structure
    the models need not to be trained.

    Returns:
        Dict[str,np.array]
    """
    net = MLPNet()

    return Converter.cvtr().convert_nn_to_dict_nparray(net)


def training(models: Dict[str,np.array], agent_nr: str, init_flag: bool = False) -> Dict[str,np.array]:
    """
    Return the trained models
    Note that each models should be decomposed into numpy arrays
    Logic should be in the form: models -- training --> new local models

    Args:

        models (Dict[str,np.array])
        init_flag (bool):  Flag to check if the model is at the init step. 
        It's False if it's an actual training step

    Returns:
        Dict[str,np.array]: the trained models.

    """

    if init_flag:
        # Prepare the training data 
        # Determine training threshold based on total data and batch size divisor

        DataManger.dm(agent_nr, int(TrainingMetaData.NUM_TRAINING_DATA 
        / TrainingMetaData.DIVIDOR))
        return init_models()


    logging.info(f'--- Client Model is now training ---')

    # Create a neural network (NN) based on global (cluster) models
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    # Define loss function
    if TrainingMetaData.BINARY  == True:
        criterion = nn.BCEWithLogitsLoss()
    elif TrainingMetaData.BINARY  == False:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
    lr=TrainingMetaData.LEARNING_RATE, momentum=TrainingMetaData.MOMENTUM)

    # Perform model training using data manager, loss function, and optimizer
    trained_net = execute_th_training(DataManger.dm(agent_nr), 
    net, criterion, optimizer)
    models = Converter.cvtr().convert_nn_to_dict_nparray(trained_net)

    return models


def compute_performance(models: Dict[str,np.array], testdata, 
    is_local: bool, agent_nr: str) -> float:

    """
    Given a set of models and test dataset, 
    compute the performance of the models 
    by uing the metric: accurcay, f1-core and auroc


    Args:
        models (Dict[str,np.array]):
        testdata ():
        is_local (bool): flag to check wheter the model i local or not.
        agent_nr (str): check the identification of the agent/client. 

    Returns:
        accuracy (float):
        f1_scores (Union[float, List[float]]):
        auroc (float):
    """

    # Convert np arrays to a neural network (NN)
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)
    
    
    CORRECT = 0
    TOTAL = 0

    all_labels = []
    all_preds = []
    probas_list = [] 

    if TrainingMetaData.BINARY == True:
        TP = FP = TN = FN = 0
        with torch.no_grad():

            for data in DataManger.dm(agent_nr).testloader:
                samples, labels = data
                outputs = net(samples)
                probs = torch.sigmoid(outputs)
                probas_list.extend(probs.detach().cpu().numpy().flatten())
                predicted = (probs >= 0.5).long().squeeze()
                
                TOTAL += labels.size(0)
                CORRECT += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy()) 

                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
        

                    if true_label == 1:
                        if pred_label == 1:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if pred_label == 1:
                            FP += 1
                        else:
                            TN += 1  

        try:
            logging.info(f"all_label i {set(all_labels)}")

            auroc = roc_auc_score(all_labels, probas_list)
        except ValueError as e:
            warnings.warn(f"AUROC computation failed due to ValueError: {e}")
            auroc = np.nan
                    
    else: 
        TP = [0] * TrainingMetaData.NUM_CLASSES 
        FP = [0] * TrainingMetaData.NUM_CLASSES 
        TN = [0] * TrainingMetaData.NUM_CLASSES 
        FN = [0] * TrainingMetaData.NUM_CLASSES 

        with torch.no_grad():
            for data in DataManger.dm(agent_nr).testloader:
                samples, labels = data
                outputs = net(samples)

                probs = torch.softmax(outputs, dim=1)
                probas_list.append(probs.detach().cpu().numpy())

                _, predicted = torch.max(outputs.data, 1)
                TOTAL += labels.size(0)
                CORRECT += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    for cls in range(TrainingMetaData.NUM_CLASSES ):
                        if true_label == cls and pred_label == cls:
                            TP[cls] += 1
                        elif true_label != cls and pred_label == cls:
                            FP[cls] += 1
                        elif true_label == cls and pred_label != cls:
                            FN[cls] += 1
                        elif true_label != cls and pred_label != cls:
                            TN[cls] += 1
        
        try:
            all_probs_np = np.vstack(probas_list)
            auroc = roc_auc_score(np.array(all_labels), all_probs_np, multi_class='ovr')
        except Exception as e:
            warnings.warn(f"AUROC computation failed due to ValueError: {e}")
            auroc = np.nan

    accuracy = float(CORRECT) / TOTAL  


    if TrainingMetaData.BINARY:
        f1_scores = f1_score(all_labels, all_preds, zero_division=0)
    else:
        f1_scores = f1_score(all_labels, all_preds, average=None, zero_division=0)

    mt = 'local'
    if not is_local:
        mt = 'Global'

    if TrainingMetaData.BINARY == True:
        logging.info(f'\n')
        logging.info(f'Performance metrics of the {mt} model with the {DataManger.dm(agent_nr).test_s} test samples:')
        logging.info(f'Accuracy: {accuracy:.2%}')
        logging.info(f"F1 Score: {np.mean(f1_scores):.2%}")
        logging.info(f"Auroc: {auroc:.2%}")

    else:
        logging.info(f'\n')
        logging.info(f'Performance metrics of the {mt} model with the {DataManger.dm(agent_nr).test_s} test samples:')
        logging.info(f'Accuracy: {accuracy:.2%}')
        for i in range(TrainingMetaData.NUM_CLASSES):
            logging.info(f"Class {i}: F1 Score: {f1_scores[i]:.2%}")
        logging.info(f"Auroc: {auroc:.2%}")

    return accuracy, f1_scores, auroc


def judge_termination(training_count: int = 0, gm_arrival_count: int = 0, 
    max_training_rounds: int = 10) -> bool:
    """
    Decides if the trainin process needs to finish based on 
    if it has reached the maximum rounds of training, if so it 
    exits from FL platform

    Args:
        training_count (int): The number of training done.
        gm_arrival_count (int): The number of times it received global models
        max_training_rounds (int): 

    Returns:
        bool: True if the training loop should continue; False if it has to stop

    """

    return training_count < max_training_rounds


def prep_test_data(agent_nr: str  ='a1') -> torch.utils.data.DataLoader:
    """
    Return the test loader from the federated agent's data.

    Args:
        agent_nr (str): Identifier for the agent ('a1', 'a2', or 'CL').
    Returns:
        torch.utils.data.DataLoader: The test data loader.
    """

    dm = DataManger.dm(agent_nr)
    return dm.testloader


def average_metric_per_round(metric_list: List[Union[List[float], float]]) -> List[float]:
    """
    Args:
        metric_list (List[Union[List[float], float]]): A list of metric values or lists per round.

    Returns:
        List[float]: Averaged metrics per round.
    """
    return [np.mean(m) if isinstance(m, list) else m for m in metric_list]


def transpose(metric_list: List[List[Any]]) -> List[List[Any]]:
    """
    Transposes a 2D list (rows become columns and vice versa).

    Args:
        metric_list (List[List[Any]]): 2D list of metrics.

    Returns:
        List[List[Any]]: Transposed metric list.
    """

    return list(map(list, zip(*metric_list)))


def store_performance_data(acc: float, f1: Union[float, List[float]], auroc: float) -> Dict:
    """
    Stores evaluation metrics in a dictionary.

    Args:
        acc (float): Accuracy score.
        f1 (Union[float, List[float]]): F1 score(s).
        auroc (float): Area under ROC curve.

    Returns:
        Dict[str, Union[float, List[float]]]: Dictionary with the metrics.
    """

    performance_data_dict = dict()
    performance_data_dict['accuracy'] = acc
    performance_data_dict['f1'] = f1
    performance_data_dict['auroc'] = auroc

    return performance_data_dict


def dicts_have_same_structure(d1: Dict[str,np.array], d2: Dict[str,np.array]) -> bool:
    """
    Checks whether two model dictionaries have identical structure and keys.

    Args:
        d1 (Dict[str,np.array]): first model dictionary.

        d2 (Dict[str,np.array]): second model dictionary.

    Returns:
        bool: True if model structures matches, False otherwise.

    """

    if not isinstance(d1, dict) or not isinstance(d2, dict):
        logging.info(f'd1 is {type(d1)}') 
        logging.info(f'd2 is {type(d2)}')
        return False

    if set(d1.keys()) != set(d2.keys()):
        logging.info(f'd1-keys: {d1.keys()}')
        logging.info(f'd2-keys: {d2.keys()}')
        return False   

    for key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            if not dicts_have_same_structure(d1[key], d2[key]):
                logging.info(f'd1-key: {d1[key]}')
                logging.info(f'd2-key: {d2[key]}')
                return False
        elif isinstance(d1[key], dict) != isinstance(d2[key], dict):
            return False

    return True


def compute_total_transfer(model_dict: Dict[str,np.array], num_clients: int, num_rounds: int, 
    bi_directional: bool = True) -> int:
    """
    Computes total communication cost in bytes for model transfers.

    Args:
        model_dict (Dict[str, np.ndarray]): Dictionary of model weights.

        num_clients (int): Number of clients.

        num_rounds (int): Number of federated learning rounds.

        bi_directional (bool): Whether upload and download are both counted.

    Returns:
        int: Total bytes transferred.
    """

    model_size_bytes = sum(arr.nbytes for arr in model_dict.values())
    TOTAL = model_size_bytes * num_clients * num_rounds
    if bi_directional:
        TOTAL *= 2
    return TOTAL


def explain_model_with_shap(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
    num_samples: int = 100) -> None:
    """
    Uses SHAP to explain model predictions on a sample batch.

    Args:
        model (torch.nn.Module): Trained PyTorch model.

        data_loader (torch.utils.data.DataLoader): DataLoader 
        for evaluation samples.

        num_samples (int): Number of samples to explain.

    Returns:
        None

    """

    model.eval()
    background_data = []
    test_data = []

    # Collect sample batches
    for inputs, _ in data_loader:
        background_data.append(inputs)
        test_data.append(inputs)
        if len(test_data) * inputs.size(0) >= num_samples:
            break

    # background set for SHAP
    background = torch.cat(background_data[:5])  
    test_samples = torch.cat(test_data)[:num_samples]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    feature_names = [f'f{i}' for i in range(test_samples.shape[1])]
    shap.summary_plot(shap_values, test_samples.detach().numpy(), 
    feature_names=feature_names)

    print_top_n_features(shap_values, test_samples.numpy(), feature_names)
    print_feature_contributions_for_sample(shap_values, sample_idx=0, 
    test_samples=test_samples, feature_names=feature_names)


def print_top_n_features(shap_values: Union[np.ndarray, List[np.ndarray]], 
    test_samples: np.ndarray, feature_names: List[str], N: int = 5) -> None:
    """
    Logs the top-N most important features by SHAP values.

    Args:
        shap_values (Union[np.ndarray, List[np.ndarray]]): SHAP values.
        test_samples (np.ndarray): Input samples.
        feature_names (List[str]): List of feature names.
        N (int): Number of top features to log.
    """    

    # For multi-class, shap_values is a list of arrays — one per class
    if isinstance(shap_values, list):
        shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_array = np.abs(shap_values)

    # Compute mean SHAP value per feature
    mean_shap = np.mean(shap_array, axis=0) 

    # Sort and print top-N
    top_indices = np.argsort(mean_shap)[::-1][:N]
    print(f"\n🔍 Top-{N} important features (by mean SHAP value):")
    for idx in top_indices:
        logging.info(f"{feature_names[idx]}: {mean_shap[idx]:.4f}")


def print_feature_contributions_for_sample(shap_values: Union[np.ndarray, List[np.ndarray]], 
    sample_idx: int, test_samples: np.ndarray, feature_names: List[str]) -> None:
    """
    Print SHAP values per feature for a single prediction of a sample.

    Args:
        shap_values (Union[np.ndarray, List[np.ndarray]]): SHAP values.
        sample_idx (int): Index of the sample.
        test_samples (np.ndarray): All input samples.
        feature_names (List[str]): Names of input features.
    """

    print(f"\n Feature contributions for sample #{sample_idx}:")

    sample_input = test_samples[sample_idx].numpy()
    
    if isinstance(shap_values, list):  # multi-class
        for class_idx, class_shap in enumerate(shap_values):
            print(f"\nClass {class_idx} contribution:")
            for f_idx, f_name in enumerate(feature_names):
                print(f"  {f_name}: value={sample_input[f_idx]:.3f}, shap={class_shap[sample_idx][f_idx]:.4f}")
    else:  # binary or single output
        for f_idx, f_name in enumerate(feature_names):
            print(f"{f_name}: value={sample_input[f_idx]:.3f}, shap={shap_values[sample_idx][f_idx]:.4f}")