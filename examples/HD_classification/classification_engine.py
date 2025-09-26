import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from typing import List, Tuple, Union, Dict, Any, Optional


from .training_utils import TrainingMetaData, training, judge_termination, compute_performance, store_performance_data, prep_test_data, compute_total_transfer, explain_model_with_shap
from fl_main.lib.util.helpers import save_model_file, set_config_file, read_config
from .plot_utils import plot_performance_federated, plot_performance_centralized
from .plot_utils import save_figure
from fl_main.agent.client import Client

from .conversion import Converter
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim

class CLRunner:
    """
    A class that runs Centralized Learning for Heart Failure classification.
    This class manages model training, evaluation, 
    performance tracking, and explainability in a centralized setting.
    """
    
    def __init__(self, max_rounds: int =100) -> None:
        """
        Initializes the runner with configurable training parameters.

        Args:
            max_rounds (int): Maximum number of training iterations.
         """
        
        logging.info('--- This is a demo of HD Classification with Centralized Learning ---')
        self.agent_nr = 'CL'
        self.models = None
        self.accuracies = []
        self.f1scores = []
        self.aurocs = []

        self.RANDOM_STATE = 42
        
        self.TRAINING_COUNT = 0
        self.MAX_TRAINING_ROUNDS = max_rounds

        self.TARGET_ACCURACY = 0.9
        self.TARGET_F1 = 0.9
        self.TARGET_AUROC = 0.9

        self.NUM_CLIENTS = 2

    def set_seed(self, seed: int =42) -> None:
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): Seed value to use.
        """ 

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self) -> None:
        """
        Executes the full training and evaluation lifecycle: Initializes model(s), 
        Performs training loop, 
        Evaluates and logs performance and 
        Saves models and visualizations.
        """    

        self.set_seed(self.RANDOM_STATE)
        self.models = training(dict(), self.agent_nr, init_flag=True)

        while judge_termination(self.TRAINING_COUNT, max_training_rounds=self.MAX_TRAINING_ROUNDS):
            # Training of model
            self.models = training(self.models, self.agent_nr)
            self.TRAINING_COUNT += 1
            logging.info(f'--- model training {self.TRAINING_COUNT} of {self.MAX_TRAINING_ROUNDS} Done ---')

            # Model evaluation (id, accuracy)
            acc, f1, auroc = compute_performance(self.models, prep_test_data(), False, self.agent_nr)
            logging.info(f'--- model evaluation Done ---')

            # Store evaluation metrics for each round
            self.accuracies.append(acc)
            self.f1scores.append(f1)
            self.aurocs.append(auroc)

        self.compute_final_metrics()
        self.log_performance_summary()
        self.explain_final_model_with_shap()
        self.save_and_plot()

    def compute_final_metrics(self) -> Tuple[List[float], List[float], List[Optional[int]]]:
        """
        Computes best and final performance metrics (accuracy, F1, AUROC),
        and identifies the training round at which performance thresholds are first met.

        Returns:
            Tuple[List[float], List[float], List[Optional[int]]]: Best metrics, Final metrics,
            Round numbers at which target metrics were reached.
        """
        
        self.best_acc = max(self.accuracies)
        if TrainingMetaData.BINARY:
            self.best_f1 = max(self.f1scores)
        else:
            self.f1scores_avg = [sum(arr)/len(arr) for arr in self.f1scores if len(arr) > 0]
            self.best_f1 = max(self.f1scores_avg)
        self.best_auroc = max(self.aurocs)

        self.best_list = [self.best_acc, self.best_f1, self.best_auroc]

        self.final_acc = self.accuracies[-1]
        self.final_f1 = self.f1scores[-1] if TrainingMetaData.BINARY else self.f1scores_avg[-1]
        self.final_auroc = self.aurocs[-1]

        self.final_list = [self.final_acc, self.final_f1, self.final_auroc]

        self.round_w_acc_higher_than_target = [i for i, acc in enumerate(self.accuracies) 
        if acc >= self.TARGET_ACCURACY]
        f1_source = self.f1scores if TrainingMetaData.BINARY else self.f1scores_avg
        self.round_w_f1_higher_than_target = [i for i, f1 in enumerate(f1_source) 
        if f1 >= self.TARGET_F1]
        self.round_w_auroc_higher_than_target= [i for i, auroc in enumerate(self.aurocs) 
        if auroc >= self.TARGET_AUROC]

        self.round_to_target_acc = self.round_w_acc_higher_than_target[0] if len(self.round_w_acc_higher_than_target) >= 1 else None
        self.round_to_target_f1 = self.round_w_f1_higher_than_target[0] if len(self.round_w_f1_higher_than_target) >=  1 else None
        self.round_to_target_auroc = self.round_w_auroc_higher_than_target[0] if len(self.round_w_auroc_higher_than_target) >= 1 else None
        
        self.round_to_target_list = [self.round_to_target_acc, self.round_to_target_f1, self.round_to_target_auroc]

        return self.best_list, self.final_list, self.round_to_target_list 

    def log_performance_summary(self) -> None:
        """
        Logs the best and final model performance,
        as well as the number of training rounds n
        eeded to meet target thresholds.
        """

        metric_list = ["acc", "f1", "auroc"]
        target_list = [self.TARGET_ACCURACY, self.TARGET_F1, self.TARGET_AUROC]

        logging.info(' ---- Best performance metrics')
        for metric, best in zip(metric_list, self.best_list):
            logging.info(f'{metric}: {best}')

        logging.info(' ---- Final performance metrics')
        for metric, final in zip(metric_list, self.final_list):
            logging.info(f'{metric}: {final}')

        logging.info(' ---- Rounds to target performance threshold')
        for metric, round in zip(metric_list, self.round_to_target_list):
            logging.info(f'{metric}: {round}')

        logging.info(' ---- Total communication cost to reach target metrics ----')
        for target, metric, round in zip(target_list, metric_list, self.round_to_target_list):
            if round is not None:
                comm = compute_total_transfer(self.models, self.NUM_CLIENTS, round)
                logging.info(f"Total communication cost to reach {target*100}% {metric}: {comm / (1024**2):.3f} MB")


    def explain_final_model_with_shap(self) -> None:
        """
        Generates SHAP explanations on the final model 
        using a held-out test dataset.
        Plots global feature importance and logs feature-level contributions.
        """

        logging.info("\n[SHAP] Running explainability on final global model...")
        final_model = Converter.cvtr().convert_dict_nparray_to_nn(self.models)
        test_loader = prep_test_data()
        explain_model_with_shap(final_model, test_loader)
    
    def save_and_plot(self) -> None:
        """
        Saves the final trained model to disk and generates performance plots.
        """
        
        perf_dict = store_performance_data(self.accuracies[-1], self.f1scores[-1], self.aurocs[-1])
        config_file = set_config_file("agent")
        config = read_config(config_file)
        save_model_file(self.models, f'{config["model_path"]}/CL', '.binaryfile', 
        perf_dict, final_flag=True, FL_flag=False)

        plot_performance_centralized(self.accuracies, self.f1scores, self.aurocs, 
        self.agent_nr, is_binary=TrainingMetaData.BINARY, plot_version="per_class") 
        logging.info(f'--- Plotting Done ---')


# classification_engine.py
class FLRunner:
    """
        A class that runs Federated Learning for Heart Disease classification.

        This class handles communication with a central server, 
        both global and local model training, model evaluation,
        and explainability of the final global model in a federated learning framework.
    """

    def __init__(self, max_rounds:int = 100) -> None:
        """
        Initializes the runner with configurable training parameters.

        Args:
            max_rounds (int): Maximum number of training iterations.
         """
        self.fl_client = Client()
        self.agent_nr = self.fl_client.agent_name

        self.RANDOM_STATE = 42

        self.TRAINING_COUNT = 0
        self.GM_ARRIVAL_COUNT = 0
        self.MAX_TRAINING_ROUNDS = max_rounds
        self.NUM_CLIENTS = 2

        self.TARGET_ACCURACY = 0.9
        self.TARGET_F1 = 0.9
        self.TARGET_AUROC = 0.9

        self.local_accuracies = []
        self.local_f1scores = []
        self.local_aurocs = []

        self.global_accuracies = []
        self.global_f1scores = []
        self.global_aurocs = []


    def run(self) -> None:
        """
        Executes the full training and evaluation lifecycle: 
        Initializes model(s), 
        Performs training loop, 
        Evaluates and logs performance 
        and Saves models and visualizations.
        """   

        self.set_seed(self.RANDOM_STATE) 
        logging.info('--- This is a demo of Heart Failure Classification with Federated Learning ---')
        logging.info(f'Your IP: {self.fl_client.agent_ip}')
        logging.info(f'Your agent name: {self.agent_nr}')

        initial_models = training(dict(), self.agent_nr, init_flag=True)

        
        self.model_size_reported = compute_total_transfer(
            initial_models,
            num_clients=self.NUM_CLIENTS,
            num_rounds=self.MAX_TRAINING_ROUNDS,
            bi_directional=True # assume upload & download
        )
        logging.info(f"Total estimated communication: {self.model_size_reported / (1024**2):.2f} MB")

        self.fl_client.send_initial_model(initial_models)
        self.fl_client.start_fl_client()

        # === Federated round begins ===
        while judge_termination(self.TRAINING_COUNT, self.GM_ARRIVAL_COUNT, self.MAX_TRAINING_ROUNDS):

            # Wait for global model
            self.global_models, self.final_gm_path, self.final_gm_file, self.final_gm_data_dict = self.fl_client.wait_for_global_model()
            self.GM_ARRIVAL_COUNT += 1
            logging.info(f'--- Global model has arrived ---')

            # === Global model evaluation ===
            global_acc, global_f1, global_auroc = compute_performance(
                self.global_models, prep_test_data(), False, self.agent_nr)
            logging.info(f'--- Global model evaluation Done ---')

            self.global_accuracies.append(global_acc)
            self.global_f1scores.append(global_f1)
            self.global_aurocs.append(global_auroc)

            # === Local training and evaluation ===

            # Local model training
            local_models = training(self.global_models, self.agent_nr)
            self.TRAINING_COUNT += 1
            logging.info(f'--- Client model training Done ---')

            # Evaluate local model
            local_acc, local_f1, local_auroc = compute_performance(
                local_models, prep_test_data(), True, self.agent_nr)
            logging.info(f'--- Local model evaluation Done ---')

            self.local_accuracies.append(local_acc)
            self.local_f1scores.append(local_f1)
            self.local_aurocs.append(local_auroc)

            # Send trained local model
            self.fl_client.send_trained_model(local_models, int(TrainingMetaData.NUM_TRAINING_DATA), local_acc)


        self.compute_final_metrics()
        self.log_performance_summary()
        self.explain_final_model_with_shap()
        self.save_and_plot()


    def compute_final_metrics(self) -> Tuple[List[float], List[float], List[Optional[int]]]:
        """
        Computes best and final performance metrics (accuracy, F1, AUROC),
        and identifies the training round at which performance thresholds are first met.

        Returns:
            Tuple[List[float], List[float], List[Optional[int]]]: Best metrics, Final metrics,
            Round numbers at which target metrics were reached.
        """

        self.best_acc = max(self.global_accuracies)
        if TrainingMetaData.BINARY:
            self.best_f1 = max(self.global_f1scores)
        else:
            self.global_f1scores_avg = [sum(arr)/len(arr) for arr in self.global_f1scores if len(arr) > 0]
            self.best_f1 = max(self.global_f1scores_avg)
        self.best_auroc = max(self.global_aurocs)

        self.best_list = [self.best_acc, self.best_f1, self.best_auroc]

        self.final_acc = self.global_accuracies[-1]
        self.final_f1 = self.global_f1scores[-1] if TrainingMetaData.BINARY else self.global_f1scores_avg[-1]
        self.final_auroc = self.global_aurocs[-1]

        self.final_list = [self.final_acc, self.final_f1, self.final_auroc]

        self.round_w_acc_higher_than_target = [i for i, acc in enumerate(self.global_accuracies) if acc >= self.TARGET_ACCURACY]
        f1_source = self.global_f1scores if TrainingMetaData.BINARY else self.global_f1scores_avg
        self.round_w_f1_higher_than_target = [i for i, f1 in enumerate(f1_source) if f1 >= self.TARGET_F1]
        self.round_w_auroc_higher_than_target = [i for i, auroc in enumerate(self.global_aurocs) if auroc >= self.TARGET_AUROC]

        self.round_to_target_acc = self.round_w_acc_higher_than_target[0] if len(self.round_w_acc_higher_than_target) >= 1 else None
        self.round_to_target_f1 = self.round_w_f1_higher_than_target[0] if len(self.round_w_f1_higher_than_target) >=  1 else None
        self.round_to_target_auroc = self.round_w_auroc_higher_than_target[0] if len(self.round_w_auroc_higher_than_target) >= 1 else None
        
        self.round_to_target_list = [self.round_to_target_acc, self.round_to_target_f1, self.round_to_target_auroc]

        return self.best_list, self.final_list, self.round_to_target_list 

    def log_performance_summary(self) -> None:
        """
        Logs the best and final model performance,
        as well as the number of training rounds needed to meet target thresholds.
        """

        metric_list = ["acc", "f1", "auroc"]
        target_list = [self.TARGET_ACCURACY, self.TARGET_F1, self.TARGET_AUROC]

        logging.info(' ---- Best performance metrics')
        for metric, best in zip(metric_list, self.best_list):
            logging.info(f'{metric}: {best}')

        logging.info(' ---- Final performance metrics')
        for metric, final in zip(metric_list, self.final_list):
            logging.info(f'{metric}: {final}')

        logging.info(' ---- Rounds to target performance threshold')
        for metric, round in zip(metric_list, self.round_to_target_list):
            logging.info(f'{metric}: {round}')

        logging.info(' ---- Total communication cost to reach target metrics ...')
        for target, metric, round in zip(target_list, metric_list, self.round_to_target_list):
            if round is not None:
                comm = compute_total_transfer(self.global_models, self.NUM_CLIENTS, round)
                logging.info(f"Total communication cost to reach {target} {metric}: {comm / (1024**2):.3f} MB")


    def explain_final_model_with_shap(self) -> None:
        """
        Generates SHAP explanations on the final model using a held-out test dataset.
        Plots global feature importance and logs feature-level contributions.
        """

        logging.info("\n[SHAP] Running explainability on final global model...")
        final_model = Converter.cvtr().convert_dict_nparray_to_nn(self.global_models)
        test_loader = prep_test_data()
        explain_model_with_shap(final_model, test_loader)

    def save_and_plot(self) -> None:
        """
        Saves the final trained model to disk and generates performance plots.
        """

        if TrainingMetaData.BINARY:
            perf_dict = store_performance_data(self.global_accuracies[-1], 
            self.global_f1scores[-1], self.global_aurocs[-1])
        else:
            perf_dict = store_performance_data(self.global_accuracies[-1], 
            self.global_f1scores_avg[-1], self.global_aurocs[-1])

        save_model_file(self.global_models, self.final_gm_path, self.final_gm_file, 
        perf_dict, final_flag=True)

        plot_performance_federated(
            self.global_accuracies,
            self.global_f1scores,
            self.global_aurocs,
            self.local_accuracies,
            self.local_f1scores,
            self.local_aurocs,
            self.agent_nr,
            TrainingMetaData.BINARY,
            "per_class"
        )
        logging.info('--- FL Plotting Done ---')

    
    def set_seed(self, seed: int = 42) -> None:
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): Seed value to use.
        """ 

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # Entry point for local testing or deployment. 
    # You can choose to run either centralized or federated learning 
    # for Heart Failure classification by uncommenting. 

    # Uncomment when Centralized learning
    #cl_engine = CLRunner()
    #cl_engine.run()

    
    fl_engine = FLRunner()
    fl_engine = fl_engine.run()
