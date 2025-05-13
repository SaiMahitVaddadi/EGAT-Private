import itertools
from train import Train
from hyperopt import fmin, tpe, hp, Trials
import optuna
import ray
from ray import tune
import nevergrad as ng
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ax import optimize
import nni
from nni.experiment import Experiment





class Tune:
    def __init__(self, params):
        self.params = params
        # Initialize variables to store the best parameters and the best score
        self.best_params = None
        self.best_score = float('-inf')


    def create_combos(self):
        self.param_combinations = list(itertools.product(*self.params.tuning_params.values()))
    
    def train_iteration(self,pars):
        param_dict = dict(zip(self.params.tuning_params.keys(), pars))
    
        # Initialize the Train class with the current parameters
        trainer = Train(params=param_dict)
    
        # Train the model and get the score
        score = trainer.TrainingProtocol()
        return score,param_dict
    


    def grid_search(self):
        for params in self.param_combinations:
            score,param_dict = self.train_iteration(params)
            # Update the best parameters if the current score is better
            if score < self.best_score:
                self.best_score = score
                self.best_params = param_dict
    

    def setup_bayesian_space(self):
        if self.params.tuning_method == 'hyperopt':
            self.space = {
                key: hp.choice(key, value) for key, value in self.params.tuning_params.items()
            }
        elif self.params.tuning_method == 'optuna':
            self.space = {
                key: hp.choice(key, value) for key, value in self.params.tuning_params.items()
            }
        elif self.params.tuning_method == 'raytune':
            self.space = {
                key: tune.choice(value) for key, value in self.params.tuning_params.items()
            }
        elif self.params.tuning_method == 'nevergrad':
            self.space = {
                key: ng.p.Choice(value) for key, value in self.params.tuning_params.items()
            }
        elif self.params.tuning_method == 'smac3':
            self.space = {
                key: CategoricalHyperparameter(key, value) for key, value in self.params.tuning_params.items()
            }


    

    def objective(self,params):
        trainer = Train(params=params)
        score = trainer.train_and_evaluate()
        if self.params.tuning_method == 'hyperopt':
            return score
        elif self.params.tuning_method == 'optuna':
            return score
        elif self.params.tuning_method == 'raytune':
            tune.report(score=score)
        elif self.params.tuning_method == 'nevergrad':
            return -score
        elif self.params.tuning_method == 'smac3':
            return -score
    

    def bayesian_optimization(self):
        if self.params.tuning_method == 'hyperopt':
            trials = Trials()
            best_params = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=self.params.tuning_iterations, trials=trials)
        elif self.params.tuning_method == 'optuna':
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=self.params.tuning_iterations)
            best_params = study.best_params
        elif self.params.tuning_method == 'raytune':
            analysis = tune.run(
                self.objective,
                config=self.space,
                num_samples=self.params.tuning_iterations,
                mode='max',
                metric='score'
            )
            best_params = analysis.best_config
        elif self.params.tuning_method == 'nevergrad':
            optimizer = ng.optimizers.OnePlusOne(parametrization=self.space, budget=self.params.tuning_iterations)
            recommendation = optimizer.minimize(self.objective)
            best_params = recommendation.value
        elif self.params.tuning_method == 'smac3':
            scenario = Scenario({
                "run_obj": "quality",
                "runcount-limit": self.params.tuning_iterations,
                "cs": self.space,
                "deterministic": "true"
            })
            smac = SMAC4HPO(scenario=scenario, tae_runner=self.objective)
            best_params = smac.optimize()
        return best_params

    
    
    def beam_search(self, beam_width=3):
        # Initialize the beam with the first set of parameters
        beam = [dict(zip(self.params.tuning_params.keys(), params)) for params in itertools.product(*[self.params.tuning_params[key][:1] for key in self.params.tuning_params])]
        best_params = None
        best_score = float('-inf')

        while beam:
            new_beam = []
            for params in beam:
                for key in self.params.tuning_params:
                    for value in self.params.tuning_params[key]:
                        new_params = params.copy()
                        new_params[key] = value
                        score, _ = self.train_iteration(new_params.values())
                        new_beam.append((score, new_params))
                        if score > best_score:
                            best_score = score
                            best_params = new_params
            # Sort the new beam by score and keep the top beam_width elements
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = [params for score, params in new_beam[:beam_width]]

        return best_params, best_score

        
    def adaptive_grid_search(self, refinement_steps=3):
        # Initialize the grid with the initial parameter ranges
        param_grid = self.params.tuning_params
        best_params = None
        best_score = float('-inf')

        for step in range(refinement_steps):
            param_combinations = list(itertools.product(*param_grid.values()))
            for params in param_combinations:
                score, param_dict = self.train_iteration(params)
                if score > best_score:
                    best_score = score
                    best_params = param_dict

            # Refine the parameter grid around the best parameters
            for key in param_grid:
                best_value = best_params[key]
                values = self.params.tuning_params[key]
                if isinstance(values[0], (int, float)):
                    # For numerical parameters, create a smaller range around the best value
                    min_value = max(min(values), best_value - (max(values) - min(values)) / 4)
                    max_value = min(max(values), best_value + (max(values) - min(values)) / 4)
                    param_grid[key] = list(np.linspace(min_value, max_value, num=len(values)))
                else:
                    # For categorical parameters, keep the best value and a few others
                    param_grid[key] = [best_value] + [v for v in values if v != best_value][:len(values) // 2]

        return best_params, best_score


    def random_search(self, num_samples=100):
        best_params = None
        best_score = float('-inf')
        for _ in range(num_samples):
            params = {key: np.random.choice(value) for key, value in self.params.tuning_params.items()}
            score, param_dict = self.train_iteration(params.values())
            if score > best_score:
                best_score = score
                best_params = param_dict
        return best_params, best_score

    def hyperband(self, max_iter=81, eta=3):
        def get_hyperband_bracket(max_iter, eta):
            s_max = int(np.log(max_iter) / np.log(eta))
            B = (s_max + 1) * max_iter
            brackets = []
            for s in range(s_max + 1):
                n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))
                r = max_iter * eta ** (-s)
                brackets.append((n, r))
            return brackets

        best_params = None
        best_score = float('-inf')
        brackets = get_hyperband_bracket(max_iter, eta)
        for n, r in brackets:
            configs = [{key: np.random.choice(value) for key, value in self.params.tuning_params.items()} for _ in range(n)]
            scores = []
            for config in configs:
                score, _ = self.train_iteration(config.values())
                scores.append((score, config))
            scores.sort(key=lambda x: x[0], reverse=True)
            configs = [config for _, config in scores[:int(n / eta)]]
            for config in configs:
                score, param_dict = self.train_iteration(config.values())
                if score > best_score:
                    best_score = score
                    best_params = param_dict
        return best_params, best_score

    def successive_halving(self, max_iter=81, eta=3):
        best_params = None
        best_score = float('-inf')
        n = len(self.params.tuning_params)
        r = max_iter
        configs = [{key: np.random.choice(value) for key, value in self.params.tuning_params.items()} for _ in range(n)]
        while len(configs) > 1:
            scores = []
            for config in configs:
                score, _ = self.train_iteration(config.values())
                scores.append((score, config))
            scores.sort(key=lambda x: x[0], reverse=True)
            configs = [config for _, config in scores[:int(len(configs) / eta)]]
            r = int(r * eta)
            for config in configs:
                score, param_dict = self.train_iteration(config.values())
                if score > best_score:
                    best_score = score
                    best_params = param_dict
        return best_params, best_score


    def ax_optimization(self):

        def ax_objective(parameterization):
            params = {key: parameterization[key] for key in self.params.tuning_params.keys()}
            score, _ = self.train_iteration(params.values())
            return score

        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": key, "type": "choice", "values": value} for key, value in self.params.tuning_params.items()
            ],
            evaluation_function=ax_objective,
            minimize=False,
            total_trials=self.params.tuning_iterations
        )

        best_params = {key: best_parameters[key] for key in self.params.tuning_params.keys()}
        best_score = max(values)
        return best_params, best_score


    def nni_search(self):

        def nni_objective(params):
            score, _ = self.train_iteration(params.values())
            nni.report_final_result(score)
            return score

        search_space = {
            key: {"_type": "choice", "_value": value} for key, value in self.params.tuning_params.items()
        }

        experiment = Experiment('local')
        experiment.config.trial_command = 'python3 /Users/svaddadi/Documents/GitHub/EGAT/src/main/train.py'
        experiment.config.trial_code_directory = '/Users/svaddadi/Documents/GitHub/EGAT/src/main'
        experiment.config.search_space = search_space
        experiment.config.tuner.name = 'TPE'
        experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
        experiment.config.max_trial_number = self.params.tuning_iterations
        experiment.config.trial_concurrency = 2

        experiment.run(8080)
        best_params = experiment.get_best_parameter()
        best_score = experiment.get_best_trial_result()

        return best_params, best_score

    def tune(self):
        self.create_combos()
        self.setup_bayesian_space()

        if self.params.tuning_method == 'grid_search':
            return self.grid_search()
        elif self.params.tuning_method in ['hyperopt', 'optuna', 'raytune', 'nevergrad', 'smac3']:
            return self.bayesian_optimization()
        elif self.params.tuning_method == 'beam_search':
            return self.beam_search()
        elif self.params.tuning_method == 'adaptive_grid_search':
            return self.adaptive_grid_search()
        elif self.params.tuning_method == 'random_search':
            return self.random_search()
        elif self.params.tuning_method == 'hyperband':
            return self.hyperband()
        elif self.params.tuning_method == 'successive_halving':
            return self.successive_halving()
        elif self.params.tuning_method == 'ax_optimization':
            return self.ax_optimization()
        elif self.params.tuning_method == 'nni_search':
            return self.nni_search()
        else:
            raise ValueError(f"Unknown tuning method: {self.params.tuning_method}")