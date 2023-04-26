import fasttext_train as ft
import word2vec_train as w2v
import utils
import json
from pathlib import Path
from bayes_opt import BayesianOptimization
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
import typing
import os
import asyncio


class ModelOptLogger(_Tracker):
    def __init__(self, path, eval_func, correct_percent, not_evaluated, reset=True):
        self._path = path if path[-5:] == ".json" else path + ".json"
        self._eval_func = eval_func
        self._correct_percent = correct_percent
        self._not_evaluated = not_evaluated
        if reset:
            try:
                os.remove(self._path)
            except OSError:
                pass
        super(ModelOptLogger, self).__init__()

    def update(self, event, instance):
        data = dict(instance.res[-1])

        now, time_elapsed, time_delta = self._time_metrics()
        data["datetime"] = {
            "datetime": now,
            "elapsed": time_elapsed,
            "delta": time_delta,
        }
        data["function"] = self._eval_func
        data["correct_percent_per_category"] = self._correct_percent
        data["not_evaluated_num"] = self._not_evaluated

        # print(dict(data))
        # if event == 'optimization:end':
        #     kwargs = {
        #         'opt_name': 
        #     }
        #     loop = asyncio.get_event_loop()
        #     loop.run_until_complete(utils.report_opt_result_teams())

        with open(self._path, "a") as f:
            f.write(json.dumps(dict(data)) + "\n")

        self._update_tracker(event, instance)


# TODO: let user choose the aquisition function and set its parameters
class BayesianOptimizer(object):

    __slots__ = ("initial_model", "hyperparams", "initial_points", "num_iterations","best_model", "best_score", "current_function",
                 "opt_name", "output_path", "aquisition_function", "kappa", "kappa_decay", "kappa_decay_delay", "xi", 
                 "current_correct_percentage", "current_not_evaluated_num")

    def __init__(self, initial_model: typing.Union[w2v.Word2VecPipeline, ft.FastTextPipeline], hyperparams: dict, initial_points: int,
                  num_iterations: int, opt_name: str, output_path: Path, aquisition_function: typing.Literal['ucb', 'ei', 'poi'] = 'ucb',
                kappa: float = 2.576, xi: float = 0, kappa_decay: float = 1, kappa_decay_delay: int = 0):
        self.initial_model = initial_model
        self.hyperparams = hyperparams
        self.initial_points = initial_points
        self.num_iterations = num_iterations
        self.best_model = None
        self.best_score = 0
        self.current_function = ""
        self.current_correct_percentage = None
        self.current_not_evaluated_num = 0
        self.opt_name = opt_name
        self.output_path = output_path
        self.aquisition_function = aquisition_function
        self.kappa = kappa
        self.kappa_decay = kappa_decay
        self.kappa_decay_delay = kappa_decay_delay
        self.xi = xi
    
    def _map_hyperparams(self, model, **kwargs):
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, int(value)) if isinstance(getattr(model, key), int) else setattr(model, key, value)
    
    def _get_local_best_score(self, scores):
        func = max(scores, key=scores.get)
        score = scores[func]
        return func, score

    def _report_opt_result_callback(self, event, instance):
        kwargs = {
            'opt_name': self.best_model.output_prefix,
            'best': self.best_score,
            'best_func': self.current_function, # TODO: this is not the overall best func i think
            'hypers': self.hyperparams,
            'category': self.current_correct_percentage # TODO: not overall best
        }
        loop = asyncio.get_event_loop()
        loop.run_until_complete(utils.report_opt_result_teams(**kwargs))
    
    def optimize(self):
        def objective_func(**kwargs):
            # kwargs['model'] = self.initial_model
            print(kwargs)
            print(self.initial_model)
            # self._map_hyperparams(**kwargs)
            self._map_hyperparams(model=self.initial_model, **kwargs)
            # for key, value in kwargs.items():
            #     if hasattr(self.initial_model, key):
            #         setattr(self.initial_model, key, int(value))
            try:
                self.initial_model.run()
            except ValueError as e:
                print(f"Training failed: {e} /nOptimization score for this iteration will be set to 0. Parameters for the next iteration will be semi-random.")
                observer._eval_func = "Training failed"
                return 0
            scores = self.initial_model.result
            self.current_function, local_best_score = self._get_local_best_score(scores)
            self.current_correct_percentage = self.initial_model.correct_percent_per_category
            self.current_not_evaluated_num = self.initial_model.not_evaluated_number
            print(self.current_function)
            print(local_best_score)
            # self.current_function = max(scores, key=scores.get)
            # local_best_score = scores[self.current_function]
            observer._eval_func = self.current_function
            observer._correct_percent = self.current_correct_percentage
            observer._not_evaluated = self.current_not_evaluated_num
            
            if local_best_score > self.best_score:
                self.best_score = local_best_score
                self.best_model = self.initial_model
                best_model_path = self.output_path / f"{self.best_model.model_name}.model"
                self.best_model.model_object.save(best_model_path.as_posix())
                print(f"[OPT]   New best: {self.best_score}")
            return local_best_score
    
        optimizer = BayesianOptimization(
            f=objective_func,
            pbounds=self.hyperparams,
            random_state=1,
            verbose=2
        )
        
        self.output_path.mkdir(exist_ok=True)
        log_path = self.output_path / f"{self.opt_name}.json"
        # observer = ModelOptLogger(path="./logs/bayes_test.json", eval_func=self.current_function)
        observer = ModelOptLogger(path=log_path.as_posix(), 
                                  eval_func=self.current_function, 
                                  correct_percent=self.current_correct_percentage, 
                                  not_evaluated=self.current_not_evaluated_num)
        aquisition_func = UtilityFunction(kind=self.aquisition_function,
                                          kappa_decay_delay=self.kappa_decay_delay,
                                          kappa_decay=self.kappa_decay,
                                          kappa=self.kappa,
                                          xi=self.xi)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, observer)
        optimizer.subscribe(event=Events.OPTIMIZATION_END, subscriber="Any hashable object", callback=self._report_opt_result_callback)
        optimizer.maximize(
            init_points=self.initial_points,
            n_iter=self.num_iterations,
            acquisition_function=aquisition_func
        )

        return optimizer.max



