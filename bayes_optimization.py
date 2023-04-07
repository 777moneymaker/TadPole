import fasttext_train as ft
import word2vec_train as w2v
import utils
import json
from pathlib import Path
from bayes_opt import BayesianOptimization
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events
import typing
import os


class ModelOptLogger(_Tracker):
    def __init__(self, path, eval_func, reset=True):
        self._path = path if path[-5:] == ".json" else path + ".json"
        self._eval_func = eval_func
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

        # print(dict(data))

        with open(self._path, "a") as f:
            f.write(json.dumps(dict(data)) + "\n")

        self._update_tracker(event, instance)


class BayesianOptimizer(object):

    __slots__ = ("initial_model", "hyperparams", "initial_points", "num_iterations","best_model", "best_score", "current_function",
                 "opt_name", "best_model_path")

    def __init__(self, initial_model: typing.Union[w2v.Word2VecPipeline, ft.FastTextPipeline], hyperparams: dict, initial_points: int,
                  num_iterations: int, opt_name: str, best_model_path: Path):
        self.initial_model = initial_model
        self.hyperparams = hyperparams
        self.initial_points = initial_points
        self.num_iterations = num_iterations
        self.best_model = None
        self.best_score = 0
        self.current_function = ""
        self.opt_name = opt_name
        self.best_model_path = best_model_path
    
    # TODO: floats to ints? - it must be possible to differentiate
    def _map_hyperparams(self, model, **kwargs):
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, int(value))
    
    def _get_local_best_score(self, scores):
        func = max(scores, key=scores.get)
        score = scores[func]
        return func, score
    
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
            self.initial_model.run()
            scores = self.initial_model.result
            self.current_function, local_best_score = self._get_local_best_score(scores)
            print(self.current_function)
            print(local_best_score)
            # self.current_function = max(scores, key=scores.get)
            # local_best_score = scores[self.current_function]
            observer._eval_func = self.current_function
            if local_best_score > self.best_score:
                self.best_score = local_best_score
                self.best_model = self.initial_model
                print(f"[OPT]   New best: {self.best_score}")
            return local_best_score
    
        optimizer = BayesianOptimization(
            f=objective_func,
            pbounds=self.hyperparams,
            random_state=1,
            verbose=2
        )
        
        # TODO: defineable logging path
        Path("logs").mkdir(exist_ok=True)
        observer = ModelOptLogger(path="./logs/bayes_test.json", eval_func=self.current_function)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, observer)
        optimizer.maximize(
            init_points=self.initial_points,
            n_iter=self.num_iterations
        )

        return optimizer.max



