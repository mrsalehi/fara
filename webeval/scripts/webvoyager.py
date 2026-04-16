from webeval.systems.websurfer import WebSurferSystem
from webeval.benchmarks import WebVoyagerBenchmark
from pathlib import Path
import numpy as np
import os
import logging
# from aztool.workspace import Workspace, AIF_WORKSPACE
import mlflow
from eval_exp import EvalExp, ModelReference, get_foundry_endpoint_configs
from webeval.oai_clients.graceful_client import GracefulRetryClient
from webeval.eval_result import EvalResult, Stage
from arg_parsing import get_eval_args
import tempfile
import json


DEFAULT_DATA_URL = '../data/webvoyager/WebVoyager_data_08312025.jsonl'


def _load_browserbase_env_from_keys_json() -> None:
    """Populate Browserbase env vars from keys.json if they are missing."""
    if os.environ.get("BROWSERBASE_API_KEY") and os.environ.get("BROWSERBASE_PROJECT_ID"):
        return

    candidate_paths = [
        Path(__file__).resolve().parents[3] / "keys.json",  # evoskill-march11/keys.json
        Path(__file__).resolve().parent / "keys.json",
    ]

    for keys_path in candidate_paths:
        if not keys_path.exists():
            continue
        try:
            with open(keys_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        api_key = data.get("browser_base_api_key")
        project_id = data.get("browser_base_project_id")
        if api_key and not os.environ.get("BROWSERBASE_API_KEY"):
            os.environ["BROWSERBASE_API_KEY"] = api_key
        if project_id and not os.environ.get("BROWSERBASE_PROJECT_ID"):
            os.environ["BROWSERBASE_PROJECT_ID"] = project_id
        break

class Callback:
    def __init__(self):
        self.scores = []

    def __call__(self, result: EvalResult, mlflow_facade, mlflow_run_id: str):
        if result.stage == Stage.EVALUATED:
            self.scores.append(result.score)
        mlflow_facade.log_metric('score', np.mean(self.scores or [0]), run_id = mlflow_run_id)


def add_webvoyager_args(parser):
    parser.add_argument('--eval_data_url', type=str, default = DEFAULT_DATA_URL, help='Azure URI to the evaluation data (None for vanilla webvoyager)')
    parser.add_argument('--local', action='store_true', help='Run model locally instead of calling a vLLM API endpoint')
    parser.add_argument('--local_model_id', type=str, default='microsoft/Fara-7B', help='HuggingFace model ID for local inference')


def main():
    args = get_eval_args(add_webvoyager_args)

    # Use an ephemeral local tracking DB unless the user provides a tracking URI.
    # This avoids failures caused by stale schemas in persistent sqlite files.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        tmp_tracking_dir = tempfile.mkdtemp(prefix="webvoyager-mlflow-")
        mlflow.set_tracking_uri(f"sqlite:///{tmp_tracking_dir}/mlflow.db")

    _load_browserbase_env_from_keys_json()

    if args.browserbase:
        assert os.environ.get("BROWSERBASE_API_KEY"), "BROWSERBASE_API_KEY environment variable must be set to use browserbase"
        assert os.environ.get("BROWSERBASE_PROJECT_ID"), "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables must be set to use browserbase"

    experiment = EvalExp(
        ws = None, 
        user = args.user,
        seed = args.seed)
  
    with experiment.start_run() as run:
        model_ref = ModelReference(args.model_url, args.model_port, args.device_id, args.web_surfer_kwargs.get('max_n_images', 3), args.gpt_solver_model_name, args.dtype, args.enforce_eager, use_external_endpoint=bool(args.model_endpoint))

        logger = logging.getLogger('webvoyager-eval')
        logger.setLevel(logging.INFO)

        mlflow.log_param('max_rounds', args.max_rounds)
        if args.web_surfer_model_type == "gpt_solver":
            mlflow.log_param('web_surfer_model_type', args.web_surfer_model_type + "/" + args.gpt_solver_model_name)
        else:
            mlflow.log_param('web_surfer_model_type', args.web_surfer_model_type)
        mlflow.log_param('fn_call_template', args.fn_call_template)

        # If using external endpoint, load all endpoint configs into a list of dicts
        if args.model_endpoint:
            websurfer_client_cfg = get_foundry_endpoint_configs(args.model_endpoint)
            logger.info(f"Loaded {len(websurfer_client_cfg)} external endpoint config(s) from {args.model_endpoint}")
            model_ref.model_url_to_log = websurfer_client_cfg[0]['base_url']  # log the first endpoint URL as
            model_ref.model_to_log = websurfer_client_cfg[0]['base_url']
            mlflow.log_param('using_external_endpoint', True)
            mlflow.log_param('endpoint_config_path', ','.join([x['base_url'] for x in websurfer_client_cfg]))
        else:
            # For local VLLM, use a simple flat config structure that FaraAgent expects
            websurfer_client_cfg = {
                "api_key": "NONE",
                "model": "gpt-4o-mini-2024-07-18",
                "base_url": f"http://0.0.0.0:{args.model_port}/v1"
            }
            if args.web_surfer_client_cfg is not None:
                websurfer_client_cfg = args.web_surfer_client_cfg        
        if args.web_surfer_kwargs:
            mlflow.log_params({f'web_surfer_kwargs.{k}': v  for k, v in args.web_surfer_kwargs.items()})
        system = WebSurferSystem(
            system_name="WebSurfer",
            web_surfer_model_type = args.web_surfer_model_type,
            max_rounds = args.max_rounds,
            websurfer_client_cfg = websurfer_client_cfg,
            start_on_target_url=True,
            browserbase=args.browserbase,
            web_surfer_kwargs=args.web_surfer_kwargs,
            gpt_solver_model_name=args.gpt_solver_model_name,
            fn_call_template=args.fn_call_template,
            step_budgets=args.step_budgets,
            use_local_model=args.local,
            local_model_id=args.local_model_id,
        )


        mlflow.log_param("eval_data", args.eval_data_url)
        mlflow.log_param("eval_model", args.eval_model)
        # set data_dir to absolute path of this file, then go to ../data/webvoyager
        data_dir = Path(__file__).resolve().parent.parent / "data" / "webvoyager"
        data_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_eval:
            eval_client = None
        else:
            eval_client = GracefulRetryClient.from_path(args.eval_oai_config, logger=logger, eval_model=args.eval_model)
        benchmark = WebVoyagerBenchmark(
            data_dir=data_dir,
            eval_method="gpt_eval" if not args.skip_eval else "exact_match",
            model_client=eval_client,
            data_az_folder = args.eval_data_url,
        )
        if isinstance(args.eval_data_url, str) and args.eval_data_url.lower().endswith('.jsonl'):
            benchmark.data_file = str(Path(args.eval_data_url).expanduser().resolve())

        mlflow.log_param('subsample', args.subsample)
        mlflow.log_param('processes', args.processes)
        mlflow.log_param('max_error_task_retries', args.max_error_task_retries)
        experiment.run(
            model_ref = model_ref,
            system = system,
            benchmark = benchmark,
            out_url = args.out_url,
            subsample = args.subsample,
            redo_eval = args.redo_eval,
            run_id = args.run_id,
            split = "webvoyager",
            processes = args.processes,
            callbacks = [Callback()],
            eval_only = args.eval_only,
            max_error_task_retries = args.max_error_task_retries,
            skip_eval = args.skip_eval)


if __name__ == "__main__":
    main()
