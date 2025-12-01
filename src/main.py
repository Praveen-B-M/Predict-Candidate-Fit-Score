# src/main.py
import argparse
import os
from .train import train
from .model import load_model
from .api import app as fastapi_app
import uvicorn

def run_train(candidates, jobs, model_path):
    from .train import train as train_fn
    train_fn(candidates, jobs, model_path)

def run_serve(model_path, port):
    os.environ.setdefault("MODEL_PATH", model_path)
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)

def run_predict(model_path, candidate, job):
    from .model import load_model
    from .predict import predict_from_text
    bundle = load_model(model_path)
    res = predict_from_text(candidate, job, bundle)
    print(res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--candidates", type=str, default="data/candidates.csv")
    parser.add_argument("--jobs", type=str, default="data/jobs.csv")
    parser.add_argument("--model-path", type=str, default="models/matcher.joblib")
    parser.add_argument("--candidate", type=str, default="")
    parser.add_argument("--job", type=str, default="")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.train:
        print("Training model...")
        from .train import train as train_fn
        train_fn(args.candidates, args.jobs, args.model_path)
    elif args.serve:
        print("Serving API and UI at http://0.0.0.0:%d/ui" % args.port)
        run_serve(args.model_path, args.port)
    elif args.predict:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError("Model missing. Train first.")
        run_predict(args.model_path, args.candidate, args.job)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
