# xtream AI Challenge - Software Engineer

## Context

Marta, a data scientist at xtream, has been working on a project for a client, a large jewelry store. She's been doing a great job, but she has a lot on her plate. So, she asked you to help her out.

Marta provided you with a jupyter notebook containing the work she's done so far and a training dataset. You can find both in this repository. You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough. Now, it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes.
Pick the best linear model -- do not worry about the xgboost model or hyperparameter tuning.
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now, you need to support **both models** that Marta developed: the linear regression and the XGBoost with hyperparameter optimization.
Be careful, in the near future you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'!
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run

![diamond](https://img.itch.zone/aW1hZ2UvMTEwMDA2OC82MzQ0MTg0LmdpZg==/794x1000/L%2Fyy05.gif)

### Installation guide
This project is managed with [`PDM`](https://pdm-project.org/en/latest). If you don't have it yet, make sure to install it.

- MacOS:
```bash
brew install pdm
```

- Linux (also works on Mac, alternatively to Homebrew):
```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

- Windows:
```bash
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | py -
```

**Next**, install the necessary dependencies and you'll be good to go.
```bash
pdm install
```

### ML Pipeline
Now you can launch the pipeline. Just run:
```bash
pdm run python scripts/run_ml_pipeline.py
```
This script starts a training cycle for **linear** and **gradient boosting** models. During each cycle, the following assets are created and stored: the processed datasets used for training, the various models, and a training log. A copy of the **best-performing** model is stored separately, for ease of access.


All related assets are mapped via unique identifiers: the details can be retrieved from the training log, which is updated at each cycle.


To simplify first-time usage, I included a ```.env``` file with minimal configurations. This ensures that the assets are saved in the correct place when you operate the tool locally. However, you can bypass it dynamically from the CLI (ideally, redirecting the assets to an external storage service):
```bash
pdm run python scripts/run_ml_pipeline.py --help
```
```
usage: python run_ml_pipeline.py [-h] [-ds DATA_SOURCE] [-dd DATA_DEST] ...

options:
  -h, --help            show this help message and exit
  -ds DATA_SOURCE, --data-source DATA_SOURCE
                        Data source path.
  -dd DATA_DEST, --data-dest DATA_DEST
                        Data storage path.
  -m MODEL_DEST, --model-dest MODEL_DEST
                        Models storage path.
  -s SOTA_DEST, --sota-dest SOTA_DEST
                        SOTA model storage path.
  -l LOG_DEST, --log-dest LOG_DEST
                        Log storage path.
  -n N_MODELS, --n-models N_MODELS
                        Number of training attempts per model type.
```

### REST API
It's time to start the FastAPI server and put to work the models we just trained!

```bash
pdm run fastapi run src/xtream_service/api/app.py
```
Once you open your local webpage, if you aren't there yet, navigate to the docs, and test to your heart's content.
```
http://127.0.0.1:8000/docs
```
If you get a security alert from your browser, please ignore it. This happens because the app is running on ```http``` (```https``` has not been configured).
Alternatively, start the server using ```fastapi dev``` instead.
