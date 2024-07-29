# xtream AI Challenge - Software Engineer

⬇️ Scroll to the end to read the assignment guidelines ⬇️

## How to run
![diamond](https://img.itch.zone/aW1hZ2UvMTEwMDA2OC82MzQ0MTg0LmdpZg==/794x1000/L%2Fyy05.gif)

### Installation ⚙️
This project is managed with [`PDM`](https://pdm-project.org/en/latest). If you don't have it yet, make sure to install it.
MacOS users can rely on Homebrew - you can find the installation guide for other systems [here](https://pdm-project.org/en/latest/#installation).
```bash
brew install pdm
```

Next, install the necessary dependencies for the project.
```bash
pdm install
```

Now install **MongoDB** (Community Edition). Like before, you can use Homebrew on MacOS - for other systems, see this [guide](https://www.mongodb.com/docs/manual/administration/install-community/).
```bash
brew tap mongodb/brew
```
```bash
brew install mongodb-community
```

### ML Pipeline 🤖
Now you can launch the pipeline. Just run:
```bash
pdm run python scripts/run_ml_pipeline.py
```
This script starts a training cycle for **linear** and **gradient boosting** models. During each cycle, the following assets are created and stored: the processed datasets used for training, the various models, and a training log. A copy of the **best-performing** model is stored separately, for ease of access.


All related assets are mapped via unique identifiers: the details can be retrieved from the training log, which is updated at each cycle.


To simplify first-time usage, I included a ```.env``` file with minimal configurations. This ensures that the assets are saved in the correct place when you operate the tool locally. However, you can bypass these settings from the CLI (ideally, redirecting the assets externally):
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

### REST API & Database
#### MongoDB 📊
Before trying out the API, be sure to start a local MongoDB instance. The uvicorn server will need to connect to it.
```bash
brew services start mongodb-community
```

#### FastAPI ⚡️
It's time to start the FastAPI uvicorn server and put to work the models we trained!
```bash
pdm run fastapi run src/xtream_service/api/diamond.py
```

Once you open your browser, navigate to the docs - if you aren't there yet, - and test to your heart's content.
```
http://127.0.0.1:8000/docs
```
All the requests/responses are stored in the MongoDB local instance. You can run ```mongosh``` in the shell to access it and check the result first-hand:
- ```show dbs``` to list existing databases.
- ```use diamond_db``` to access the database we created.
- ```show collections``` to list the existing collections (database tables).
- ```db.diamond_request.find().pretty()``` or ```db.diamond_response.find().pretty()``` to view the data.

All actions are also logged in the shell while FastAPI is running.


**P.S.** 🚨 If the uvicorn server triggers a security alert in your browser, please ignore it. This happens because the app is running on ```http``` (```https``` has not been configured). Alternatively, you can simply start the server using ```fastapi dev```.


**P.P.S.** When you are done playing with the API, don't forget to put MongoDB to sleep 😴
```bash
brew services stop mongodb/brew/mongodb-community
```

Thanks for checking out this project 🤓

-----
-----

## Assignment

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
