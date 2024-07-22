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
### Installation
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

### Automated pipeline

### REST API
