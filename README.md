# MLPipeline_v1.0
An example application showcasing full end to end machine learning pipeline built in Python. Taking an existing built ML algorithm to production is an altogether different game where we need to think of how we will build a scalable pipeline that can be integrated with other applications/product. Read more about the approach here <insert blog link>

## Brief Description and Usage
The application reads input from the configuration file (.ini). There are 2 major modules in the application 
1. Training module -  This module takes input from the data sources and trains the ML algorithm. Later it serializes the ML algorithm built so that it can be used in the Predict module
2. Predict Module -  This module loads the trained model and gives the output based on the input data provided. It can be tweaked as per the need of integrating it to other applications

Usage (for training) -  python MLTrain\MLTrain.py --config Others\mlconfigtrain.Ini
Usage (for Prediction) - python MLTrain\MLPredict.py --config Others\mlconfigpredict.Ini

