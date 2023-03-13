# Portfolio

Hello, I am Alec Hantson a ML/Data enthusiast from Belgium! 👋🇧🇪

In my portfolio you can find all of my demos/projects to show my technical knowledge about data and AI (and even references to some non AI projects at the bottom👀). All of the code are notebooks to show my thinking and display my knowledge in a clean way.🤓

If you are interested in that good academic stuff, be sure to check my [bachelor thesis](https://github.com/HantsonAlec/Portfolio/blob/main/Alec_Hantson_BP_final.pdf)!📚

## Table of content:

-   [MACHINE LEARNING](#machine-learning)
    -   [Liniar regression](#liniar-regression)
        -   [Weight predictor](#weight-predictor)
        -   [Lego price predictor](#lego-price-predictor)
    -   [Logistic regression](#logistic-regression)
        -   [Diabetes detector](#diabetes-detector)
        -   [HR helper](#hr-helper)
    -   [Support Vector Machine](#support-vector-machine)
        -   [Canced detection](#cancer-detection)
        -   [Face classifier](#face-classifier)
    -   [Naive Bayes](#naive-bayes)
        -   [Cybertrolls classifier](#cybertrolls-classifier)
    -   [Unsupervised Learning](#unsupervised-learning)
        -   [Customer clustering](#customer-clustering)
    -   [Fuzzy Logic](#fuzzy-logic)
        -   [Traffic lights](#traffic-lights)
        -   [Tipping](#tipping)
-   [DEEP LEARNING](#deep-learning)
    -   [Neural Networks](#neural-networks)
        -   [Character Recognizer](#character-recognizer)
    -   [Convolutional Neural Networks](#convolutional-neural-networks)
        -   [Fashion MNIST](#fashion-mnist)
        -   [Car detection](#car-detection)
    -   [Auto encoders](#auto-encoder)
        -   [Face reconstruction](#face-reconstruction)
        -   [Fraud detection](#fraud-detection)
        -   [Movie recommender](#movie-recommender)
    -   [Long Short Term Memory](#long-short-term-memory)
        -   [Airline sentiment analysis](#face-reconstruction)
-   [REINFORCEMENT LEARNING](#reinforcement-learning)
    -   [Q-learning / Deep Q-learning](#q-learning-/-deep-q-learning)
        -   [Cartpole with Q learning](#cartpole-with-q-learning)
        -   [LunarLander with DQN](#lunarlander-with-dqn)
        -   [Mountaincar with DQN](#mountaincar-with-dqn)
    -   [SARSA / Deep SARSA](#sarsa-learning-/-deep-sarsa)
        -   [Cartpole with SARSA](#cartpole-with-sarsa)
        -   [Mountaincar with Deep SARSA](#mountaincar-with-deep-sarsa)
    -   [Policy gradient](#policy-gradient)
        -   [Cartpole with policy gradient](#cartpole-with-policy-gradient)
-   [SEARCH ALGORITHMS](#search-algorithms)
    -   [Best First Search](#best-first-search)
    -   [Depth First Search](#depth-first-search)
    -   [Iterative Deepening Depth First Search](#iterative-deepening-depth-first-search)
    -   [Bidirectional Search](#bidirectional-search)
    -   [A star](#a-star)
    -   [Uniform Sost Search](#uniform-cost-search)
-   [RESEARCH PROJECT](#research-project)
-   [OTHER](#other)
    -   [Smart Air](#smart-air)
    -   [EveFind](#evefind)
-   [DATACAMP](#datacamp)
    -   [Netflix data](#Netflix-data-on-the-office)
    -   [Google playstore](#Google-playstore-insights)

# Machine Learning

## Liniar regression

Two little project that helped me gain knowledge and understanding of liniar regression models.

What I learned:

-   How to implement linear regression.
-   Data analysis
-   Using different regularistions and comparing them.

### Weight predictor

Simple liniar regression model to predict the weight of a person based on things the person did that week. [View code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Linear_Regression/Linear_Regression_weight_prediction.ipynb)

### Lego price predictor

Predicting the price of a lego set based on features like difficulty and review scores. [View code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Linear_Regression/Linear_Regression_lego_price_prediction.ipynb)

## Logistic regression

With these project I want to show my understanding about working with logistic regression models. Analysis and understanding results.

What I learned:

-   Using Logistic regression.
-   Data analysis and the impact of unbalanced data.
-   Understanding impact of false negatives in some situations.

### Diabetes detector

Model that predicts if a patient has diabetes or not based on medical parameters. I also edited the model to lower the number of flase negatives since we don't want to tell patient they don't have diabetes when they do. [View code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Logistic_Regression/Logistic_Regression_Diabetes_Classifier.ipynb)

### HR helper

With this model an HR department could detect if a employee is planning on leaving the company or not. If they are then they can act on it quickly so he will stay. [View code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Logistic_Regression/Logistic_Regression_HR_Classifier.ipynb)

## Support Vector Machine

Showing my technical knowledge and understanding of SVM's and optimizing with grid search and random search.

What I learned:

-   Ins and outs of Support Vector Machines.
-   Using grid search and random search for hyperparameter optimization.
-   Evaluating difference between SVM's and Logistic regression models.
-   Understanding and using PCA.

### Cancer detection

Compared Logistic regression model with SVM model in detecting cancer based on medical info. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Support_Vector_Machines/SVM_Cancer_Detector.ipynb)

### Face classifier

Here I created a simple face classifier with the purpose of showing knowledge about PCA. Eigenfaces are used to classify Faces. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Support_Vector_Machines/Face%20Detection.ipynb)

## Naive Bayes

NLP model with machine learling instead of deep learning. This shows my technical knowledge of Naive bayes models and a little bit oversampling/augmentation for text.

What I learned:

-   Working with Naive Bayes models.
-   Data oversampling and augmentation.
-   Bag of words principles.

### Cybertrolls classifier

Model that detects if a tweet is from a cybertroll or not. Based on the text and length of the sentence. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Naive_Bayes/Naive_Bayes_Cybertrolls.ipynb)

## Unsupervised Learning

Exploring different clustering techniques.

What I learned:

-   Working with different clustering techniques
-   Learning how to find right parameters for clustering like elbow technique.

### Customer clustering

Using supermarket customer data and clustering to find target groups. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Machine%20Learning/Unsupervised_Learning/Unsupervised_Learning_Clustering.ipynb)

# Deep learning

## Neural Networks

Very basic models, used to gain and show knowledge of the basic working

What I learned:

-   Using Neural Networks
-   Using Tensorflow

### Character recognizer

Simple Neural network that can detect characters based on tabular data instead of images. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/Neural_Networks/Character_Recognizer.ipynb)

## Convolutional Neural Networks

Working with CNN models, both from scratch and pretrained.

What I learned:

-   CNN layers
-   Pretrained models
-   Making object detection from scratch

### Fashion MNIST

CNN model that was trained from scratch to recognize different clothing pieces on the famous fashion MNIST dataset. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/CNN/Fashion_mnist.ipynb)

### Car Detection

Trying out CNN models both trained from scratch and with the use of VGG19. Once model was trained, it was used for object detection. The object detection was made from scratch with a rolling window. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/CNN/Car_detection_from_scratch.ipynb)

## Auto Encoders

Working with auto encoders based on normal Neural Networks and Convolutional Neural Networks.

What I learned:

-   Undercomplete autoencoders
-   Overcomplete autoencoders
-   Denoising autoencoders

### Face reconstruction

Trying out various autoencoder models with CNN layers to reconstruct ocluded faces. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/Auto_Encoders/Face_reconstruction.ipynb)

### Fraud detection

Using an undercomplete autoencoder to detect fraud transactions. Experimenting with threshold to find sweet spot. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/Auto_Encoders/Fraud_Detection.ipynb)

### Movie recommender

Building a movie recommender based on the MovieLens database. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/Auto_Encoders/Movie_recommender.ipynb)

## Long Short Term Memory

Using LSTM models to work with sequences of any kind.(time series or text)

What I learned:

-   Working with time series
-   Using embeddings and tokenizers
-   Different type of LSTM layers

### Maintenance Prediction

Predicting when an engine needs maintenance. First with a normal classification model as baseline before using the time series approach. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/LSTM/Maintenance_prediction.ipynb)

### Airline sentiment

Classifying tweets send about different airlines based on their sentiment. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Deep%20Learning/LSTM/Airline_sentiment.ipynb)

# Reinforcement learning

## Q-learning / Deep Q-learning

Using the Q learning and DQN algorithm to train agents for solving various tasks.

What I learned:

-   Working with Q learning
-   Working with DQN
-   Creating all kinds of rewards to train the agent

### Cartpole with Q learning

Using the Q learning algorithm to train an agent to balance a stick up right. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/Q/cartpole_Q.py)

### LunarLander with DQN

An agent that tries to land a lunar lander between two flags [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/Q/DQN_LunarLander.py)

### MountainCar with DQN

Training an agent that needs to learn how to reach the top of a mountain with a car. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/Q/DQN_MountainCar.py)

## SARSA / Deep SARSA

Using SARSA and Deep SARSA to train agents for solving various tasks. In general these agent will take a less greedy and more save approach.

What I learned:

-   Working with SARSA
-   Working with Deep SARSA
-   Creating all kinds of rewards to train the agent

### Cartpole with SARSA

Converting the Q learning implementation to SARSA. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/SARSA/cartpole_SARSA.py)

### MountainCar with Deep SARSA

Converting the Q learning implementation to SARSA. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/SARSA/DeepSarsa_MountainCar.py)

## Policy Gradient

Making use of policy gradient instead of Q learning or SARSA.

What I learned:

-   Working with policy gradient
-   The implementation difference between PG and Q/SARSA

### Cartpole with policy gradient

Converting the Q learning implementation to policy gradient. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Reinforcement%20Learning/Policy_Gradient/pg_cartpole.py)

# Search Algorithms

Using different algorithms to solve various problems.

What I learned:

-   Differences between algorithms
-   How to implement the different algorithms
-   Learned difference between uninformed and informed search

### Best First Search

Solving a maze with the help of the BFS algorithm. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/BFS/lBFS_ladders_Snakes.py)

### Depth First Search

Instead of BFS now using DFS to solve a maze. Using different algorithms for the same problems really shows the difference between them. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/DFS/DFS_labyrinth.py)

### Iterative Deepening Depth First Search

Last but not least in the Maze serie, Using IDDFS to solve a maze. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/IDS/IDS_maze.py)

### Bidirectional Search

Using Bidirectional Search to solve an 8 puzzle. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/Bidirectional/bidrectional_8_puzzle.py)

### A star

Switching from uninformed search to infromed search to solve the 8 puzzle. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/A*/A_star.py)

### Uniform Cost Search

Getting the best way to run a spartan race with the help of UCS. [View Code](https://github.com/HantsonAlec/Portfolio/blob/main/Search_Algorithms/UCS/UCS_spartan_race.py)

## Fuzzy Logic

Fuzzy logic can help in situation where we have no straight forward right or wrong.

What I learned:

-   Situations to use fuzzy logic
-   converting real life situations into code

### Traffic lights

Fuzzy logic to determine when to switch lights to red or green. [View code and table](https://github.com/HantsonAlec/Portfolio/tree/main/Fuzzy_logic)

### Tipping

With the help of fuzzy logic determening how much to tip at a restaurant [View code and table](https://github.com/HantsonAlec/Portfolio/tree/main/Fuzzy_logic)

# Research project

During my 3th year MCT, I made a research project that laid the founding for my undergraduate thesis. This project/thesis tries to answer the question: "Can the use of transformers gives any advantage to self-driving cars?". The technical part canbe found here. [View Code](https://github.com/HantsonAlec/Research-Project-CARLA)

# Other

Some other academic projects without AI can be found below.

## Smart Air

A project created during my first year MCT. Smart Air is an IoT device that checks the air quality and tries to improve it when needed. [View Code](https://github.com/HantsonAlec/smartAir)

## EveFind

Evefind is a webapp to help EV drivers find charging spots around them. [View Code](https://github.com/HantsonAlec/EveFind)

# Datacamp

Some small datacamp projects to keep the knowledge fresh.

## Netflix data on the office

Data investigation into movie data to answer questions about evolution of movie length. [View Code](https://github.com/HantsonAlec/Portfolio/tree/main/Datacamp/Investigating%20Netflix%20Movies%20and%20Guest%20Stars%20in%20The%20Office)

## Google playstore insights

Creating insights into the Google Play Store apps. [View Code](https://github.com/HantsonAlec/Portfolio/tree/main/Datacamp/The%20Android%20App%20Market%20on%20Google%20Play)
