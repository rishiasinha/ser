# Detecting Emotion From Speech With Machine Learning

## Problem
Recognizing emotion does not just rely on words the person is saying, rather, human beings are capable of recognizing the emotional state of the speaker though attributes such as the tone and pitch of the voice. Pets are also capable of recognizing emotion, even though they do not understand the words that are being spoken. The goal of this project was to develop and train a Machine Learning (ML) model that can detect emotion from audio speech samples.

## Procedure
I implemented an ML model in Python based on a neural network. The model was trained on audio samples from different voice actors.  I leveraged libraries such as librosa, soundfile, and sklearn and built the model using a Multi-Layer Perceptron classifier (MLPClassifier). 

## Data
I used the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset which contains 7,356 files from 24 professional actors (12 female, 12 male), vocalizing emotions calm, happy, sad, angry, fearful, surprise, and disgust. 

## Interpretation
I used the soundfile library to read the audio files and librosa to extract features such as the Mel Frequency Cepstral Coefficient (MFCC), which represents the short-term power spectrum of a sound, and Chroma, pertaining to the 12 different pitch classes. I trained the MLPClassifier on 90% of the dataset by optimizing the log-loss function using stochastic gradient descent. 

## Conclusions
My ML model obtained an accuracy of 80.5% detection rate across the 4 different emotion classes on the test data.
