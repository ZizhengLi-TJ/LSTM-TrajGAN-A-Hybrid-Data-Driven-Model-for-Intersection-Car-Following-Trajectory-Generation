# LSTM-TrajGAN-A-Hybrid-Data-Driven-Model-for-Intersection-Car-Following-Trajectory-Generation

## Overview
LSTM-TrajGAN is a hybrid trajectory generation model that combines Long Short-Term Memory (LSTM) networks and Generative Adversarial Networks (GANs) to address the challenge of data gaps in car-following behavior at intersections. By synthesizing realistic vehicle trajectories, LSTM-TrajGAN aims to enhance road traffic management systems' capabilities, especially in scenarios with limited real-world data.

## Motivation
The control potential of road traffic management systems is significantly enriched by continuous trajectory data. However, data gaps arise due to challenges in obtaining comprehensive datasets—particularly in car-following segments—because of low connected vehicle rates and visual occlusions at intersections. LSTM-TrajGAN was developed to address these limitations by generating car-following trajectories that reflect both macroscopic traffic flow consistency and the behavioral uncertainties of drivers, which traditional models often overlook.

## Key Features
Hybrid Model Architecture: Combines LSTM and GAN to capture car-following behaviors under intersection-specific traffic flow characteristics.
Low Data Dependency: Performs well with limited training data, making it ideal for scenarios where extensive datasets are unavailable.
Uncertainty Modeling: Effectively models the uncertainties in car-following behaviors
UAV Data Collection: Utilizes aerial videos captured by Unmanned Aerial Vehicles (UAVs) at intersections for enhanced data gathering and model training.

## File Descriptions
model.py: This is the core modeling framework for LSTM-TrajGAN. It defines the primary architecture for generating car-following trajectories by combining LSTM and GAN models to simulate vehicle-following behavior.

losses.py: Contains the loss functions for the generator and discriminator in LSTM-TrajGAN. These functions guide the training process, helping improve the quality of generated trajectories.

train.py: The training script used to load data and optimize the generator and discriminator during the training phase. Running this file will initiate the full training process for the model.

predict.py: A script for generating predictions on test data. It generates trajectories for new data, allowing observation of the model’s performance on unseen inputs.

evaluate.py: The evaluation script, designed to analyze and assess the effectiveness of the generated car-following trajectories, providing quantitative metrics for model performance.

## Note
This code structure and logic are intended for learning purposes only; training data is not provided.
