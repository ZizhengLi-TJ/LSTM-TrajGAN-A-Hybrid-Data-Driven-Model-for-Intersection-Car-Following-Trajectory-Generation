# LSTM-TrajGAN-A-Hybrid-Data-Driven-Model-for-Intersection-Car-Following-Trajectory-Generation

## Overview
LSTM-TrajGAN is a hybrid trajectory generation model that combines Long Short-Term Memory (LSTM) networks and Generative Adversarial Networks (GANs) to address the challenge of data gaps in car-following behavior at intersections. By synthesizing realistic vehicle trajectories, LSTM-TrajGAN aims to enhance road traffic management systems' capabilities, especially in scenarios with limited real-world data.

## Motivation
The control potential of road traffic management systems is significantly enriched by continuous trajectory data. However, data gaps arise due to challenges in obtaining comprehensive datasets—particularly in car-following segments—because of low connected vehicle rates and visual occlusions at intersections. LSTM-TrajGAN was developed to address these limitations by generating car-following trajectories that reflect both macroscopic traffic flow consistency and the behavioral uncertainties of drivers, which traditional models often overlook.

## Key Features
Hybrid Model Architecture: Combines LSTM and GAN to capture car-following behaviors under intersection-specific traffic flow characteristics.
Low Data Dependency: Performs well with limited training data, making it ideal for scenarios where extensive datasets are unavailable.
Uncertainty Modeling: Effectively models the uncertainties in car-following behaviors, achieving a high discriminator score of 0.81 and deviations within 15% of real-world trajectories.
UAV Data Collection: Utilizes aerial videos captured by Unmanned Aerial Vehicles (UAVs) at intersections for enhanced data gathering and model training.
