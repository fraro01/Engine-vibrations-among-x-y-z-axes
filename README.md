# Engine-vibrations-among-x-y-z-axes

Dashboard for **Advanced Methods of Monitoring and Design of Systems** 

This file is an analysis for engine fault detection based on a *synthetic dataset* designed for engine failure in mechanical systems, particularly in the automotive industry.\
It includes various sensor readings and operational parameters collected during both normal and faulty engine states. The data is simulated to reflect different fault conditions ranging from normal operation to severe engine faults.\
[Dataset source](https://www.kaggle.com/datasets/ziya07/engine-failure-detection-dataset/data)

This repository contains two main elements:

- **Jupyter Notebook:** contains the data analysis of the data collected from Kaggle, with explanations and details of all the statistics behind them.
- **Dashboard code:** code for the dashboard synchronized on [Render](https://engine-vibrations-among-xyz-axes.onrender.com), it is a dashboard that shows the relations between the vibrations among the 3 axes of the engine with the fault status, based on the **Root Mean Square** and the **Fast Fourier Transform**, with thresholds and baselines added in Frequency Spectral representation.

![Engine axis](engine_axi.JPG "Engine axis (x,y,z)")

