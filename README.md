# 3D Tree Structure Analysis from LiDAR Point Cloud

This repository contains a Python-based pipeline for processing 3D tree point cloud data (acquired via LiDAR) to analyze tree structure, including stem and crown segmentation, geometric analysis, and volumetric estimation.

## Project Overview

- Segments point cloud data into stem and crown using PCA and vertical binning.
- Performs adaptive binning and RANSAC cylinder fitting for detailed stem analysis.
- Conducts crown structure analysis including canopy layering, crown base height, and individual branch volume/density.
- Provides 3D visualizations and detailed text reports.
- Handles large point cloud datasets using Dask.

## Files

- `task_3.py`: Loads raw point cloud data and performs stem-crown segmentation.
- `stem_analysis.ipynb`: Analyzes stem geometry using adaptive binning and RANSAC.
- `crown_analysis.ipynb`: Analyzes crown structure, including convex hull, alpha shape, and branch analysis.

## Dependencies

Install the required libraries using pip:


## Technologies Used

- Python
- NumPy
- Dask
- Matplotlib
- scikit-learn
- SciPy
- alphashape

## Outputs

- `stem_points.txt`, `crown_points.txt`: Segmented point cloud data
- `adaptive_stem_analysis1.txt`: Stem geometry report
- 3D plots and visualizations of tree structure


