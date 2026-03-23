# 🛣️ AI/ML Black Spot Analysis Pipeline
### Intelligent Transport Systems Project
**Dataset:** Monthly Accident Reports 2015–2025 | **Route:** NH-53 (Old NH-6) Surat–Hazira Tollway

## Overview
This repository contains an end-to-end AI/ML pipeline for road accident black spot analysis. Using 11 years (2015-2025) of accident data from the NH-53 corridor, this project identifies dangerous road segments, analyzes contributing factors, and predicts accident severity using Machine Learning.

## Features & Analyses
The Jupyter Notebook (`blackspot_analysis.ipynb`) includes 19 distinct analysis sections:

1. **Exploratory Data Analysis (EDA):** Yearly trends, monthly heatmaps, time-of-day distributions.
2. **Interactive Geospatial Heatmap:** Folium-based map rendering accident hotspots along the highway corridor.
3. **Temporal Analysis:** Night vs. Day comparison, Weather × Time interaction, and moving average trend analysis.
4. **Behavioral & Structural Insights:** Accident causes, vehicle types involved, and long-term persistent black spots.
5. **Chainage Density:** Accident frequency and severity aggregated by 5-kilometer chainage bins.
6. **ML Clustering (DBSCAN & KMeans):** Spatial clustering to algorithmically identify black spot zones, comparing density-based and centroid-based approaches.
7. **Severity Prediction (Random Forest & XGBoost):** Classification models trained to predict whether an accident will be Fatal, Grievous, Minor, or Non-Injury based on environmental, temporal, and road features.
8. **Risk Scoring:** A composite Risk Rank table sorting the top 20 most dangerous locations by factoring in both frequency and severity.

## Requirements
To run this notebook, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
Launch the Jupyter Notebook:
```bash
jupyter notebook blackspot_analysis.ipynb
```
Run **Kernel → Restart & Run All** to execute the pipeline. 
- *Note:* The interactive `folium` heatmap will be exported as `blackspot_heatmap.html` in the working directory. Plot images will also be saved automatically.

## Project Structure
- `blackspot_analysis.ipynb`: The main AI/ML pipeline notebook.
- `Monthly Reports updated till May 2025 NH-53.xlsx`: The 2,305-record accident dataset.
- `requirements.txt`: Python package dependencies.

## Key Findings & Recommendations
- **Over-speeding** accounts for over 50% of all recorded accidents.
- The **06:00–09:00** and **17:00–20:00** windows are the peak high-risk periods.
- Although nighttime accidents are less frequent, their **fatality rate is significantly higher**.
- **Heavy vehicles and motorcycles** are the most frequently involved, with pedestrians and cyclists suffering the highest average severity when involved.
- Recommendations include targeted speed enforcement, temporal patrolling during rush hours, installing rumble strips/barriers at identified DBSCAN clusters, and strict helmet/speed-limiter enforcement.
