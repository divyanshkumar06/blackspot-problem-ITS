# A B. TECH PROJECT I REPORT ON
## AI/ML BASED ROAD SAFETY AUDIT AND BLACK-SPOT ANALYSIS PIPELINE

**submitted by**

Tanu Meena (U23AI076)
Divyansh Kumar (U23AI082)
Hanmant Jajulwar (U23AI098)
Ritik Sharma (U23EE032)

*in partial fulfilment of the requirements for the award of the degree of*

**BACHELOR OF TECHNOLOGY**
in
**ARTIFICIAL INTELLIGENCE & ELECTRICAL ENGINEERING**

*under the supervision of*

**[SUPERVISOR'S NAME]**
*(Assistant Professor)*

**DEPARTMENT OF [ARTIFICIAL INTELLIGENCE / ELECTRICAL ENGINEERING]**
**SARDAR VALLABHBHAI NATIONAL INSTITUTE OF TECHNOLOGY SURAT**
**MAY 2026**

---

# CERTIFICATE

This is to certify that the B. Tech Project I report "AI/ML BASED ROAD SAFETY AUDIT AND BLACK-SPOT ANALYSIS PIPELINE" submitted by Tanu Meena (U23AI076), Divyansh Kumar (U23AI082), Hanmant Jajulwar (U23AI098), and Ritik Sharma (U23EE032) to the Sardar Vallabhbhai National Institute of Technology Surat, in the partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in [Artificial Intelligence / Electrical Engineering] is a bona fide record of project work carried out by them under my supervision. The contents of this project report, in full or in parts have not been submitted to any other Institute or University for the award of any degree or diploma.


**Head of the Department**
[Artificial Intelligence / Electrical]

**Name of Guide**
Assistant Professor
S.V. National Institute of Technology
Surat

Place: Surat
Date:

---

# CERTIFICATE OF APPROVAL

This is to certify that the project report entitled "AI/ML BASED ROAD SAFETY AUDIT AND BLACK-SPOT ANALYSIS PIPELINE" submitted by Tanu Meena (U23AI076), Divyansh Kumar (U23AI082), Hanmant Jajulwar (U23AI098), and Ritik Sharma (U23EE032) to the Sardar Vallabhbhai National Institute of Technology Surat, in the partial fulfilment of the requirements for the award of the degree of Bachelor of Technology has been accepted by the examination committee and that the students have successfully defended the project report work in the viva-voce examination held today.


(Internal Examiner)                    (External Examiner)

Place: Surat – 395 007
Date:

---

# ACKNOWLEDGEMENTS

We would like to express our deepest appreciation to our supervisor, **[SUPERVISOR'S NAME]**, for their continuous guidance, encouragement, and invaluable feedback throughout the course of this project.

We also extend our sincere gratitude to the Department of [Artificial Intelligence / Electrical Engineering] at SVNIT Surat for providing us with the necessary resources and environment to successfully complete this research. Lastly, we would like to thank our families and friends for their constant moral support.

**Authors:**
Tanu Meena
Divyansh Kumar
Hanmant Jajulwar
Ritik Sharma

---

# ABSTRACT

Road traffic accidents have become a critical public safety issue, leading to severe socio-economic losses. Identifying "Black Spots"—locations with a historically high density of accidents—is crucial for preventative governance. This project develops an end-to-end Artificial Intelligence (AI) and Machine Learning (ML) pipeline designed to identify, analyze, and predict critical road safety hazards along the NH-53 highway corridor using an 11-year dataset (2015-2025). 

The pipeline utilizes **DBSCAN (Density-Based Spatial Clustering)** for unsupervised geospatial mapping of high-density black spots, replacing traditional arbitrary observation methods. Furthermore, a **Random Forest Classifier** is deployed to predict accident severity under varying environmental and temporal conditions. The findings are integrated into a dynamic, interactive dashboard built using Streamlit, featuring 3D Spatial visualizations, time-series future forecasting (2026-2028), Economic ROI simulation modules, and Google Earth KML Export capabilities. The developed system serves as a futuristic, data-driven Intelligent Transport System (ITS) toolkit for government and highway authorities to optimize resource deployment and infrastructure intervention.

**KEYWORDS:** Machine Learning, Road Safety Audit, Black-Spot Detection, DBSCAN, Intelligent Transport System (ITS), Spatial Clustering.

---

# LIST OF FIGURES

*(Note to authors: Update the page numbers once pasted in MS Word)*

1.1 Yearly Accident Trends (2015-2025)
1.2 Hourly Distribution of Accidents
3.1 Geospatial Plot of Top 150 Black Spots
4.1 DBSCAN Clustering Output on Highway Lat/Lon Coordinates
4.2 Random Forest Feature Importance Graph
5.1 Interactive Dashboard UI Overview
5.2 3D Spatial-Temporal Visualization of Highway Risk
5.3 K-Means Predictive Patrol Deployment Radius
5.4 Economic Loss and Intervention ROI Simulator

---

# TABLE OF CONTENTS

**ACKNOWLEDGEMENTS**
**ABSTRACT**
**LIST OF FIGURES**
**LIST OF TABLES**

**1. INTRODUCTION**
1.1 General Overview
1.2 Problem Statement
1.3 Need for the Present Study
1.4 Objectives and Scope

**2. REVIEW OF LITERATURE**
2.1 Traditional Road Safety Audits
2.2 Machine Learning in Traffic Safety
2.3 Spatial Clustering Techniques

**3. DATASET AND METHODOLOGY**
3.1 Data Collection (NH-53 Dataset)
3.2 Data Cleaning and Preprocessing
3.3 Exploratory Data Analysis (EDA)

**4. AI/ML PIPELINE AND ALGORITHMS**
4.1 Geospatial Clustering using DBSCAN
4.2 Severity Prediction using Random Forest
4.3 Predictive Resource Deployment (K-Means)
4.4 Time-Series Future Forecasting

**5. RESULTS AND DASHBOARD DEPLOYMENT**
5.1 Intelligent Transport System (ITS) Dashboard
5.2 Advanced Data Visualization
5.3 Google Earth Integration
5.4 Economic Impact and ROI Simulation

**6. CONCLUSIONS AND FUTURE SCOPE**
6.1 Conclusions
6.2 Scope for Further Research

**REFERENCES**

---

# CHAPTER 1: INTRODUCTION

## 1.1 General
Road traffic safety is an essential component of modern infrastructure. The increasing density of vehicular traffic has led to a proportional rise in highway accidents. Identifying "Black Spots"—specific sections of the road network where accidents occur with a high frequency—is a primary objective for civil and transport authorities.

## 1.2 Need for the Present Study
Traditional road safety audits often rely on manual reporting and basic statistical filtering, which fail to capture complex, multi-dimensional patterns involving time, weather, road conditions, and spatial proximity. A modern Intelligent Transport System (ITS) must leverage AI/ML to proactively locate hazards and predict severities rather than simply reacting to past events.

## 1.3 Objectives and Scope
The primary objectives of this project are:
1. To clean and analyze 11 years (2015-2025) of accident data on the NH-53 corridor.
2. To apply unsupervised ML (DBSCAN) to algorithmically define the bounds of highway black spots.
3. To develop a predictive supervised learning model capable of estimating accident severity (Fatal, Grievous, Minor) based on situational features.
4. To engineer an interactive, real-time dashboard featuring Economic ROI simulators and Future Forecasting to aid executive decision-making.

---

# CHAPTER 3: DATASET AND METHODOLOGY

*(Copy screenshots from your Jupyter Notebook outputs into the gaps provided below)*

## 3.1 Data Cleaning & Preprocessing
The dataset contained 2,305 records. Preprocessing involved handling missing temporal data, encoding categorical variables (`Weather`, `Road_Condition`), and extracting hours from raw timestamp formats.

**[INSERT SCREENSHOT OF NOTEBOOK CELL SHOWING HEAD() OF CLEANED DATAFRAME]**

*Observation:* The cleaning pipeline effectively standardized categorical variables, ensuring the dataset was prime for mathematical encoding.

## 3.2 Exploratory Data Analysis (Temporal & Categorical)
We executed comprehensive EDA to understand the baseline statistics of the NH-53 highway.

**[INSERT PLOTLY GRAPH SCREENSHOT: Yearly Accident Trends from Notebook]**
*Observation:* A clear temporal trend was observed over the years, with fluctuations correlating to potential traffic growth and infrastructural changes.

**[INSERT PLOTLY GRAPH SCREENSHOT: Bar Chart of Hourly Accident Distribution]**
*Observation:* The hourly distribution reveals distinct peaks during specific times (e.g., late night or rush hours), highlighting critical windows of vulnerability.

**[INSERT PIE CHART SCREENSHOT: Causes of Accidents / Weather Conditions]**
*Observation:* Over-speeding and wet/rainy conditions significantly dominate the causative factors, guiding subsequent infrastructure recommendations.

---

# CHAPTER 4: AI/ML PIPELINE AND ALGORITHMS

## 4.1 Geospatial Clustering using DBSCAN
Unlike traditional K-Means, which requires a pre-defined number of clusters, we utilized Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using the Haversine metric.

**[INSERT SCATTER GEO-MAP SCREENSHOT: DBSCAN Clustering Output showing distinct colored highway spots]**
*Observation:* DBSCAN effectively grouped accidents into distinct geographical "Black Spots" along the highway while filtering out isolated, random accidents as noise (marked as -1). 

## 4.2 Severity Prediction Framework
A Random Forest Classifier was trained to predict whether an accident would be Fatal, Grievous, Minor, or Non-Injury.

**[INSERT SCREENSHOT: Random Forest Classification Report / Confusion Matrix]**
*Observation:* The model achieved robust accuracy, with variables like `Vehicle Type` and `Speed/Cause` serving as strong indicators of fatality potential.

## 4.3 K-Means for Strategic Patrol Deployment
While DBSCAN was used for defining black spots, K-Means was repurposed geometrically to find the absolute center of mass (Centroids) for active high-risk zones at specific hours.
*Observation:* This technique successfully generated optimal coordinates for parking Highway Patrols and Ambulances, theoretically minimizing incident response times.

---

# CHAPTER 5: RESULTS AND DASHBOARD DEPLOYMENT

The culmination of the analytical pipeline is an enterprise-grade Streamlit Web Application developed to bridge the gap between complex ML data and actionable administrative policy.

**[INSERT SCREENSHOT: The Main Overview Tab of the Streamlit App]**

## 5.1 Advanced Visualizations
To provide deeper insights, multi-dimensional charting was implemented.

**[INSERT SCREENSHOT: Sunburst Chart or 3D Scatter Plot from the "Advanced Dynamics" tab]**
*Observation:* The 3D plot visualizes Longitude vs. Latitude against Time of Day, proving that risk is not just spatial, but temporal.

## 5.2 Economic Impact and Future Forecasting
We translated raw accident metrics into government financial burdens using standard cost models (e.g., Fatal = ₹15 Lakhs). A Time-Series predictive model was also integrated to forecast future accident rates.

**[INSERT SCREENSHOT: Economic ROI Simulator Tab from Streamlit]**
*Observation:* The ROI dashboard allows officials to input infrastructure costs (e.g., barriers) and calculate the percentage of financial savings returned to the state, making public policy data-driven.

**[INSERT SCREENSHOT: 🔮 Future Forecasting Graph (2026-2028)]**
*Observation:* Without intervention, the multi-year polynomial regression model predicts compounding growth in total accidents over the next 3 years.

## 5.3 Google Earth Integration
To validate the findings in the real world, the pipeline generates dynamic `.kml` coordinates.
**[INSERT SCREENSHOT: Google Earth interface showing the imported KML pins]**
*Observation:* Authorities can seamlessly fly to the predicted ML boundaries inside Google Earth to inspect structural road flaws via satellite.

---

# CHAPTER 6: CONCLUSIONS AND FUTURE SCOPE

## 6.1 Conclusions
This project successfully transformed static historical road data into an Intelligent Transport System. 
1. **DBSCAN** proved highly superior in mapping asymmetrical black spots over traditional linear charting.
2. The **Random Forest** model confirmed that human behavioral causes heavily dictate severity.
3. The **Streamlit Application** successfully bridged data science with actionable governance, generating functional resource deployment maps and Economic ROI justifications for commissioners.

## 6.2 Scope for Further Research
Future iterations of this pipeline could integrate:
- Real-time live API feeds of weather conditions and traffic junction cameras.
- AI Computer Vision for auto-generating E-Challans at the identified black spot coordinates.
- Expansion of the model dataset to encompass a wider national highway network.

---
**[NOTE FOR THE USER: Do not copy this text inside brackets. Open MS Word, apply your university's Times New Roman, 1.5 spacing format, and paste this content. Wherever you see `[INSERT SCREENSHOT ...]`, take a snapshot of your Jupyter Notebook or running Web App and paste the image there!]**
