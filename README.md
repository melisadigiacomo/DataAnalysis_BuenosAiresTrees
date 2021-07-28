# Online Database Analysis: Project Overview 
* Created a tool that downloads databases from a website and have functions to build plots and perform variable comparisson by t-test.
* Downloaded databases using python.
* Developed functions to manipulate the datasets, calculate t-statistic and build plots. 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** urllib.request, pandas, numpy, spicy.stats, matplotlib, seaborn

## Download Databases
Developed a code to download databases from https://data.buenosaires.gob.ar/dataset
* Buenos Aires trees (SIDEWALKS): arbolado-publico-lineal-2017-2018.csv
* Buenos Aires trees (PARKS): arbolado-en-espacios-verdes.csv
 
## Analyzing the data
Seaborn and matplotlib modules were used for graphics. First, the ten most common trees in Buenos Aires parks were investigated. Eucalipto took first place. 

A boxplot of diameters of this species in two different environments (parks and sidewalks) was built. The comparisson of the diameters showed no significant differences between the two environments (t-statistic=0.89, p=0.3731). 

Based on the scientific name of the three most common species of tress in Buenos Aires parks, a pairplot of heights and diameters was built. As for diameter and heigh, Eucalipto was the species with the highest values.
