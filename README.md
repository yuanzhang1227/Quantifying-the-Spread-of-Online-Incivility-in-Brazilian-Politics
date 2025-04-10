# Replication Code for "Quantifying the Spread of Online Incivility in Brazilian Politics"

This repository lists all the Python and R scripts used for replicating the results in the paper:  
**Quantifying the Spread of Online Incivility in Brazilian Politics**

---

## 1. Datasets

Due to ethical considerations, the original datasets cannot be publicly shared. However, we provide detailed descriptions of the dataset structures to support understanding and replication efforts.

### File Descriptions

#### `followship.csv`
Network data mapping connections between survey participants and political influencers:
- `Source`: Twitter/X handles of survey respondents
- `Target`: Twitter/X handles of followed political influencers

#### `influencer_annotation.csv`
Annotated profiles of Brazilian political influencers during the 2022 presidential election, including:
- `Username`: Account handle
- `Description`: Profile biography text
- `Account type`: Manual labels (Politician/Media/Individual)  
- `Political ideology`: Manual labels (Left/Right/Center)  
- `Personal support`: Manual labels (Lula camp/Bolsonaro camp)  
- `Social identity`: Manual labels (Religious/Women/LGBTQ/Black) 

#### `Influencer_Incivility_Predicted_timestamped.csv`
Classified Twitter/X posts from Brazilian political influencers during the 2022 election, with automated incivility detection across four dimensions (IMP, PHAVPR, HSST, THREAT):
- Post metadata: `User_ID`, `Username`, `Text`, `Published_time`
- `datetime_formatted`: YY-MM-DD timestamp
- Classification results (per dimension):
  - `labels_[DIMENSION]`: Binary prediction (0=civil, 1=uncivil)
  - `probas_class_0_[DIMENSION]`: Civil probability score
  - `probas_class_1_[DIMENSION]`: Uncivil probability score

#### Filtered Subsets
Posts labeled as 1 with a predicted probability exceeding the 0.7 threshold for each incivility dimension:
- `IMP_Incivility_Predicted_timestamped.csv`
- `PHAVPR_Incivility_Predicted_timestamped.csv` 
- `HSST_Incivility_Predicted_timestamped.csv`
- `THREAT_Incivility_Predicted_timestamped.csv`

---

## 2. Classifiers

The machine learning classifiers used in this study are available in a separate repository:  
[Multidimensional Political Incivility Detection](https://github.com/yuanzhang1227/multidimensional_political_incivility_detection)
