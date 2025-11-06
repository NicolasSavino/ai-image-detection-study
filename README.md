# AI Image Detection Study - R Analysis Pipeline

## Project Overview

This project contains the complete R analysis pipeline for the study: **Can people accurately tell apart AI-generated images and real images?**

The study investigates whether participants can distinguish between AI-generated and real photographs at rates better than chance (50%), and explores individual and content differences in detection ability using signal detection theory.

## Repository Structure

```
ai_image_detection_study/
├── ai_image_study.R          # Main analysis script (simulated data)
├── README.md                 # This file
├── data/
│   ├── trials_data.csv       # Trial-level data (output)
│   └── participant_sdt_data.csv  # Participant-level with SDT metrics (output)
├── outputs/
│   ├── plot_1_accuracy_distribution.png
│   ├── plot_2_sdt_space.png
│   ├── plot_3_accuracy_by_major_ai.png
│   └── plot_4_accuracy_by_category.png
└── .gitignore
```

## Research Design

### Main Research Question
Can people accurately tell apart AI-generated images and real images?

### Primary Hypothesis
- **H0**: Participants' accuracy is equal to or lower than chance (≤ 50%)
- **H1**: Participants' accuracy is higher than chance (> 50%)
- **Test**: One-sample t-test comparing mean accuracy to 0.50 (one-sided)

### Secondary Analyses

**Signal Detection Theory Metrics:**
- **Sensitivity (d')**: Ability to discriminate AI from real images
  - H0: mean d' = 0
  - H1: mean d' > 0
  
- **Response Bias (c)**: Tendency to default to "real"
  - H0: mean c = 0
  - H1: mean c > 0

### Exploratory Analyses

1. **Trial-Level Mixed-Effects Logistic Regression**
   - DV: Trial accuracy (correct/incorrect)
   - Fixed effects: image category, confidence, major, AI familiarity
   - Random effect: participant intercept
   - Purpose: Identify which image types/conditions affect accuracy

2. **Participant-Level Linear Regression**
   - Predict d' and c from individual differences
   - Predictors: major (visual/media vs. other), AI tool familiarity, mean confidence
   - Purpose: Identify which participant characteristics predict detection ability

## Study Details

### Participant Data Collected
- Age range
- Gender
- Student status
- Major/field of study
- AI image tool familiarity (DALL-E, Midjourney, other, none)
- Self-rated detection skill (1-6 scale)
- Social media exposure to suspected AI images

### Trial Procedure
- **Image presentation**: Series of images (AI-generated or real photographs)
- **Key constraint**: No human faces in images
- **Ratio**: 1:1 AI to real (addresses limitation from prior work using 3:1)
- **Images per participant**: 24 total (12 AI, 12 real)
- **Image categories**: landscape, protest/crowd, social/domestic, product, nature

### Response Measures
1. **Classification**: "AI-generated" or "Real photograph" (forced choice)
2. **Confidence**: 1-4 Likert scale (1 = not confident, 4 = very confident)
3. **Realism**: 1-4 Likert scale (1 = not realistic, 4 = completely realistic)
4. **Credibility**: 1-4 Likert scale (1 = not believable, 4 = completely believable)

## How to Use This Script

### Prerequisites
- R 3.6+
- Required packages: tidyverse, lme4, broom, knitr

### Installation

```r
# Install required packages (one-time)
install.packages(c("tidyverse", "lme4", "broom", "knitr"))
```

### Running the Analysis

```r
# Source the entire script
source("ai_image_study.R")
```

Or run line-by-line in RStudio/VS Code.

### Output

The script produces:

1. **Console Output**
   - Primary analysis results (accuracy vs. chance)
   - Secondary analysis results (d' and c tests)
   - Mixed-effects model summary
   - Linear regression results
   - Summary statistics by group

2. **Data Files** (saved to working directory)
   - `trials_data.csv`: Full trial-level data
   - `participant_sdt_data.csv`: Participant-level data with SDT metrics

3. **Visualizations** (PNG files)
   - `plot_1_accuracy_distribution.png`: Histogram of accuracy with chance line
   - `plot_2_sdt_space.png`: d' vs c scatter plot colored by major
   - `plot_3_accuracy_by_major_ai.png`: Boxplots by major and AI familiarity
   - `plot_4_accuracy_by_category.png`: Accuracy by image category

## Signal Detection Theory (SDT) Explanation

### Key Concepts

**Sensitivity (d')**: Reflects discriminability—how well a participant can distinguish AI from real images, independent of bias.
- Calculated as: d' = Z(hit rate) - Z(false alarm rate)
- Higher d' = better discrimination
- d' = 0 means performance at chance

**Response Bias (c)**: Reflects the participant's tendency to prefer one response over the other.
- Calculated as: c = -0.5 × [Z(hit rate) + Z(false alarm rate)]
- Positive c = bias toward saying "Real"
- c = 0 means no bias
- Negative c = bias toward saying "AI"

### SDT Contingency Table

|  | Participant says "AI" | Participant says "Real" |
|--|--|--|
| **Image is AI** | Hit | Miss |
| **Image is Real** | False Alarm | Correct Rejection |

## Data Analysis Plan

### Statistical Software
- **R packages used**:
  - `tidyverse`: Data wrangling and visualization
  - `lme4`: Mixed-effects models
  - `broom`: Tidy statistical output
  - `knitr`: Report generation

### Simulated Data

The current script uses **simulated data** for testing and development:
- 100 participants
- 24 trials per participant
- Accuracy simulated at ~60% for AI, ~65% for Real (slightly above chance)
- All demographic and response variables randomized

### When You Have Real Data

1. **Export Qualtrics data as CSV**
2. **Update data import section** (currently uses simulated data)
3. **Ensure column names match** the structure used in the script
4. **Run the complete pipeline**

## Modifications for Your Real Data

When you collect real data through Qualtrics, you'll need to:

1. Export your Qualtrics responses as CSV
2. Replace the data simulation section (lines 50-120) with:
```r
# Load real data
trials <- read.csv("your_qualtrics_data.csv")
participants <- trials %>% distinct(participant_id, age_range, gender, ...)
```

3. Ensure column names in your data match expectations:
   - `participant_id`, `image_type` (AI/Real), `response_correct`, `confidence`, etc.
4. Keep everything else the same—the analysis will work with real data

## Key References

- Velasquez-Salamanca et al. (2024): Prior work showing ~55-65% accuracy in similar tasks and bias toward calling images "real"
- Signal Detection Theory: Green & Swets (1966)

## Contact & Notes

- **Study staff**: Nicolas Savino & S. Emre Kuraner
- **Data privacy**: All data handling complies with HIPAA protocols
- **IRB**: Study approved by IRB; protocols available to study staff and reviewers only

---

**Last Updated**: November 2025
**Status**: Ready for pilot testing with simulated data
