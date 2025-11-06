# ============================================================================
# AI Image Detection Study - Complete Analysis Pipeline
# Research Question: Can people accurately tell apart AI-generated and real images?
# ============================================================================

# Clear environment
rm(list = ls())

# ============================================================================
# 1. LOAD PACKAGES
# ============================================================================

# Install packages if needed (uncomment to run)
# install.packages(c("tidyverse", "lme4", "psycho", "broom", "knitr"))

library(tidyverse)      # Data wrangling and visualization
library(lme4)           # Mixed-effects models
library(broom)          # Tidy model outputs
library(knitr)          # Report generation

# ============================================================================
# 2. SET SEED FOR REPRODUCIBILITY
# ============================================================================

set.seed(42)

# ============================================================================
# 3. CREATE SIMULATED DATA
# ============================================================================

# Parameters for simulation
n_participants <- 100
n_trials_per_participant <- 24  # 12 AI, 12 real (1:1 ratio)
image_categories <- c("landscape", "protest_crowd", "social_domestic", "product", "nature")

# Create participant-level data
participants <- tibble(
  participant_id = 1:n_participants,
  age_range = sample(c("18-25", "26-35", "36-45", "46+"), n_participants, replace = TRUE),
  gender = sample(c("Male", "Female", "Non-Binary", "Other"), n_participants, replace = TRUE),
  student_status = sample(c("Current student", "Graduated", "Never attended"), n_participants, replace = TRUE),
  major = sample(c("Visual/Media", "Other"), n_participants, replace = TRUE, prob = c(0.3, 0.7)),
  ai_familiarity = sample(c("Yes - DALL-E", "Yes - Midjourney", "Yes - Other", "No"), n_participants, replace = TRUE),
  self_rated_detection = sample(1:6, n_participants, replace = TRUE),
  social_media_exposure = sample(c("Never", "Rarely", "Sometimes", "Often", "Very Often"), n_participants, replace = TRUE)
)

# Create trial-level data
set.seed(42)
trials <- expand_grid(
  participant_id = 1:n_participants,
  trial_num = 1:n_trials_per_participant
) %>%
  mutate(
    # 1:1 ratio of AI to real images
    image_type = ifelse(trial_num <= 12, "AI", "Real"),
    image_category = sample(image_categories, n(), replace = TRUE),
    # Simulate accuracy: people are slightly better than chance, with more errors on AI
    true_accuracy = ifelse(image_type == "AI", 0.60, 0.65),
    # Generate response (1 = correct, 0 = incorrect)
    response_correct = rbinom(n(), 1, true_accuracy),
    # Confidence in response (1-4 scale)
    confidence = sample(1:4, n(), replace = TRUE),
    # Realism rating (1-4 scale)
    realism = sample(1:4, n(), replace = TRUE),
    # Credibility rating (1-4 scale)
    credibility = sample(1:4, n(), replace = TRUE),
    # What they guessed (AI or Real) - correlated with accuracy
    participant_guess = ifelse(response_correct == 1, 
                               image_type, 
                               ifelse(image_type == "AI", "Real", "AI")),
    # Convert to binary for signal detection (1 = said "AI", 0 = said "Real")
    response_ai = ifelse(participant_guess == "AI", 1, 0)
  ) %>%
  left_join(participants, by = "participant_id")

# ============================================================================
# 4. ORGANIZE DATA BY IMAGE TYPE FOR SIGNAL DETECTION
# ============================================================================

# For signal detection theory, we need:
# - Hits: correctly identified AI as AI
# - False Alarms: incorrectly identified Real as AI
# - Misses: incorrectly identified AI as Real
# - Correct Rejections: correctly identified Real as Real

signal_detection_data <- trials %>%
  mutate(
    hit = ifelse(image_type == "AI" & response_ai == 1, 1, 0),
    false_alarm = ifelse(image_type == "Real" & response_ai == 1, 1, 0),
    miss = ifelse(image_type == "AI" & response_ai == 0, 1, 0),
    correct_rejection = ifelse(image_type == "Real" & response_ai == 0, 1, 0)
  )

# Calculate participant-level signal detection metrics
participant_sdt <- signal_detection_data %>%
  group_by(participant_id) %>%
  summarise(
    n_trials = n(),
    # Count hits, misses, false alarms, correct rejections
    hits = sum(hit),
    misses = sum(miss),
    false_alarms = sum(false_alarm),
    correct_rejections = sum(correct_rejection),
    # Hit rate and false alarm rate
    hit_rate = hits / (hits + misses),
    fa_rate = false_alarms / (false_alarms + correct_rejections),
    .groups = "drop"
  ) %>%
  mutate(
    # Calculate d' (sensitivity)
    # Using the formula: d' = z(hit_rate) - z(fa_rate)
    # Add small constant to avoid infinite values
    hit_rate_adj = pmax(pmin(hit_rate, 0.99), 0.01),
    fa_rate_adj = pmax(pmin(fa_rate, 0.99), 0.01),
    d_prime = qnorm(hit_rate_adj) - qnorm(fa_rate_adj),
    # Calculate c (bias/criterion)
    # c = -0.5 * (z(hit_rate) + z(fa_rate))
    c = -0.5 * (qnorm(hit_rate_adj) + qnorm(fa_rate_adj)),
    # Overall accuracy across all trials
    accuracy = (hits + correct_rejections) / n_trials
  ) %>%
  select(participant_id, n_trials, hits, misses, false_alarms, correct_rejections, 
         hit_rate, fa_rate, d_prime, c, accuracy) %>%
  left_join(participants, by = "participant_id")

# ============================================================================
# 5. PRIMARY ANALYSIS: One-Sample t-test (Accuracy vs. Chance)
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("PRIMARY ANALYSIS: Accuracy vs. Chance (50%)\n")
cat("=" %*% 80, "\n")

# Test if mean accuracy is significantly above 0.50
primary_test <- t.test(
  participant_sdt$accuracy,
  mu = 0.50,
  alternative = "greater",
  paired = FALSE
)

cat("\nOne-sample t-test: Mean accuracy vs. 0.50\n")
cat("Mean accuracy: ", round(mean(participant_sdt$accuracy), 3), "\n")
cat("SD: ", round(sd(participant_sdt$accuracy), 3), "\n")
cat("t-statistic: ", round(primary_test$statistic, 3), "\n")
cat("p-value: ", round(primary_test$p.value, 4), "\n")
cat("95% CI: [", round(primary_test$conf.int[1], 3), ", ", round(primary_test$conf.int[2], 3), "]\n")

if (primary_test$p.value < 0.05) {
  cat("\nConclusion: REJECT H0. Participants' accuracy is significantly above chance.\n")
} else {
  cat("\nConclusion: FAIL TO REJECT H0. Participants' accuracy is not significantly above chance.\n")
}

# ============================================================================
# 6. SECONDARY ANALYSIS: Signal Detection Theory
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("SECONDARY ANALYSIS: Signal Detection Theory (d' and c)\n")
cat("=" %*% 80, "\n")

# Test d' vs. 0 (sensitivity)
dprime_test <- t.test(
  participant_sdt$d_prime,
  mu = 0,
  alternative = "greater",
  paired = FALSE
)

cat("\nOne-sample t-test: Mean d' vs. 0\n")
cat("Mean d': ", round(mean(participant_sdt$d_prime), 3), "\n")
cat("SD: ", round(sd(participant_sdt$d_prime), 3), "\n")
cat("t-statistic: ", round(dprime_test$statistic, 3), "\n")
cat("p-value: ", round(dprime_test$p.value, 4), "\n")

if (dprime_test$p.value < 0.05) {
  cat("Conclusion: REJECT H0. Sensitivity (d') is significantly above 0.\n")
} else {
  cat("Conclusion: FAIL TO REJECT H0. Sensitivity (d') is not significantly above 0.\n")
}

# Test c vs. 0 (response bias)
c_test <- t.test(
  participant_sdt$c,
  mu = 0,
  alternative = "greater",
  paired = FALSE
)

cat("\nOne-sample t-test: Mean c vs. 0\n")
cat("Mean c: ", round(mean(participant_sdt$c), 3), "\n")
cat("SD: ", round(sd(participant_sdt$c), 3), "\n")
cat("t-statistic: ", round(c_test$statistic, 3), "\n")
cat("p-value: ", round(c_test$p.value, 4), "\n")

if (c_test$p.value < 0.05) {
  cat("Conclusion: REJECT H0. Response bias (c) is significantly positive.\n")
  cat("            (Participants have a bias to call images 'real')\n")
} else {
  cat("Conclusion: FAIL TO REJECT H0. No significant response bias detected.\n")
}

# ============================================================================
# 7. EXPLORATORY ANALYSIS: Trial-Level Mixed-Effects Logistic Regression
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("EXPLORATORY ANALYSIS: Trial-Level Mixed-Effects Logistic Regression\n")
cat("=" %*% 80, "\n")

# Prepare data for logistic regression
model_data <- trials %>%
  mutate(
    major_binary = ifelse(major == "Visual/Media", 1, 0),
    ai_familiarity_binary = ifelse(ai_familiarity %in% c("Yes - DALL-E", "Yes - Midjourney", "Yes - Other"), 1, 0),
    image_category = as.factor(image_category)
  )

# Fit mixed-effects logistic regression
# DV: response_correct (1 = correct, 0 = incorrect)
# Fixed effects: image_category, confidence, major_binary, ai_familiarity_binary
# Random effect: random intercept for participant

trial_model <- glmer(
  response_correct ~ image_category + scale(confidence) + major_binary + ai_familiarity_binary + (1 | participant_id),
  data = model_data,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)

cat("\nMixed-Effects Logistic Regression Results:\n")
cat("DV: Trial-level accuracy (1 = correct, 0 = incorrect)\n\n")
print(summary(trial_model))

cat("\n\nInterpretation of Fixed Effects (log-odds scale):\n")
fixed_effects <- fixef(trial_model)
cat("Intercept: ", round(fixed_effects[1], 3), "\n")
cat("Image category and other predictors follow above\n")

# ============================================================================
# 8. EXPLORATORY ANALYSIS: Participant-Level Linear Regression
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("EXPLORATORY ANALYSIS: Participant-Level Linear Regression\n")
cat("=" %*% 80, "\n")

# Prepare participant-level data
participant_model_data <- participant_sdt %>%
  mutate(
    major_binary = ifelse(major == "Visual/Media", 1, 0),
    ai_familiarity_binary = ifelse(ai_familiarity %in% c("Yes - DALL-E", "Yes - Midjourney", "Yes - Other"), 1, 0),
    mean_confidence = rowMeans(
      trials %>%
        filter(participant_id %in% participant_sdt$participant_id) %>%
        select(confidence) %>%
        as.matrix(), 
      na.rm = TRUE
    )
  )

# Recalculate mean confidence properly
mean_confidence <- trials %>%
  group_by(participant_id) %>%
  summarise(mean_confidence = mean(confidence), .groups = "drop")

participant_model_data <- participant_model_data %>%
  left_join(mean_confidence, by = "participant_id")

# Model 1: Predicting d' (sensitivity)
dprime_model <- lm(
  d_prime ~ major_binary + ai_familiarity_binary + mean_confidence,
  data = participant_model_data
)

cat("\nLinear Regression: Predicting d' (Sensitivity)\n")
print(summary(dprime_model))

# Model 2: Predicting c (response bias)
c_model <- lm(
  c ~ major_binary + ai_familiarity_binary + mean_confidence,
  data = participant_model_data
)

cat("\n\nLinear Regression: Predicting c (Response Bias)\n")
print(summary(c_model))

# ============================================================================
# 9. SUMMARY STATISTICS AND VISUALIZATION
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("SUMMARY STATISTICS\n")
cat("=" %*% 80, "\n")

cat("\nOverall Performance:\n")
cat("Mean accuracy: ", round(mean(participant_sdt$accuracy), 3), "\n")
cat("Mean d': ", round(mean(participant_sdt$d_prime), 3), "\n")
cat("Mean c: ", round(mean(participant_sdt$c), 3), "\n")
cat("Mean hit rate: ", round(mean(participant_sdt$hit_rate), 3), "\n")
cat("Mean false alarm rate: ", round(mean(participant_sdt$fa_rate), 3), "\n")

cat("\n\nPerformance by Major:\n")
participant_sdt %>%
  group_by(major) %>%
  summarise(
    n = n(),
    mean_accuracy = mean(accuracy),
    sd_accuracy = sd(accuracy),
    mean_dprime = mean(d_prime),
    mean_c = mean(c),
    .groups = "drop"
  ) %>%
  print()

cat("\n\nPerformance by AI Familiarity:\n")
participant_sdt %>%
  group_by(ai_familiarity) %>%
  summarise(
    n = n(),
    mean_accuracy = mean(accuracy),
    sd_accuracy = sd(accuracy),
    mean_dprime = mean(d_prime),
    mean_c = mean(c),
    .groups = "drop"
  ) %>%
  print()

# ============================================================================
# 10. CREATE VISUALIZATIONS
# ============================================================================

# Plot 1: Distribution of accuracy
p1 <- ggplot(participant_sdt, aes(x = accuracy)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", alpha = 0.7, color = "black") +
  geom_vline(xintercept = 0.50, linetype = "dashed", color = "red", linewidth = 1) +
  geom_vline(xintercept = mean(participant_sdt$accuracy), color = "darkgreen", linewidth = 1) +
  labs(
    title = "Distribution of Participant Accuracy",
    x = "Accuracy (Proportion Correct)",
    y = "Number of Participants",
    subtitle = "Red line = Chance (0.50), Green line = Mean accuracy"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 2: d' vs c scatter
p2 <- ggplot(participant_sdt, aes(x = d_prime, y = c, color = major)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "Signal Detection Space: Sensitivity (d') vs Response Bias (c)",
    x = "d' (Sensitivity: higher = better discrimination)",
    y = "c (Response Bias: positive = bias toward 'Real')",
    color = "Major"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 3: Accuracy by major and AI familiarity
p3 <- trials %>%
  group_by(participant_id, major, ai_familiarity) %>%
  summarise(accuracy = mean(response_correct), .groups = "drop") %>%
  ggplot(aes(x = major, y = accuracy, fill = ai_familiarity)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 0.50, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Accuracy by Major and AI Tool Familiarity",
    x = "Major",
    y = "Accuracy",
    fill = "AI Familiarity"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 4: Accuracy by image category
p4 <- trials %>%
  ggplot(aes(x = image_category, y = response_correct, fill = image_type)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1) +
  labs(
    title = "Accuracy by Image Category and Type",
    x = "Image Category",
    y = "Trial Accuracy (1 = Correct, 0 = Incorrect)",
    fill = "Image Type"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save plots
ggsave("plot_1_accuracy_distribution.png", p1, width = 8, height = 6)
ggsave("plot_2_sdt_space.png", p2, width = 8, height = 6)
ggsave("plot_3_accuracy_by_major_ai.png", p3, width = 10, height = 6)
ggsave("plot_4_accuracy_by_category.png", p4, width = 10, height = 6)

cat("\n\nVisualizations saved:\n")
cat("- plot_1_accuracy_distribution.png\n")
cat("- plot_2_sdt_space.png\n")
cat("- plot_3_accuracy_by_major_ai.png\n")
cat("- plot_4_accuracy_by_category.png\n")

# ============================================================================
# 11. EXPORT RESULTS
# ============================================================================

cat("\n")
cat("=" %*% 80, "\n")
cat("EXPORTING DATA AND RESULTS\n")
cat("=" %*% 80, "\n")

# Save processed data
write.csv(trials, "trials_data.csv", row.names = FALSE)
write.csv(participant_sdt, "participant_sdt_data.csv", row.names = FALSE)

cat("\nData files saved:\n")
cat("- trials_data.csv (trial-level data)\n")
cat("- participant_sdt_data.csv (participant-level with SDT metrics)\n")

cat("\n" %*% 80, "\n")
cat("ANALYSIS COMPLETE\n")
cat("=" %*% 80, "\n")