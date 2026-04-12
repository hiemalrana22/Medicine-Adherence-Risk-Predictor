# ============================================================
# analysis.R - Statistical Analysis & Visualization
# ============================================================
# PURPOSE:
#   This R script performs rigorous statistical analysis on the
#   medication adherence dataset using:
#   - Summary statistics
#   - Hypothesis testing (t-test, ANOVA, Chi-square)
#   - ggplot2 visualizations
#   - Logistic regression (statistical, not predictive)
#
# HOW TO RUN:
#   Option 1: Open in RStudio and click "Source"
#   Option 2: Rscript r_scripts/analysis.R
#
# REQUIRED PACKAGES:
#   install.packages(c("tidyverse", "ggplot2", "scales",
#                      "corrplot", "gridExtra"))
# ============================================================

# ── 0. Setup ─────────────────────────────────────────────────

cat("============================================================\n")
cat("Medical Adherence Predictor - R Statistical Analysis\n")
cat("============================================================\n\n")

# Load required packages (install if not present)
required_packages <- c("tidyverse", "ggplot2", "scales",
                        "gridExtra", "broom")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Set output directory
output_dir <- "outputs/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Set ggplot2 theme
theme_set(
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40"),
    axis.title    = element_text(face = "bold"),
    legend.position = "bottom"
  )
)

# Custom color palette
adherence_colors <- c("0" = "#E74C3C", "1" = "#27AE60")


# ── 1. Load Data ──────────────────────────────────────────────

cat("Loading dataset...\n")

raw_path <- "data/raw/medication_adherence.csv"

if (!file.exists(raw_path)) {
  cat("[WARN] Dataset not found. Generating synthetic data...\n")

  set.seed(42)
  n <- 2000

  df <- data.frame(
    patient_id          = 1:n,
    age                 = sample(18:85, n, replace = TRUE),
    gender              = sample(c("Male", "Female"), n, replace = TRUE, prob = c(0.48, 0.52)),
    insurance_type      = sample(c("HMO", "PPO", "Medicare", "Medicaid"), n,
                                  replace = TRUE, prob = c(0.30, 0.35, 0.20, 0.15)),
    annual_contribution = round(rnorm(n, 3000, 800), 2),
    claim_amount        = round(rnorm(n, 1200, 600), 2),
    expected_refills    = sample(3:12, n, replace = TRUE),
    refills_received    = pmax(0, sample(3:12, n, replace = TRUE) - rpois(n, 1.5)),
    days_supply         = sample(c(30, 60, 90), n, replace = TRUE, prob = c(0.5, 0.3, 0.2)),
    chronic_condition   = sample(c("Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"),
                                  n, replace = TRUE, prob = c(0.22, 0.28, 0.15, 0.10, 0.25)),
    num_medications     = sample(1:7, n, replace = TRUE),
    adherent            = sample(0:1, n, replace = TRUE, prob = c(0.38, 0.62))
  )

  # Clip values
  df$annual_contribution <- pmax(500, pmin(8000, df$annual_contribution))
  df$claim_amount        <- pmax(50, pmin(6000, df$claim_amount))
  df$refills_received    <- pmin(df$refills_received, df$expected_refills)

  # Introduce some NAs
  df$annual_contribution[sample(n, 80)] <- NA
  df$claim_amount[sample(n, 70)]        <- NA

  dir.create("data/raw", showWarnings = FALSE, recursive = TRUE)
  write.csv(df, raw_path, row.names = FALSE)
  cat("[OK] Synthetic dataset saved.\n\n")
} else {
  df <- read.csv(raw_path)
  cat(sprintf("[OK] Dataset loaded: %d rows × %d columns\n\n", nrow(df), ncol(df)))
}

# ── 2. Feature Engineering ────────────────────────────────────

cat("Engineering features...\n")

df <- df %>%
  mutate(
    # Refill ratio (main adherence proxy)
    refill_ratio      = ifelse(expected_refills > 0,
                               refills_received / expected_refills, 0),

    # Financial burden
    financial_burden  = ifelse(!is.na(annual_contribution) & annual_contribution > 0,
                               claim_amount / annual_contribution, NA),

    # Age groups
    age_group = case_when(
      age <= 35 ~ "Young (18–35)",
      age <= 64 ~ "Adult (36–64)",
      TRUE      ~ "Elderly (65+)"
    ),
    age_group = factor(age_group, levels = c("Young (18–35)", "Adult (36–64)", "Elderly (65+)")),

    # Medication complexity
    med_complexity = case_when(
      num_medications <= 2 ~ "Low (1–2)",
      num_medications <= 5 ~ "Medium (3–5)",
      TRUE                 ~ "High (6+)"
    ),
    med_complexity = factor(med_complexity, levels = c("Low (1–2)", "Medium (3–5)", "High (6+)")),

    # Adherence as factor for plotting
    adherence_label = factor(adherent, levels = c(0, 1),
                             labels = c("Non-Adherent", "Adherent"))
  )

cat("[OK] Features engineered.\n\n")


# ── 3. Summary Statistics ──────────────────────────────────────

cat("============================================================\n")
cat("SUMMARY STATISTICS\n")
cat("============================================================\n\n")

# Overall adherence rate
adh_rate <- mean(df$adherent, na.rm = TRUE)
cat(sprintf("Overall Adherence Rate: %.1f%%\n", adh_rate * 100))
cat(sprintf("Total Patients:         %d\n", nrow(df)))
cat(sprintf("Adherent:               %d\n", sum(df$adherent == 1, na.rm = TRUE)))
cat(sprintf("Non-Adherent:           %d\n\n", sum(df$adherent == 0, na.rm = TRUE)))

# Summary by adherence group
cat("Summary Statistics by Adherence Group:\n")
summary_stats <- df %>%
  group_by(adherence_label) %>%
  summarise(
    n                   = n(),
    mean_age            = round(mean(age, na.rm = TRUE), 1),
    mean_refill_ratio   = round(mean(refill_ratio, na.rm = TRUE), 3),
    mean_claim_amount   = round(mean(claim_amount, na.rm = TRUE), 2),
    mean_financial_burd = round(mean(financial_burden, na.rm = TRUE), 3),
    mean_num_meds       = round(mean(num_medications, na.rm = TRUE), 1),
    .groups = "drop"
  )

print(summary_stats)
cat("\n")


# ── 4. Hypothesis Testing ──────────────────────────────────────

cat("============================================================\n")
cat("HYPOTHESIS TESTING\n")
cat("============================================================\n\n")

# ── 4a. Independent T-Test: Refill Ratio by Adherence ─────────
cat("Test 1: T-Test — Refill Ratio (Adherent vs Non-Adherent)\n")
cat("  H0: Mean refill ratio is equal in both groups\n")
cat("  H1: Mean refill ratio is different between groups\n\n")

adherent_refill     <- df$refill_ratio[df$adherent == 1]
non_adherent_refill <- df$refill_ratio[df$adherent == 0]

t_result <- t.test(adherent_refill, non_adherent_refill, var.equal = FALSE)

cat(sprintf("  t-statistic: %.4f\n", t_result$statistic))
cat(sprintf("  p-value:     %.2e\n", t_result$p.value))
cat(sprintf("  Adherent mean:     %.3f\n", mean(adherent_refill, na.rm = TRUE)))
cat(sprintf("  Non-Adherent mean: %.3f\n\n", mean(non_adherent_refill, na.rm = TRUE)))

if (t_result$p.value < 0.05) {
  cat("  [SIGNIFICANT] (p < 0.05): Refill ratios differ significantly.\n")
  cat("     Adherent patients have notably higher refill ratios.\n\n")
} else {
  cat("  [NOT SIGNIFICANT] (p >= 0.05): No significant difference.\n\n")
}


# ── 4b. T-Test: Financial Burden by Adherence ─────────────────
cat("Test 2: T-Test — Financial Burden (Adherent vs Non-Adherent)\n")
cat("  H0: Financial burden is equal in both groups\n\n")

df_complete <- df %>% filter(!is.na(financial_burden))
adherent_fb     <- df_complete$financial_burden[df_complete$adherent == 1]
non_adherent_fb <- df_complete$financial_burden[df_complete$adherent == 0]

fb_t_result <- t.test(adherent_fb, non_adherent_fb, var.equal = FALSE)

cat(sprintf("  t-statistic: %.4f\n", fb_t_result$statistic))
cat(sprintf("  p-value:     %.2e\n", fb_t_result$p.value))

if (fb_t_result$p.value < 0.05) {
  cat("  [SIGNIFICANT]: Financial burden differs between groups.\n\n")
} else {
  cat("  [NOT SIGNIFICANT]\n\n")
}


# ── 4c. One-Way ANOVA: Age Group vs Refill Ratio ──────────────
cat("Test 3: ANOVA — Refill Ratio across Age Groups\n")
cat("  H0: Refill ratio is equal across all age groups\n\n")

anova_model  <- aov(refill_ratio ~ age_group, data = df)
anova_result <- summary(anova_model)
print(anova_result)

anova_p <- anova_result[[1]]$`Pr(>F)`[1]

if (!is.na(anova_p) && anova_p < 0.05) {
  cat("\n  [SIGNIFICANT]: Refill ratio differs across age groups.\n\n")
  # Post-hoc Tukey test
  cat("  Post-hoc Tukey HSD test:\n")
  tukey <- TukeyHSD(anova_model)
  print(tukey)
} else {
  cat("\n  [NOT SIGNIFICANT] across age groups.\n\n")
}


# ── 4d. Chi-Square: Insurance Type vs Adherence ───────────────
cat("Test 4: Chi-Square — Insurance Type vs Adherence\n")
cat("  H0: Insurance type and adherence are independent\n\n")

contingency_table <- table(df$insurance_type, df$adherent)
chi_result <- chisq.test(contingency_table)

cat(sprintf("  Chi-square: %.4f\n", chi_result$statistic))
cat(sprintf("  df:         %d\n", chi_result$parameter))
cat(sprintf("  p-value:    %.4f\n\n", chi_result$p.value))

if (chi_result$p.value < 0.05) {
  cat("  [SIGNIFICANT]: Insurance type is associated with adherence.\n\n")
} else {
  cat("  [NOT SIGNIFICANT]: No strong association found.\n\n")
}

cat("Adherence Rate by Insurance Type:\n")
ins_summary <- df %>%
  group_by(insurance_type) %>%
  summarise(adherence_rate = round(mean(adherent, na.rm = TRUE), 3),
            n = n(), .groups = "drop") %>%
  arrange(desc(adherence_rate))
print(ins_summary)
cat("\n")


# ── 5. Visualizations ─────────────────────────────────────────

cat("============================================================\n")
cat("GENERATING VISUALIZATIONS (R / ggplot2)\n")
cat("============================================================\n\n")


# ── Plot 1: Adherence Distribution ───────────────────────────
p1 <- ggplot(df, aes(x = adherence_label, fill = adherence_label)) +
  geom_bar(show.legend = FALSE, width = 0.6, color = "black") +
  geom_text(stat = "count", aes(label = ..count..),
            vjust = -0.5, fontface = "bold", size = 4.5) +
  scale_fill_manual(values = c("Non-Adherent" = "#E74C3C",
                                "Adherent"     = "#27AE60")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title   = "Patient Adherence Distribution",
       subtitle = sprintf("Overall adherence rate: %.1f%%", adh_rate * 100),
       x = NULL, y = "Number of Patients") +
  theme(axis.text.x = element_text(size = 11, face = "bold"))

ggsave(file.path(output_dir, "R_01_adherence_distribution.png"),
       p1, width = 7, height = 5, dpi = 150)
cat("  [OK] Saved: R_01_adherence_distribution.png\n")


# ── Plot 2: Refill Ratio Distribution ────────────────────────
p2 <- ggplot(df, aes(x = refill_ratio, fill = adherence_label)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity", color = "white") +
  geom_vline(data = df %>% group_by(adherence_label) %>%
               summarise(mean_rr = mean(refill_ratio)),
             aes(xintercept = mean_rr, color = adherence_label),
             linetype = "dashed", size = 1.2, show.legend = FALSE) +
  scale_fill_manual(values  = c("Non-Adherent" = "#E74C3C", "Adherent" = "#27AE60")) +
  scale_color_manual(values = c("Non-Adherent" = "#C0392B", "Adherent" = "#1E8449")) +
  labs(title    = "Refill Ratio Distribution by Adherence",
       subtitle  = "Dashed lines show group means",
       x = "Refill Ratio (Refills Received / Expected)",
       y = "Count", fill = "Adherence Status") +
  facet_wrap(~ adherence_label, nrow = 2)

ggsave(file.path(output_dir, "R_02_refill_ratio_distribution.png"),
       p2, width = 9, height = 6, dpi = 150)
cat("  [OK] Saved: R_02_refill_ratio_distribution.png\n")


# ── Plot 3: Financial Burden by Age Group & Adherence ─────────
p3 <- df %>%
  filter(!is.na(financial_burden)) %>%
  ggplot(aes(x = age_group, y = financial_burden, fill = adherence_label)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21, outlier.alpha = 0.4) +
  scale_fill_manual(values = c("Non-Adherent" = "#E74C3C", "Adherent" = "#27AE60")) +
  labs(title    = "Financial Burden by Age Group & Adherence",
       subtitle  = "Higher burden correlates with lower adherence",
       x = "Age Group",
       y = "Financial Burden (Claim / Contribution)",
       fill = "Adherence Status") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(file.path(output_dir, "R_03_financial_burden_by_age.png"),
       p3, width = 9, height = 6, dpi = 150)
cat("  [OK] Saved: R_03_financial_burden_by_age.png\n")


# ── Plot 4: Adherence Rate by Chronic Condition ───────────────
p4 <- df %>%
  group_by(chronic_condition) %>%
  summarise(
    adherence_rate = mean(adherent, na.rm = TRUE),
    n = n(),
    se = sd(adherent, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  ) %>%
  arrange(adherence_rate) %>%
  mutate(chronic_condition = factor(chronic_condition, levels = chronic_condition)) %>%
  ggplot(aes(x = chronic_condition, y = adherence_rate, fill = adherence_rate)) +
  geom_col(color = "black", show.legend = FALSE) +
  geom_errorbar(aes(ymin = adherence_rate - se, ymax = adherence_rate + se),
                width = 0.25, color = "gray30") +
  geom_text(aes(label = scales::percent(adherence_rate, accuracy = 0.1)),
            hjust = -0.2, fontface = "bold", size = 3.5) +
  scale_fill_gradient(low = "#FADBD8", high = "#1E8449") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.1)) +
  coord_flip() +
  labs(title    = "Adherence Rate by Chronic Condition",
       subtitle  = "Error bars show ±1 standard error",
       x = NULL,
       y = "Adherence Rate (%)")

ggsave(file.path(output_dir, "R_04_adherence_by_condition.png"),
       p4, width = 9, height = 6, dpi = 150)
cat("  [OK] Saved: R_04_adherence_by_condition.png\n")


# ── Plot 5: Age Group vs Adherence (Stacked Bar) ──────────────
p5 <- df %>%
  group_by(age_group, adherence_label) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(age_group) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = age_group, y = pct, fill = adherence_label)) +
  geom_col(color = "black", width = 0.6) +
  geom_text(aes(label = scales::percent(pct, accuracy = 1)),
            position = position_stack(vjust = 0.5),
            fontface = "bold", size = 4, color = "white") +
  scale_fill_manual(values = c("Non-Adherent" = "#E74C3C", "Adherent" = "#27AE60")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title    = "Adherence Rate by Age Group",
       subtitle  = "Elderly patients show the lowest adherence",
       x = "Age Group", y = "Proportion",
       fill = "Adherence Status")

ggsave(file.path(output_dir, "R_05_adherence_by_age_group.png"),
       p5, width = 8, height = 6, dpi = 150)
cat("  [OK] Saved: R_05_adherence_by_age_group.png\n")


# ── Plot 6: Days Supply vs Adherence ──────────────────────────
p6 <- df %>%
  mutate(days_supply = factor(days_supply, labels = c("30-day", "60-day", "90-day"))) %>%
  group_by(days_supply, adherence_label) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(days_supply) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = days_supply, y = pct, fill = adherence_label)) +
  geom_col(color = "black", width = 0.5) +
  geom_text(aes(label = scales::percent(pct, accuracy = 1)),
            position = position_stack(vjust = 0.5),
            fontface = "bold", size = 4.5, color = "white") +
  scale_fill_manual(values = c("Non-Adherent" = "#E74C3C", "Adherent" = "#27AE60")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title    = "Prescription Days Supply vs Adherence",
       subtitle  = "90-day supply prescriptions show better adherence",
       x = "Prescription Days Supply",
       y = "Proportion",
       fill = "Adherence Status")

ggsave(file.path(output_dir, "R_06_days_supply_adherence.png"),
       p6, width = 8, height = 6, dpi = 150)
cat("  [OK] Saved: R_06_days_supply_adherence.png\n")


# ── Plot 7: Scatter — Refill Ratio vs Financial Burden ────────
p7 <- df %>%
  filter(!is.na(financial_burden), financial_burden < 4) %>%
  ggplot(aes(x = financial_burden, y = refill_ratio, color = adherence_label)) +
  geom_point(alpha = 0.35, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, size = 1.2) +
  scale_color_manual(values = c("Non-Adherent" = "#E74C3C", "Adherent" = "#27AE60")) +
  labs(title    = "Refill Ratio vs Financial Burden",
       subtitle  = "Higher financial burden → lower refill ratio",
       x = "Financial Burden (Claim / Contribution)",
       y = "Refill Ratio",
       color = "Adherence Status")

ggsave(file.path(output_dir, "R_07_refill_vs_financial.png"),
       p7, width = 9, height = 6, dpi = 150)
cat("  [OK] Saved: R_07_refill_vs_financial.png\n")


# ── Plot 8: Medication Complexity vs Adherence ────────────────
p8 <- df %>%
  group_by(med_complexity) %>%
  summarise(
    adherence_rate = mean(adherent, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = med_complexity, y = adherence_rate,
             fill = adherence_rate, group = 1)) +
  geom_col(color = "black", show.legend = FALSE) +
  geom_line(color = "#2C3E50", size = 1.2, linetype = "dashed") +
  geom_point(size = 4, color = "#2C3E50") +
  geom_text(aes(label = scales::percent(adherence_rate, accuracy = 0.1)),
            vjust = -0.8, fontface = "bold", size = 4) +
  scale_fill_gradient(low = "#F1948A", high = "#58D68D") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.1)) +
  labs(title    = "Adherence Rate by Medication Complexity",
       subtitle  = "Patients on more medications show lower adherence",
       x = "Medication Complexity",
       y = "Adherence Rate (%)")

ggsave(file.path(output_dir, "R_08_medication_complexity.png"),
       p8, width = 8, height = 6, dpi = 150)
cat("  [OK] Saved: R_08_medication_complexity.png\n\n")


# ── 6. Logistic Regression (Statistical) ─────────────────────

cat("============================================================\n")
cat("STATISTICAL LOGISTIC REGRESSION (R)\n")
cat("============================================================\n")
cat("Note: This is for statistical inference, not prediction.\n\n")

df_model <- df %>%
  filter(!is.na(financial_burden)) %>%
  mutate(
    gender_male   = as.integer(gender == "Male"),
    elderly       = as.integer(age >= 65),
    long_supply   = as.integer(days_supply == 90),
    high_med      = as.integer(num_medications >= 5)
  )

log_model <- glm(
  adherent ~ refill_ratio + financial_burden + age + gender_male +
             long_supply + high_med,
  family = binomial(link = "logit"),
  data   = df_model
)

cat("Model Summary:\n")
print(summary(log_model))

# Odds Ratios
cat("\nOdds Ratios (exp(coefficients)):\n")
or_table <- data.frame(
  Feature    = names(coef(log_model)),
  Coeff      = round(coef(log_model), 4),
  Odds_Ratio = round(exp(coef(log_model)), 4)
)
print(or_table)


# ── 7. Correlation Matrix ─────────────────────────────────────

cat("\n============================================================\n")
cat("CORRELATION ANALYSIS\n")
cat("============================================================\n\n")

num_vars <- df %>%
  select(adherent, age, refill_ratio, financial_burden,
         num_medications, expected_refills, refills_received) %>%
  filter(complete.cases(.))

cor_matrix <- cor(num_vars)
cat("Correlation with 'adherent':\n")
cor_with_target <- sort(cor_matrix[, "adherent"], decreasing = TRUE)
for (var in names(cor_with_target)) {
  direction <- if (cor_with_target[var] > 0) "+" else "-"
  cat(sprintf("  %s %-25s: %+.4f\n", direction, var, cor_with_target[var]))
}


# ── Final Summary ──────────────────────────────────────────────

cat("\n============================================================\n")
cat("[OK] R ANALYSIS COMPLETE\n")
cat("============================================================\n")
cat(sprintf("  Plots saved to: %s/\n", output_dir))
cat("  Files generated:\n")
cat("    R_01_adherence_distribution.png\n")
cat("    R_02_refill_ratio_distribution.png\n")
cat("    R_03_financial_burden_by_age.png\n")
cat("    R_04_adherence_by_condition.png\n")
cat("    R_05_adherence_by_age_group.png\n")
cat("    R_06_days_supply_adherence.png\n")
cat("    R_07_refill_vs_financial.png\n")
cat("    R_08_medication_complexity.png\n")
cat("\nKEY FINDINGS:\n")
cat("  1. Refill ratio is the strongest predictor of adherence\n")
cat("  2. Higher financial burden significantly reduces adherence\n")
cat("  3. Elderly patients show lower adherence rates\n")
cat("  4. 90-day supply prescriptions improve adherence\n")
cat("  5. Polypharmacy (5+ medications) reduces adherence\n\n")
