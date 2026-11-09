# Install & load required packages
install.packages("readxl")
install.packages("dplyr")
library(readxl)
library(dplyr)

# Step 1: Read the raw Excel data
vital <- read_excel("vital signs-Dataset.xlsx")
cat("Total rows in raw data:", nrow(vital), "\n")

# Step 2: Remove duplicate patient IDs
vital <- distinct(vital, patient_id, .keep_all = TRUE)
cat("After removing duplicates:", nrow(vital), "\n")

# Step 3: Convert numeric columns from character to numeric
vital$age <- as.numeric(vital$age)
vital$weight <- as.numeric(vital$weight)
vital$bp <- as.numeric(vital$bp)
vital$heart_rate <- as.numeric(vital$heart_rate)

# Step 4: Keep only 'Treatment' and 'Control' arms
vital <- filter(vital, arm %in% c("Treatment", "Control"))
cat("After keeping only Treatment and Control groups:", nrow(vital), "\n")

# Step 5: Handle missing values (fill with rounded mean)
if ("age" %in% names(vital)) {
  vital$age[is.na(vital$age)] <- round(mean(vital$age, na.rm = TRUE))
}
if ("bp" %in% names(vital)) {
  vital$bp[is.na(vital$bp)] <- round(mean(vital$bp, na.rm = TRUE))
}
if ("weight" %in% names(vital)) {
  vital$weight[is.na(vital$weight)] <- round(mean(vital$weight, na.rm = TRUE))
}
if ("heart_rate" %in% names(vital)) {
  vital$heart_rate[is.na(vital$heart_rate)] <- round(mean(vital$heart_rate, na.rm = TRUE))
}

# Step 6: Add Temp column with decimal values
set.seed(123)
vital$Temp <- round(runif(nrow(vital), min = 36, max = 37.5), 1)

# Step 7: Summary check
cat("Summary of cleaned data:\n")
print(summary(vital))

# Step 8: Save cleaned data
write.csv(vital, "vital_signs_clean.csv", row.names = FALSE)
cat("Cleaning complete! File saved as 'vital_signs_clean.csv'\n")

# Step 9: Print cleaned data
print(vital)
