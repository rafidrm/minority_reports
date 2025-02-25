args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]
predictors <- args[2]  # comma-separated list of q_* columns
output_dir <- args[3]
temp_dir <- args[4]  # experiment-specific temp directory
dataset <- args[5]   # ECPD or ZOD

if(!dir.exists(output_dir)){
  dir.create(output_dir, recursive=TRUE)
}

library(lme4)
library(readr)

# Read training data
df <- read_csv(data_file)

# Check if response is constant
if (length(unique(df$disagree)) <= 1) {
  stop("Error: 'disagree' column has no variation (all values are the same)")
}

# Calculate class weights
n_samples <- nrow(df)
n_classes <- table(df$disagree)
class_weights <- n_samples / (2 * n_classes)
observation_weights <- ifelse(df$disagree == 1, 
                            class_weights[2], 
                            class_weights[1])

# Build formula based on dataset
if(dataset == "ECPD") {
    # For ECPD: include predictors if available
    form_str <- "disagree ~ 1"
    if(predictors != ""){
        # Split the comma-separated predictors and join with ' + '
        pred_list <- unlist(strsplit(predictors, ","))
        pred_str <- paste(pred_list, collapse=" + ")
        form_str <- paste(form_str, "+", pred_str)
    }
    form_str <- paste(form_str, "+ (1|user_id) + (1|crop_id)")
} else {
    # For ZOD: only random effects
    form_str <- "disagree ~ 1 + (1|user_id) + (1|crop_id)"
}
formula <- as.formula(form_str)

# Print class weight information
cat("\nClass weights applied:")
cat("\nClass 0 (majority):", class_weights[1])
cat("\nClass 1 (minority):", class_weights[2])
cat("\n")

# Print formula being used
cat("\nFitting model with formula:\n")
cat(deparse(formula), "\n")

# Fit the model using glmer with weights
model <- glmer(formula, 
              family = binomial, 
              data = df, 
              weights = observation_weights,
              control=glmerControl(optimizer="bobyqa"))

# Save fixed effects
fixed_effects <- as.data.frame(coef(summary(model)))
fixed_effects$param <- rownames(fixed_effects)
fixed_effects <- fixed_effects[, c("param", "Estimate")]
colnames(fixed_effects) <- c("param", "coef")
write.csv(fixed_effects, file = file.path(output_dir, "fixed_effects_simulation.csv"), row.names = FALSE)

# Save random effects: for user_id and crop_id
ranef_model <- ranef(model)
worker_re <- ranef_model$user_id
worker_re_df <- data.frame(worker_id = rownames(worker_re), effect = worker_re[,1])
write.csv(worker_re_df, file = file.path(output_dir, "worker_effects_simulation.csv"), row.names = FALSE)

crop_re <- ranef_model$crop_id
crop_re_df <- data.frame(crop_id = rownames(crop_re), effect = crop_re[,1])
write.csv(crop_re_df, file = file.path(output_dir, "crop_effects_simulation.csv"), row.names = FALSE) 