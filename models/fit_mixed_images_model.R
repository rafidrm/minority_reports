library(lme4)
library(dplyr)

# Read data and arguments
args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]
feature_cols <- strsplit(args[2], ",")[[1]]

cat("\nReading data...\n")
df <- read.csv(data_file)
cat("Data loaded:", nrow(df), "rows\n")

# Fit models
cat("\nFitting full model...\n")
formula <- as.formula(paste0('disagree ~ ', 
                           paste(feature_cols, collapse=" + "),
                           ' + (1|image_path)'))
cat("Using formula:\n", deparse(formula), "\n")

model <- glmer(formula, family=binomial, data=df, 
              control=glmerControl(optimizer="bobyqa"))

cat("\nFitting null model with random effects...\n")
null_model <- glmer(disagree ~ 1 + (1|image_path), 
                   family=binomial, data=df,
                   control=glmerControl(optimizer="bobyqa"))

cat("\nFitting simple null model...\n")
simple_null_model <- glm(disagree ~ 1, family=binomial, data=df)

cat("\nExtracting and saving results...\n")

# Save model summaries with model-specific prefix
cat("Saving model summary...\n")
sink('tmp/meimages_model_summary.txt')
print(summary(model))
sink()

# Save fixed effects
fixed <- data.frame(
    coef=fixef(model),
    std_err=sqrt(diag(vcov(model))),
    row.names=names(fixef(model))
)
fixed$z <- fixed$coef / fixed$std_err
fixed$p_value <- 2 * pnorm(-abs(fixed$z))
write.csv(fixed, 'tmp/meimages_fixed_effects.csv')

# Save random effects
re <- ranef(model, condVar=TRUE)
random <- data.frame(
    image_path = rownames(re$image_path),
    effect = re$image_path[,1],
    std_err = sqrt(attr(re$image_path, "postVar")[1,,])
)
write.csv(random, 'tmp/meimages_random_effects.csv', row.names=FALSE)

# Save predictions
predictions <- predict(model, type='response')
write.csv(data.frame(pred=predictions), 'tmp/meimages_predictions.csv', row.names=FALSE)

# Save model fit statistics
cat("\nSaving model fit statistics...\n")
ll_full <- as.numeric(logLik(model))
ll_null <- as.numeric(logLik(null_model))
ll_simple_null <- as.numeric(logLik(simple_null_model))
aic <- AIC(model)
mcfadden_r2 <- 1 - (ll_full / ll_null)
mcfadden_r2_simple <- 1 - (ll_full / ll_simple_null)

fit_stats <- data.frame(
    loglik = ll_full,
    null_loglik = ll_null,
    simple_null_loglik = ll_simple_null,
    aic = aic,
    mcfadden_r2 = mcfadden_r2,
    mcfadden_r2_simple = mcfadden_r2_simple
)
write.csv(fit_stats, 'tmp/meimages_fit_stats.csv', row.names=FALSE) 