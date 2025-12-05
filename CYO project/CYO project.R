################################################################################
# CYO project.R
# Place this in the same folder as:
#   Video_Games_Sales_as_at_22_Dec_2016.csv
################################################################################

# -----------------------------
# 0. Packages
# -----------------------------
required_pkgs <- c(
  "data.table", "tidyverse", "janitor", "caret", "glmnet", "randomForest",
  "xgboost", "vip", "Matrix", "lubridate", "skimr", "rlang", "scales", "gridExtra"
)

install_if_missing <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}
install_if_missing(required_pkgs)

set.seed(2025)

# -----------------------------
# 1. Load data (relative path)
# -----------------------------
data_file <- "Video_Games_Sales_as_at_22_Dec_2016.csv"
if (!file.exists(data_file)) stop("Place Video_Games_Sales_as_at_22_Dec_2016.csv in this folder.")

vg_raw <- data.table::fread(data_file, na.strings = c("", "NA", "N/A", "tbd")) %>% as_tibble()
cat("Loaded dataset with", nrow(vg_raw), "rows and", ncol(vg_raw), "columns.\n")

# -----------------------------
# 2. Initial cleaning & types
# -----------------------------
vg <- vg_raw %>% janitor::clean_names()

as_num_safe <- function(x) {
  x <- ifelse(x %in% c("tbd", "TBD", "", "NA"), NA, x)
  suppressWarnings(as.numeric(x))
}

possible_num_cols <- c("critic_score","critic_count","user_score","user_count",
                       "year_of_release","na_sales","eu_sales","jp_sales",
                       "other_sales","global_sales")

for (col in intersect(possible_num_cols, names(vg))) vg[[col]] <- as_num_safe(vg[[col]])
for (col in intersect(c("platform","genre","publisher","rating","developer"), names(vg))) vg[[col]] <- as.factor(vg[[col]])

# -----------------------------
# 3. Filter & features
# -----------------------------
if (!"global_sales" %in% names(vg)) stop("global_sales column missing")
vg <- vg %>% filter(!is.na(global_sales), global_sales > 0)
cat("Rows after keeping positive sales:", nrow(vg), "\n")

vg <- vg %>%
  mutate(year_of_release = ifelse("year_of_release" %in% names(vg), as.integer(year_of_release), NA_integer_),
         release_decade = case_when(
           is.na(year_of_release) ~ "unknown",
           year_of_release < 1980 ~ "pre_1980",
           TRUE ~ paste0(floor(year_of_release / 10) * 10, "s")
         ) %>% as.factor(),
         age_years = ifelse(!is.na(year_of_release), 2016 - year_of_release, NA_real_)
  )

if ("genre" %in% names(vg)) {
  vg <- vg %>% mutate(
    genre = as.character(genre),
    genre_count = ifelse(is.na(genre), 0, str_count(genre, "\\|") + 1),
    primary_genre = ifelse(genre_count >= 1, sapply(str_split(genre, "\\|"), `[`, 1), NA_character_) %>% as.factor()
  )
}

if ("platform" %in% names(vg)) {
  vg <- vg %>% left_join(vg %>% group_by(platform) %>% summarise(platform_avg_sales = mean(global_sales, na.rm=TRUE), platform_n = n()), by="platform")
}
if ("publisher" %in% names(vg)) {
  vg <- vg %>% left_join(vg %>% group_by(publisher) %>% summarise(publisher_avg_sales = mean(global_sales, na.rm=TRUE), publisher_n = n()), by="publisher")
}

if ("critic_score" %in% names(vg)) vg <- vg %>% mutate(critic_score_scaled = critic_score / 100)
if ("user_score" %in% names(vg))   vg <- vg %>% mutate(user_score_scaled = user_score / 10)

vg <- vg %>% mutate(log_sales = log1p(global_sales))

# -----------------------------
# 4. Candidate features & imputation
# -----------------------------
candidate_features <- c("critic_score","critic_count","user_score","user_count",
                        "age_years","platform","primary_genre","publisher",
                        "platform_avg_sales","publisher_avg_sales","genre_count","release_decade")
candidate_features <- intersect(candidate_features, names(vg))
cat("Candidate features:", paste(candidate_features, collapse = ", "), "\n")

model_df <- vg %>% select(all_of(c(candidate_features, "global_sales", "log_sales")))

# numeric median impute
numeric_feats <- model_df %>% select(where(is.numeric)) %>% select(-global_sales, -log_sales) %>% names()
for (col in numeric_feats) model_df[[col]][is.na(model_df[[col]])] <- median(model_df[[col]], na.rm = TRUE)

# categorical NA -> "Unknown"
cat_feats <- setdiff(candidate_features, numeric_feats)
for (col in cat_feats) {
  model_df[[col]] <- as.character(model_df[[col]])
  model_df[[col]][is.na(model_df[[col]]) | model_df[[col]] == ""] <- "Unknown"
  model_df[[col]] <- as.factor(model_df[[col]])
}

# -----------------------------
# 5. Train/test split (80/20)
# -----------------------------
set.seed(2025)
train_index <- createDataPartition(model_df$log_sales, p = 0.8, list = FALSE)
train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]
cat("Training rows:", nrow(train_df), "Test rows:", nrow(test_df), "\n")

# evaluation helpers
RMSE_orig <- function(true_orig, pred_orig) sqrt(mean((true_orig - pred_orig)^2))
MAE_orig  <- function(true_orig, pred_orig) mean(abs(true_orig - pred_orig))

# -----------------------------
# 6. Target-encode high-cardinality factors (training-only stats)
# -----------------------------
target_encode_column <- function(train_df, test_df, col, target="log_sales", m = 10) {
  stats <- train_df %>%
    group_by(.data[[col]]) %>%
    summarise(n = n(), sumy = sum(.data[[target]], na.rm = TRUE), .groups = "drop") %>%
    mutate(level_mean = sumy / n)
  global_mean <- mean(train_df[[target]], na.rm = TRUE)
  stats <- stats %>% mutate(smoothed = (sumy + m * global_mean) / (n + m))
  new_col <- paste0(col, "_te")
  train_out <- train_df %>% left_join(stats %>% select(!!sym(col), smoothed), by = col) %>%
    mutate(!!sym(new_col) := smoothed) %>% select(-smoothed, -all_of(col))
  test_out <- test_df %>% left_join(stats %>% select(!!sym(col), smoothed), by = col) %>%
    mutate(!!sym(new_col) := ifelse(is.na(smoothed), global_mean, smoothed)) %>% select(-smoothed, -all_of(col))
  list(train = train_out, test = test_out)
}

K <- 53
high_card_cols <- names(train_df %>% select(where(is.factor)))[sapply(train_df %>% select(where(is.factor)), nlevels) > K]
if (length(high_card_cols) > 0) cat("High-cardinality factors to encode:", paste(high_card_cols, collapse = ", "), "\n")
for (hc in high_card_cols) {
  res <- target_encode_column(train_df, test_df, hc, target = "log_sales", m = 10)
  train_df <- res$train
  test_df  <- res$test
}
# convert remaining character to factors
train_df <- train_df %>% mutate(across(where(is.character), as.factor))
test_df  <- test_df  %>% mutate(across(where(is.character), as.factor))

# -----------------------------
# 7. EDA plots 
# -----------------------------
if (interactive()) {
  # 1. Distribution: Global Sales
  p1 <- ggplot(model_df, aes(global_sales)) +
    geom_histogram(bins = 60, fill = "steelblue", color = "white") +
    scale_x_continuous(labels = comma) +
    labs(title = "Distribution of Global Sales (millions)", x = "Global Sales (M)", y = "Count")
  print(p1)
  
  # 2. Distribution: log_sales
  p2 <- ggplot(model_df, aes(log_sales)) +
    geom_histogram(bins = 60, fill = "darkgreen", color = "white") +
    labs(title = "Distribution of log1p(Global Sales)", x = "log1p(Global Sales)", y = "Count")
  print(p2)
  
  # 3. Sales by primary_genre (mean)
  if ("primary_genre" %in% names(model_df)) {
    p3 <- model_df %>%
      group_by(primary_genre) %>%
      summarise(mean_sales = mean(global_sales, na.rm = TRUE), n = n()) %>%
      arrange(desc(mean_sales)) %>%
      slice_head(n = 20) %>%
      ggplot(aes(reorder(primary_genre, mean_sales), mean_sales)) +
      geom_col(fill = "tomato") + coord_flip() +
      labs(title = "Mean Global Sales by Primary Genre (top 20)", x = "Primary Genre", y = "Mean Sales (M)")
    print(p3)
  }
  
  # 4. Sales by platform (mean)
  if ("platform" %in% names(model_df)) {
    p4 <- model_df %>%
      group_by(platform) %>%
      summarise(mean_sales = mean(global_sales, na.rm = TRUE), n = n()) %>%
      arrange(desc(mean_sales)) %>%
      slice_head(n = 20) %>%
      ggplot(aes(reorder(platform, mean_sales), mean_sales)) +
      geom_col(fill = "purple") + coord_flip() +
      labs(title = "Mean Global Sales by Platform (top 20)", x = "Platform", y = "Mean Sales (M)")
    print(p4)
  }
  
  # 5. Critic/User score vs log_sales
  if ("critic_score" %in% names(model_df)) {
    p5 <- ggplot(model_df, aes(critic_score, log_sales)) + geom_point(alpha = 0.3) + geom_smooth(method = "loess") +
      labs(title = "log1p(Global Sales) vs Critic Score", x = "Critic Score", y = "log1p(Global Sales)")
    print(p5)
  }
  if ("user_score" %in% names(model_df)) {
    p6 <- ggplot(model_df, aes(user_score, log_sales)) + geom_point(alpha = 0.3) + geom_smooth(method = "loess") +
      labs(title = "log1p(Global Sales) vs User Score", x = "User Score", y = "log1p(Global Sales)")
    print(p6)
  }
  
  # 6. Correlation heatmap for numeric predictors
  num_for_corr <- model_df %>% select(where(is.numeric)) %>% select(-global_sales, -log_sales)
  if (ncol(num_for_corr) >= 2) {
    cor_mat <- cor(num_for_corr, use = "pairwise.complete.obs")
    cor_df <- as.data.frame(as.table(cor_mat))
    p7 <- ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
      geom_tile() + scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) + labs(title = "Correlation heatmap (numeric predictors)")
    print(p7)
  }
}

# -----------------------------
# 8. GLMNET via caret (regularized linear) - build model.matrix from combined
# -----------------------------
combined_glm <- bind_rows(train_df %>% mutate(.dataset = "train"), test_df %>% mutate(.dataset = "test"))
combined_glm <- combined_glm %>% mutate(across(where(is.character), ~ ifelse(. == "" | is.na(.), "Unknown", .))) %>% mutate(across(where(is.character), as.factor))
combined_glm <- combined_glm %>% mutate(across(where(is.factor), droplevels))
one_level_cols <- names(combined_glm)[sapply(combined_glm, function(x) is.factor(x) && nlevels(x) < 2)]
if (length(one_level_cols) > 0) {
  combined_glm <- combined_glm %>% select(-all_of(one_level_cols))
  cat("Dropped single-level cols for GLMNET:", paste(one_level_cols, collapse = ", "), "\n")
}
train_df_glm <- combined_glm %>% filter(.dataset == "train") %>% select(-.dataset)
test_df_glm  <- combined_glm %>% filter(.dataset == "test")  %>% select(-.dataset)

glmnet_x <- model.matrix(log_sales ~ ., data = train_df_glm)[, -1, drop = FALSE]
glmnet_test_x <- model.matrix(log_sales ~ ., data = test_df_glm)[, -1, drop = FALSE]
if (ncol(glmnet_x) != ncol(glmnet_test_x)) stop("GLMNET matrix column mismatch")
glmnet_y <- train_df$log_sales

ctrl <- trainControl(method = "cv", number = 5)
glmnet_grid <- expand.grid(alpha = c(0, 0.5, 1), lambda = 10^seq(-4, 0, length = 8))

cat("Training GLMNET (caret)...\n")
glmnet_fit <- caret::train(x = glmnet_x, y = glmnet_y, method = "glmnet", trControl = ctrl, tuneGrid = glmnet_grid, metric = "RMSE")
print(glmnet_fit)

# GLMNET coefficient path and lambda plot
if (interactive()) {
  plot(glmnet_fit$finalModel, xvar = "lambda", label = TRUE)
  title("GLMNET coefficient paths (final model)")
}

pred_glm_log  <- predict(glmnet_fit, glmnet_test_x)
pred_glm_orig <- expm1(pred_glm_log)
rmse_glm <- RMSE_orig(test_df$global_sales, pred_glm_orig)
mae_glm  <- MAE_orig(test_df$global_sales, pred_glm_orig)
cat(sprintf("GLMNET - RMSE: %.5f, MAE: %.5f\n", rmse_glm, mae_glm))

# -----------------------------
# 9. Random Forest via caret
# -----------------------------
clean_predictors <- function(df) {
  df2 <- df %>% mutate(across(where(is.factor), droplevels))
  keep <- sapply(df2, function(col) if (is.factor(col)) nlevels(col) >= 2 else TRUE)
  df2 <- df2[, keep, drop = FALSE]
  df2
}

train_x_rf <- train_df %>% select(-global_sales, -log_sales) %>% clean_predictors()
test_x_rf  <- test_df  %>% select(-global_sales, -log_sales) %>% clean_predictors()

# align columns
missing_cols <- setdiff(names(train_x_rf), names(test_x_rf))
if (length(missing_cols) > 0) for (mc in missing_cols) test_x_rf[[mc]] <- NA
test_x_rf <- test_x_rf[, names(train_x_rf), drop = FALSE]

# coerce test factors to train levels & fill NA
for (col in names(train_x_rf)) {
  if (is.factor(train_x_rf[[col]])) {
    train_lv <- levels(train_x_rf[[col]])
    if (!"Unknown" %in% train_lv) train_lv <- c(train_lv, "Unknown")
    train_x_rf[[col]] <- factor(as.character(train_x_rf[[col]]), levels = train_lv)
    test_x_rf[[col]] <- factor(as.character(test_x_rf[[col]]), levels = train_lv)
    test_x_rf[[col]][is.na(test_x_rf[[col]])] <- ifelse("Unknown" %in% train_lv, "Unknown", train_lv[1])
  }
}

p_features <- ncol(train_x_rf)
rf_grid <- expand.grid(mtry = unique(pmax(1, floor(c(sqrt(p_features), sqrt(p_features)*1.5)))))

cat("Training Random Forest (caret)...\n")
rf_fit <- caret::train(x = train_x_rf, y = train_df$log_sales, method = "rf", trControl = ctrl, tuneGrid = rf_grid, ntree = 300, importance = TRUE)
print(rf_fit)

pred_rf_log  <- predict(rf_fit, test_x_rf)
pred_rf_orig <- expm1(pred_rf_log)
rmse_rf <- RMSE_orig(test_df$global_sales, pred_rf_orig)
mae_rf  <- MAE_orig(test_df$global_sales, pred_rf_orig)
cat(sprintf("Random Forest - RMSE: %.5f, MAE: %.5f\n", rmse_rf, mae_rf))

# RF varImp plot
if (interactive()) {
  vi_rf <- varImp(rf_fit, scale = TRUE)
  print(vi_rf)
  plot(vi_rf, top = 20, main = "Random Forest Variable Importance")
}

# -----------------------------
# 10. XGBoost (direct xgboost workflow: numeric matrices + training curve)
#   - build numeric model.matrix (one-hot) from combined train/test to ensure identical columns
# -----------------------------
xgb_combined <- bind_rows(train_df %>% mutate(.dataset = "train"), test_df %>% mutate(.dataset = "test"))
xgb_combined <- xgb_combined %>% mutate(across(where(is.character), ~ ifelse(. == "" | is.na(.), "Unknown", .))) %>% mutate(across(where(is.character), as.factor))
xgb_combined <- xgb_combined %>% mutate(across(where(is.factor), droplevels))
# drop single-level factors if present
one_lvl <- names(xgb_combined)[sapply(xgb_combined, function(x) is.factor(x) && nlevels(x) < 2)]
if (length(one_lvl) > 0) { xgb_combined <- xgb_combined %>% select(-all_of(one_lvl)); cat("Dropped single-level xgb cols:", paste(one_lvl, collapse=", "), "\n") }

xgb_mm <- model.matrix(log_sales ~ . - global_sales - .dataset, data = xgb_combined)
train_x_xgb <- xgb_mm[xgb_combined$.dataset == "train", , drop = FALSE]
test_x_xgb  <- xgb_mm[xgb_combined$.dataset == "test", , drop = FALSE]
train_y_xgb <- train_df$log_sales

# convert to xgb.DMatrix
dtrain <- xgboost::xgb.DMatrix(data = train_x_xgb, label = train_y_xgb)
dtest  <- xgboost::xgb.DMatrix(data = test_x_xgb, label = test_df$log_sales)

# xgboost params - regression
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# training with watchlist and early stopping to get training curve
watchlist <- list(train = dtrain, test = dtest)
nrounds <- 1000
early_stop_rounds <- 30

cat("Training XGBoost (direct xgboost::xgb.train) with early stopping...\n")
xgb_model <- xgboost::xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = nrounds,
  watchlist = watchlist,
  early_stopping_rounds = early_stop_rounds,
  print_every_n = 50,
  verbose = 1
)

# training log: get evaluation log for plotting
eval_log <- xgb_model$evaluation_log

# predictions and back-transform
pred_xgb_log <- predict(xgb_model, newdata = test_x_xgb)
pred_xgb_orig <- expm1(pred_xgb_log)
rmse_xgb <- RMSE_orig(test_df$global_sales, pred_xgb_orig)
mae_xgb  <- MAE_orig(test_df$global_sales, pred_xgb_orig)
cat(sprintf("XGBoost - RMSE: %.5f, MAE: %.5f\n", rmse_xgb, mae_xgb))

# XGBoost feature importance (gain)
if (interactive()) {
  importance_matrix <- xgb.importance(model = xgb_model, feature_names = colnames(train_x_xgb))
  print(head(importance_matrix, 20))
  xgb.plot.importance(importance_matrix[1:20, ], main = "XGBoost - Top 20 Feature Importance (gain)")
  
  # Plot training curve (rmse)
  p_train <- ggplot(eval_log, aes(x = iter)) +
    geom_line(aes(y = train_rmse, color = "train")) +
    geom_line(aes(y = test_rmse, color = "test")) +
    labs(title = "XGBoost training curve (RMSE)", x = "Iteration", y = "RMSE") +
    scale_color_manual(values = c("train" = "blue", "test" = "red"))
  print(p_train)
}

# -----------------------------
# 11. Model comparison & diagnostics (key plots)
# -----------------------------
model_summary <- tibble(
  model = c("GLMNET", "RandomForest", "XGBoost"),
  RMSE_orig = c(rmse_glm, rmse_rf, rmse_xgb),
  MAE_orig  = c(mae_glm, mae_rf, mae_xgb)
) %>% arrange(RMSE_orig)

print(model_summary)
cat("Best by RMSE:", model_summary$model[1], "\n")

# Pred vs Observed for best model
best_model <- model_summary$model[1]
best_pred <- switch(best_model, GLMNET = pred_glm_orig, RandomForest = pred_rf_orig, XGBoost = pred_xgb_orig)

if (interactive()) {
  p_predobs <- ggplot(data.frame(obs = test_df$global_sales, pred = best_pred), aes(x = obs, y = pred)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(title = paste("Predicted vs Observed -", best_model), x = "Observed Global Sales", y = "Predicted Global Sales")
  print(p_predobs)
  
  resid_best <- test_df$global_sales - best_pred
  p_resid <- ggplot(data.frame(resid = resid_best), aes(x = resid)) +
    geom_histogram(bins = 50, fill = "gray", color = "black") +
    labs(title = paste("Residuals -", best_model), x = "Residual (obs - pred)", y = "Count")
  print(p_resid)
  
  p_resid_vs_age <- NULL
  if ("age_years" %in% names(test_df)) {
    p_resid_vs_age <- ggplot(data.frame(resid = resid_best, age_years = test_df$age_years), aes(x = age_years, y = resid)) +
      geom_point(alpha = 0.4) + geom_smooth(method = "loess") +
      labs(title = paste("Residuals vs Age (years) -", best_model), x = "Age (years)", y = "Residual")
    print(p_resid_vs_age)
  }
}