#Takes several minutes to load full proyect
# 0. Loading Packages

packages <- c(
  "tidyverse",
  "caret",
  "lubridate",
  "data.table",
  "gridExtra"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Helper RMSE function
RMSE <- function(true, predicted) {
  sqrt(mean((true - predicted)^2))
}

# 1. CREATE edx AND final_holdout_test SETS

# MovieLens 10M dataset:
options(timeout = 120)

dl <- "ml-10M100K.zip"
if (!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if (!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if (!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(
  str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(
    userId = as.integer(userId),
    movieId = as.integer(movieId),
    rating = as.numeric(rating),
    timestamp = as.integer(timestamp)
  )

movies <- as.data.frame(
  str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE)

colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Create final holdout test set (10%)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1,
                                  p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Ensure userId and movieId in test set exist in training set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed back into edx
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# 2. EXPLORATORY DATA ANALYSIS (EDA)

# Convert timestamp
edx <- edx %>% mutate(rating_date = as_datetime(timestamp))

# ------------------------------------------
# Rating distribution
ggplot(edx, aes(rating)) +
  geom_histogram(binwidth = 0.5, fill="steelblue", color="white") +
  labs(title="Rating Distribution", x="Rating", y="Count")

# ------------------------------------------
# Movie rating counts
movie_rating_counts <- edx %>%
  group_by(movieId) %>%
  summarize(
    count = n(),
    avg_rating = mean(rating)
  )

ggplot(movie_rating_counts, aes(count)) +
  geom_histogram(bins=50, fill="green", color="white") +
  scale_x_log10() +
  labs(title="Ratings per Movie (log scale)", x="Count", y="Frequency")

# ------------------------------------------
# User rating counts
user_rating_counts <- edx %>%
  group_by(userId) %>%
  summarize(count = n())

ggplot(user_rating_counts, aes(count)) +
  geom_histogram(bins=50, fill="yellow", color="white") +
  scale_x_log10() +
  labs(title="Ratings per User (log scale)", x="Count", y="Frequency")

# ------------------------------------------
# Genre counts
genre_counts <- edx %>%
  separate_rows(genres, sep="\\|") %>%
  count(genres, sort=TRUE)

ggplot(genre_counts, aes(reorder(genres, n), n)) +
  geom_col(fill="tomato") +
  coord_flip() +
  labs(title="Genre Popularity", x="Genre", y="Count")

# 3. CREATE TRAIN/TEST SPLIT FOR MODEL DEVELOPMENT

set.seed(1, sample.kind="Rounding")
dev_index <- createDataPartition(edx$rating, p=0.1, list=FALSE)

train_dev <- edx[-dev_index, ]
test_dev  <- edx[dev_index, ]

# Ensure only known movies/users
test_dev <- test_dev %>%
  semi_join(train_dev, by="movieId") %>%
  semi_join(train_dev, by="userId")

removed <- anti_join(edx[dev_index, ], test_dev)
train_dev <- rbind(train_dev, removed)

# 4. MODELING

mu <- mean(train_dev$rating)

# MODEL 1 — GLOBAL AVERAGE
pred1 <- rep(mu, nrow(test_dev))
rmse_1 <- RMSE(test_dev$rating, pred1)

# MODEL 2 — MOVIE EFFECT
movie_effects <- train_dev %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

pred2 <- test_dev %>%
  left_join(movie_effects, by="movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

rmse_2 <- RMSE(test_dev$rating, pred2)

# MODEL 3 — MOVIE + USER EFFECT
user_effects <- train_dev %>%
  left_join(movie_effects, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred3 <- test_dev %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(user_effects, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_3 <- RMSE(test_dev$rating, pred3)

# MODEL 4 — GENRE EFFECT
train_genre <- train_dev %>%
  separate_rows(genres, sep="\\|") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu))

test_genre <- test_dev %>%
  separate_rows(genres, sep="\\|") %>%
  left_join(train_genre, by="genres") %>%
  group_by(userId, movieId, rating) %>%
  summarize(b_g = mean(b_g, na.rm=TRUE), .groups="drop")

pred4 <- test_dev %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(user_effects, by="userId") %>%
  left_join(test_genre, by=c("movieId","userId","rating")) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

rmse_4 <- RMSE(test_dev$rating, pred4)

# MODEL 5 — REGULARIZED MOVIE + USER EFFECT
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(lambda) {
  
  b_i <- train_dev %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- train_dev %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  preds <- test_dev %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  RMSE(test_dev$rating, preds)
})

lambda_results <- tibble(lambda=lambdas, rmse=rmses)
best_lambda <- lambda_results$lambda[which.min(lambda_results$rmse)]

# Plot tuning curve
ggplot(lambda_results, aes(lambda, rmse)) +
  geom_line() +
  geom_point() +
  labs(title="Lambda Tuning Curve", x="Lambda", y="RMSE")

# Fit final regularized model
lambda <- best_lambda

b_i_reg <- train_dev %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

b_u_reg <- train_dev %>%
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

pred5 <- test_dev %>%
  left_join(b_i_reg, by="movieId") %>%
  left_join(b_u_reg, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_5 <- RMSE(test_dev$rating, pred5)

# 5. MODEL COMPARISON TABLE

model_results <- tibble(
  Model = c(
    "Global Average",
    "Movie Effect",
    "Movie + User Effect",
    "Movie + User + Genre Effect",
    "Regularized Movie + User"
  ),
  RMSE = c(rmse_1, rmse_2, rmse_3, rmse_4, rmse_5)
)

model_results

# 6. FINAL MODEL TRAINED ON FULL edx + TEST ON final_holdout_test

mu_full <- mean(edx$rating)

b_i_full <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_full) / (n() + best_lambda))

b_u_full <- edx %>%
  left_join(b_i_full, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_full - b_i) / (n() + best_lambda))

final_predictions <- final_holdout_test %>%
  left_join(b_i_full, by="movieId") %>%
  left_join(b_u_full, by="userId") %>%
  mutate(pred = mu_full + b_i + b_u) %>%
  pull(pred)

final_rmse <- RMSE(final_holdout_test$rating, final_predictions)

cat("FINAL RMSE on final_holdout_test:", final_rmse, "\n")