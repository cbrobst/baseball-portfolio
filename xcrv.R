#### packages and initialization ####

library(caret)
library(xgboost)
library(dplyr)
library(parallel)
library(ggplot2)
library(pdp)
library(mgcv)
library(randomForest)
setwd("baseball-portfolio")

ncores = detectCores()

bip = readRDS("bip.RDS")
run_values <- readRDS("rv.RDS")

#### eda ####

ggplot(bip, aes(x = landing_dist, y = launch_angle)) + 
  stat_density_2d(aes(fill = after_stat(density)), geom = "raster", contour = FALSE) +
  scale_fill_viridis_c(option = "viridis") +
  theme_minimal()
# appears that landing distance may have been tagged at the point where
# the batted ball was fielded
# since negative launch angle should never be going 200+ feet
# so we should be interpreting carefully
# this may cause target leakage on ground balls getting thru the infield

featurePlot(
  x = bip[, c("launch_speed", "launch_angle", "spray_angle")],
  y = bip$TB,
  plot = "pairs"
)


featurePlot(
  x = bip[, c("launch_speed", "launch_angle", "spray_angle","hc_x", "hc_y",
              "bat_speed","attack_angle","swing_path_tilt")],
  y = bip$TB,
  plot = "pairs"
)

#### begin model type selection ####


test_set = bip %>% filter(year == 2026)
train_set = bip %>% filter(year < 2026)


rm(bip)
gc() # just bc i am working with so little processing power here

set.seed(2005)
train_set$partition = sample(1:10,nrow(train_set), replace = T)

train_set %>% group_by(partition) %>% summarise(count = n())

preds = data.frame()

for(group in 1:10){
  print(paste0("beginning partition ", group, " out of 10 at ", Sys.time()))
  
  ols = lm(run_value~launch_speed+poly(launch_angle,2)+poly(spray_angle,2),
           data = train_set %>% filter(partition != group))
  ols_interaction = lm(run_value~launch_speed*poly(launch_angle,2)*poly(spray_angle,2),
                       data = train_set %>% filter(partition != group))
  print("Begin gam")
  gam_model <- gam( run_value ~ s(launch_speed) + s(launch_angle) + s(spray_angle), 
                    data = train_set %>% filter(partition != group), method = "REML" )
  print("begin rf")
  rf = randomForest(x = train_set[train_set$partition != group, 
                                  c("launch_speed", "launch_angle", "spray_angle")],
                    y = as.factor(train_set[train_set$partition != group,]$TB),
                    ntree = 200, mtry = 2, importance = F, nodesize = 100)
  # this is a large node size, intended bc similar BIP should have similar outcomes
  # and bc runtime is a huge obstacle on my laptop
  # and smaller nodesize might be useful on better machines
  # although i feel ok with it at 100 given the large data size
  print("begin xgb")
  dtrain <- xgb.DMatrix(
    data = as.matrix((train_set %>% filter(partition != group))[,c("launch_speed", "launch_angle", "spray_angle")]),
    label = as.integer((train_set %>% filter(partition != group))$TB)
  )
  xgb <- xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 5,
      max_depth = 3,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 1
    ),
    data = dtrain,
    nrounds = 200,
    verbose = 1
  )
  rm(dtrain)
  gc()
  print("done training this partition, beginning predictions")
  
  pred_df <- train_set %>%
    filter(partition == group) %>%
    mutate(ols = predict(ols, newdata = .),
           ols_int = predict(ols_interaction, newdata = .),
           gam = predict(gam_model, newdata = .),
           rf = predict(rf, 
                        newdata = select(., launch_speed, launch_angle, spray_angle), 
                        type = "prob") %*% run_values$run_value,
           xgb = matrix(predict(xgb, newdata = as.matrix(select(., 
                                launch_speed, launch_angle, spray_angle)), 
                                type = "prob"),
                        ncol = 5, byrow = TRUE) %*% run_values$run_value) %>%
    select(run_value, ols, ols_int, gam, rf, xgb)
  
  preds = bind_rows(pred_df, preds)
  gc()
  
}

rmsedf <- data.frame(
  model = c("OLS", "OLS_interaction", "GAM", "Random_Forest", "XGBoost"),
  RMSE = c(sqrt(mean((preds$run_value - preds$ols)^2)),
           sqrt(mean((preds$run_value - preds$ols_int)^2)),
           sqrt(mean((preds$run_value - preds$gam)^2)),
           sqrt(mean((preds$run_value - preds$rf)^2)),
           sqrt(mean((preds$run_value - preds$xgb)^2))  ))

write.csv(rmsedf, "rmse.csv", row.names = FALSE)

#### model tuning ####

tune_grid = expand.grid(
  max_depth = c(2, 3, 4, 5),
  min_child_weight = c(1, 5, 10, 20),
  eta = c(0.025, 0.05, 0.1, 0.20),
  subsample = c(0.7, 0.85, 1),
  gamma = c(0, 0.5, 1)
  # early stopping rounds? if we add nrounds param
)
tune_grid$row_num = seq_len(nrow(tune_grid))
tune_grid$rmse = rep(NA, nrow(tune_grid))
preds = data.frame()

for(row in seq_len(nrow(tune_grid))){
  if(!is.na(tune_grid$rmse[row])){next}
  
  print(paste0("beginning tuning grid row ", row, " out of ", nrow(tune_grid)))
  print(Sys.time())
  for(group in 1:10){
    dtrain <- xgb.DMatrix(
      data = as.matrix((train_set %>% filter(partition != group))[,c("launch_speed", "launch_angle", "spray_angle")]),
      label = as.integer((train_set %>% filter(partition != group))$TB)
    )
    xgb <- xgb.train(
      params = list(
        objective = "multi:softprob",
        num_class = 5,
        max_depth = tune_grid$max_depth[row],
        eta = tune_grid$eta[row],
        subsample = tune_grid$subsample[row],
        colsample_bytree = 1,
        gamma = tune_grid$gamma[row],
        min_child_weight = tune_grid$min_child_weight[row]
      ),
      data = dtrain,
      nrounds = 10/(tune_grid$eta[row]),
      verbose = 1
    )
    rm(dtrain)
    gc()
    pred_df <- train_set %>%
      filter(partition == group) %>%
      mutate(xgb = matrix(predict(xgb, newdata = as.matrix(select(., 
                                        launch_speed, launch_angle, spray_angle)), 
                                  type = "prob"),
                          ncol = 5, byrow = TRUE) %*% run_values$run_value) %>%
      select(run_value, xgb)
    
    preds = bind_rows(pred_df, preds)
    gc()
    
  }
  tune_grid$rmse[row] = sqrt(mean((preds$run_value - preds$xgb)^2))
  write.csv(tune_grid, "tune_rmse.csv", row.names = F)
  
}




#### test set ####





#### yoy desc/pred/stickiness w RV, xRV, savants xwoba ####



