#### packages and initialization ####

library(caret)
library(xgboost)
library(dplyr)
library(parallel)
library(ggplot2)
library(pdp)
library(readr)
library(mgcv)
library(randomForest)
library(MetricsWeighted)
setwd("baseball-portfolio")

ncores = detectCores()

bip = readRDS("bip.RDS")
run_values <- readRDS("rv.RDS")


test_set = bip %>% filter(year == 2026)
train_set = bip %>% filter(year < 2026)


rm(bip)
gc() # just bc i am working with so little processing power here

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

#### model selection ####


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


if (file.exists("tune_rmse.csv")) {
  tune_grid <- read_csv("tune_rmse.csv")
} else {

  tune_grid = expand.grid(
    max_depth = c(2, 3, 4, 5),
    min_child_weight = c(1, 5, 10, 20),
    eta = c(0.025, 0.05, 0.1, 0.20),
    subsample = c(0.7, 0.85, 1),
    gamma = c(0, 0.5, 1)
  )
  tune_grid$row_num = seq_len(nrow(tune_grid))
  tune_grid$rmse = rep(NA, nrow(tune_grid))

}

set.seed(2005)
train_set$partition = sample(1:5,nrow(train_set), replace = T)
# larger search space, fewer folds will train faster
# i prefer 10, but i cannot let this run for 100+ hours nonstop

train_set %>% group_by(partition) %>% summarise(count = n())

for(row in seq_len(nrow(tune_grid))){
  if(!is.na(tune_grid$rmse[row])){next}
  
  print(paste0("beginning tuning grid row ", row, " out of ", nrow(tune_grid)))
  print(Sys.time())
  preds = data.frame()
  
  for(group in 1:5){
    dtrain <- xgb.DMatrix(
      data = as.matrix((train_set %>% filter(partition != group))[,c("launch_speed", "launch_angle", "spray_angle")]),
      label = as.integer((train_set %>% filter(partition != group))$TB)
    )
    dvalid <- xgb.DMatrix(
      data = as.matrix((train_set %>% filter(partition == group))[,c("launch_speed", "launch_angle", "spray_angle")]),
      label = as.integer((train_set %>% filter(partition == group))$TB)
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
      watchlist = list(
        train = dtrain,
        eval = dvalid
      ),
      early_stopping_rounds = 50,
      verbose = 0
    )
    rm(dtrain)
    rm(dvalid)
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

#### final model ####

best_tune = tune_grid[which.min(tune_grid$rmse),]

dtrain <- xgb.DMatrix(
  data = as.matrix(train_set[,c("launch_speed", "launch_angle", "spray_angle")]),
  label = as.integer(train_set$TB)
)
xgb <- xgb.train(
  params = list(
    objective = "multi:softprob",
    num_class = 5,
    max_depth = best_tune$max_depth[1],
    eta = best_tune$eta[1],
    subsample = best_tune$subsample[1],
    colsample_bytree = 1,
    gamma = best_tune$gamma[1],
    min_child_weight = best_tune$min_child_weight[1]
  ),
  data = dtrain,
  nrounds = 10/(best_tune$eta[1]),
  verbose = 0
)
rm(dtrain)
gc()

saveRDS(xgb, "xgb_xcrv.RDS")

#### test set ####

# woba weights for comparison to savant model
# https://www.fangraphs.com/tools/guts?type=cn
run_values$woba = c(0, 0.890,	1.261,	1.596,	2.049	)

test_set <- test_set %>%
  mutate(xrv = matrix(predict(xgb, newdata = as.matrix(select(., 
                                  launch_speed, launch_angle, spray_angle)), 
                              type = "prob"),
                      ncol = 5, byrow = TRUE) %*% run_values$run_value,
         xwobacon = matrix(predict(xgb, newdata = as.matrix(select(., 
                                   launch_speed, launch_angle, spray_angle)), 
                                   type = "prob"),
                           ncol = 5, byrow = TRUE) %*% run_values$woba)


# batted ball rmse
# player-season rmse

RMSE(test_set$xrv, test_set$run_value)
cor(test_set$expected_woba, test_set$run_value)^2
cor(test_set$xwobacon, test_set$run_value)^2
# our new model is much more descriptive than savant, measured by Rsq


#### yoy desc/pred/stickiness w RV, xRV, savants xwoba ####

all_bip = bind_rows(train_set, test_set) %>%
  mutate(xrv = matrix(predict(xgb, newdata = as.matrix(select(., 
                                   launch_speed, launch_angle, spray_angle)), 
                              type = "prob"),
                      ncol = 5, byrow = TRUE) %*% run_values$run_value,
         xwobacon = matrix(predict(xgb, newdata = as.matrix(select(., 
                                        launch_speed, launch_angle, spray_angle)), 
                                   type = "prob"),
                           ncol = 5, byrow = TRUE) %*% run_values$woba)


RMSE(all_bip$xrv, all_bip$run_value)
cor(all_bip$expected_woba, all_bip$run_value)^2
cor(all_bip$xwobacon, all_bip$run_value)^2

player_season = all_bip %>% group_by(batter_id, batter_name, year) %>%
  summarise(bip = n(), 
            my_xwobacon = mean(xwobacon),
            savant_xwobacon = mean(expected_woba),
            my_xrv = mean(xrv),
            actual_rv = mean(run_value),
            EV = mean(launch_speed),
            GB = mean(ifelse(launch_angle < 10, 1, 0))
            )

yoy = inner_join(player_season, player_season %>% mutate(year = year - 1),
                 by = c("batter_id","batter_name","year"), suffix = c("_pre","_post"))

weighted_cor(yoy$my_xwobacon_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$savant_xwobacon_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$my_xrv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$actual_rv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2


weighted_cor(yoy$my_xwobacon_pre, yoy$my_xwobacon_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$savant_xwobacon_pre, yoy$savant_xwobacon_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$my_xrv_pre, yoy$my_xrv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$actual_rv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2

#### developing an underfit metric for better future season predictiveness ####

ev_lm = lm(run_value~launch_speed, all_bip)
ev_la_model = lm(run_value~launch_speed*poly(launch_angle,2), all_bip)

all_bip$qc_rv = (predict(ev_lm, all_bip)+predict(ev_la_model, all_bip))/2



player_season = all_bip %>% group_by(batter_id, batter_name, year) %>%
  summarise(bip = n(), 
            my_xwobacon = mean(xwobacon),
            savant_xwobacon = mean(expected_woba),
            my_xrv = mean(xrv),
            actual_rv = mean(run_value),
            QC = mean(qc_rv),
            EV = mean(launch_speed),
            GB = mean(ifelse(launch_angle < 10, 1, 0))
  )

yoy = inner_join(player_season, player_season %>% mutate(year = year - 1),
                 by = c("batter_id","batter_name","year"), suffix = c("_pre","_post"))

weighted_cor(yoy$my_xwobacon_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$savant_xwobacon_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$my_xrv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$actual_rv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$QC_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2


weighted_cor(yoy$my_xwobacon_pre, yoy$my_xwobacon_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$savant_xwobacon_pre, yoy$savant_xwobacon_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$my_xrv_pre, yoy$my_xrv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$actual_rv_pre, yoy$actual_rv_post, pmin(yoy$bip_pre, yoy$bip_post))^2
weighted_cor(yoy$QC_pre, yoy$QC_post, pmin(yoy$bip_pre, yoy$bip_post))^2

