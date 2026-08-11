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
  y = bip$run_value,
  plot = "pairs"
)


featurePlot(
  x = bip[, c("launch_speed", "launch_angle", "spray_angle","hc_x", "hc_y",
              "bat_speed","attack_angle","swing_path_tilt")],
  y = bip$run_value,
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
  gam_model <- gam( run_value ~ s(launch_speed) + s(launch_angle) + s(spray_angle), 
                    data = train_set %>% filter(partition != group), method = "REML" )
  rf = randomForest(x = train_set[train_set$partition != group, 
                                  c("launch_speed", "launch_angle", "spray_angle")],
                    y = train_set[train_set$partition != group,]$run_value,
                    ntree = 500, mtry = 2, importance = F)
  train = train_set %>% filter(partition != group)
  xgb = xgboost(x = as.matrix(train[, c("launch_speed","launch_angle","spray_angle")]),
                y = train$run_value,
                objective = "reg:squarederror",
                nrounds = 500,
                max_depth = 3,
                eta = 0.05,
                subsample = 0.8,
                colsample_bytree = 1,
                verbose = 0  )
  rm(train)
  gc()
  print("done training this partition, beginning predictions")
  
  pred_df <- train_set %>%
    filter(partition == group) %>%
    mutate(ols = predict(ols, newdata = .),
           ols_int = predict(ols_interaction, newdata = .),
           gam = predict(gam_model, newdata = .),
           rf = predict(rf, newdata = select(., launch_speed, launch_angle, spray_angle)),
           xgb = predict(xgb, newdata = as.matrix(select(., 
                              launch_speed, launch_angle, spray_angle)))) %>%
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



