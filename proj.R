#### packages and initialization ####

library(caret)
library(xgboost)
library(dplyr)
library(parallel)
library(ggplot2)
library(pdp)
library(readr)
library(mgcv)
library(splines)
library(scam)
library(MetricsWeighted)
setwd("baseball-portfolio")

#### set up player-season data from event level + modeled columns ####

run_values = readRDS("rv.RDS")
run_values$woba = c(0, 0.890,	1.261,	1.596,	2.049	)
# woba weights for comparison to savant model
# https://www.fangraphs.com/tools/guts?type=cn


if (file.exists("player_season.RDS")) {
  player_season = readRDS("player_season.RDS")
} else {
  bip = readRDS("bip.RDS")
  xgb = readRDS("xgb_xcrv.RDS")
  
  bip = bip %>% 
    mutate(xrv = matrix(predict(xgb, newdata = as.matrix(select(., 
                                launch_speed, launch_angle, spray_angle)), 
                                type = "prob"),
                        ncol = 5, byrow = TRUE) %*% run_values$run_value,
           xwobacon = matrix(predict(xgb, newdata = as.matrix(select(., 
                                     launch_speed, launch_angle, spray_angle)), 
                                     type = "prob"),
                             ncol = 5, byrow = TRUE) %*% run_values$woba)
  
  
  ev_lm = lm(run_value~launch_speed, bip)
  ev_la_model = mgcv::gam(run_value~te(launch_speed, launch_angle), data = bip)
  
  bip$qc_rv = (predict(ev_lm, bip)+predict(ev_la_model, bip))/2
  
  
  
  player_season = bip %>% group_by(batter_id, batter_name, year, age_bat) %>%
    summarise(bip = n(), 
              my_xwobacon = mean(xwobacon),
              savant_xwobacon = mean(expected_woba),
              my_xrv = mean(xrv),
              actual_rv = mean(run_value),
              QC = mean(qc_rv),
              EV = mean(launch_speed),
              bat_speed = mean(bat_speed, na.rm = T),
              GB = mean((launch_angle < 10)),
              EV90 = quantile(launch_speed, 0.90),
              LA1030 = mean(ifelse(launch_angle > 10, ifelse(launch_angle < 30, 1, 0), 0))
    )
  
  saveRDS(player_season, "player_season.RDS")
  
  rm(bip)
  gc()
}

#### generate decayed/downweighted, aggregated data ####

future = player_season %>% mutate(age = age_bat, future_season = year) %>%
  ungroup() %>% # player_season was a grouped df
  select(batter_id, batter_name, future_season, age, bip, actual_rv) %>%
  filter(bip > 99) # arbitrary
# but reflective of the fact that we want to predict large/stable sample outcomes
# even if the inputs are small/unstable samples

df = future %>% left_join(player_season, by = c("batter_id","batter_name"), 
                          suffix = c("_future","_past")) %>%
  filter(year < future_season, year >= future_season - 3) %>%
  group_by(batter_id, batter_name, future_season, age, bip_future, actual_rv_future) %>%
  summarise(bip30 = sum(bip_past * 0.30^(future_season - year)),
            bip40 = sum(bip_past * 0.40^(future_season - year)),
            bip50 = sum(bip_past * 0.50^(future_season - year)),
            bip60 = sum(bip_past * 0.60^(future_season - year)),
            bip70 = sum(bip_past * 0.70^(future_season - year)),
            bip80 = sum(bip_past * 0.80^(future_season - year)),
            bip90 = sum(bip_past * 0.90^(future_season - year)),
            my_xwobacon30 = weighted.mean(my_xwobacon,bip_past * 0.30^(future_season - year)),
            my_xwobacon40 = weighted.mean(my_xwobacon,bip_past * 0.40^(future_season - year)),
            my_xwobacon50 = weighted.mean(my_xwobacon,bip_past * 0.50^(future_season - year)),
            my_xwobacon60 = weighted.mean(my_xwobacon,bip_past * 0.60^(future_season - year)),
            my_xwobacon70 = weighted.mean(my_xwobacon,bip_past * 0.70^(future_season - year)),
            my_xwobacon80 = weighted.mean(my_xwobacon,bip_past * 0.80^(future_season - year)),
            my_xwobacon90 = weighted.mean(my_xwobacon,bip_past * 0.90^(future_season - year)),
            savant_xwobacon30 = weighted.mean(savant_xwobacon,bip_past * 0.30^(future_season - year)),
            savant_xwobacon40 = weighted.mean(savant_xwobacon,bip_past * 0.40^(future_season - year)),
            savant_xwobacon50 = weighted.mean(savant_xwobacon,bip_past * 0.50^(future_season - year)),
            savant_xwobacon60 = weighted.mean(savant_xwobacon,bip_past * 0.60^(future_season - year)),
            savant_xwobacon70 = weighted.mean(savant_xwobacon,bip_past * 0.70^(future_season - year)),
            savant_xwobacon80 = weighted.mean(savant_xwobacon,bip_past * 0.80^(future_season - year)),
            savant_xwobacon90 = weighted.mean(savant_xwobacon,bip_past * 0.90^(future_season - year)),
            my_xrv30 = weighted.mean(my_xrv,bip_past * 0.30^(future_season - year)),
            my_xrv40 = weighted.mean(my_xrv,bip_past * 0.40^(future_season - year)),
            my_xrv50 = weighted.mean(my_xrv,bip_past * 0.50^(future_season - year)),
            my_xrv60 = weighted.mean(my_xrv,bip_past * 0.60^(future_season - year)),
            my_xrv70 = weighted.mean(my_xrv,bip_past * 0.70^(future_season - year)),
            my_xrv80 = weighted.mean(my_xrv,bip_past * 0.80^(future_season - year)),
            my_xrv90 = weighted.mean(my_xrv,bip_past * 0.90^(future_season - year)),
            actual_rv_past30 = weighted.mean(actual_rv_past,bip_past * 0.30^(future_season - year)),
            actual_rv_past40 = weighted.mean(actual_rv_past,bip_past * 0.40^(future_season - year)),
            actual_rv_past50 = weighted.mean(actual_rv_past,bip_past * 0.50^(future_season - year)),
            actual_rv_past60 = weighted.mean(actual_rv_past,bip_past * 0.60^(future_season - year)),
            actual_rv_past70 = weighted.mean(actual_rv_past,bip_past * 0.70^(future_season - year)),
            actual_rv_past80 = weighted.mean(actual_rv_past,bip_past * 0.80^(future_season - year)),
            actual_rv_past90 = weighted.mean(actual_rv_past,bip_past * 0.90^(future_season - year)),
            QC30 = weighted.mean(QC,bip_past * 0.30^(future_season - year)),
            QC40 = weighted.mean(QC,bip_past * 0.40^(future_season - year)),
            QC50 = weighted.mean(QC,bip_past * 0.50^(future_season - year)),
            QC60 = weighted.mean(QC,bip_past * 0.60^(future_season - year)),
            QC70 = weighted.mean(QC,bip_past * 0.70^(future_season - year)),
            QC80 = weighted.mean(QC,bip_past * 0.80^(future_season - year)),
            QC90 = weighted.mean(QC,bip_past * 0.90^(future_season - year)),
            EV30 = weighted.mean(EV,bip_past * 0.30^(future_season - year)),
            EV40 = weighted.mean(EV,bip_past * 0.40^(future_season - year)),
            EV50 = weighted.mean(EV,bip_past * 0.50^(future_season - year)),
            EV60 = weighted.mean(EV,bip_past * 0.60^(future_season - year)),
            EV70 = weighted.mean(EV,bip_past * 0.70^(future_season - year)),
            EV80 = weighted.mean(EV,bip_past * 0.80^(future_season - year)),
            EV90 = weighted.mean(EV,bip_past * 0.90^(future_season - year)),
            bat_speed30 = weighted.mean(bat_speed,bip_past * 0.30^(future_season - year)),
            bat_speed40 = weighted.mean(bat_speed,bip_past * 0.40^(future_season - year)),
            bat_speed50 = weighted.mean(bat_speed,bip_past * 0.50^(future_season - year)),
            bat_speed60 = weighted.mean(bat_speed,bip_past * 0.60^(future_season - year)),
            bat_speed70 = weighted.mean(bat_speed,bip_past * 0.70^(future_season - year)),
            bat_speed80 = weighted.mean(bat_speed,bip_past * 0.80^(future_season - year)),
            bat_speed90 = weighted.mean(bat_speed,bip_past * 0.90^(future_season - year)),
            GB30 = weighted.mean(GB,bip_past * 0.30^(future_season - year)),
            GB40 = weighted.mean(GB,bip_past * 0.40^(future_season - year)),
            GB50 = weighted.mean(GB,bip_past * 0.50^(future_season - year)),
            GB60 = weighted.mean(GB,bip_past * 0.60^(future_season - year)),
            GB70 = weighted.mean(GB,bip_past * 0.70^(future_season - year)),
            GB80 = weighted.mean(GB,bip_past * 0.80^(future_season - year)),
            GB90 = weighted.mean(GB,bip_past * 0.90^(future_season - year))
            # going to omit EV90, LA1030
            
            ) %>% ungroup()


#### regression target model ####

set.seed(2005)
df$partition = sample(1:10,nrow(df), replace = T)

df %>% group_by(partition) %>% summarise(count = n())


features <- c(
  "my_xwobacon30", "my_xwobacon40", "my_xwobacon50",
  "my_xwobacon60", "my_xwobacon70", "my_xwobacon80", "my_xwobacon90",
  "savant_xwobacon30", "savant_xwobacon40", "savant_xwobacon50",
  "savant_xwobacon60", "savant_xwobacon70", "savant_xwobacon80", "savant_xwobacon90",
  "my_xrv30", "my_xrv40", "my_xrv50", "my_xrv60", "my_xrv70", "my_xrv80", "my_xrv90",
  "QC30", "QC40", "QC50", "QC60", "QC70", "QC80", "QC90",
  "EV30", "EV40", "EV50", "EV60", "EV70", "EV80", "EV90",
  "bat_speed30", "bat_speed40", "bat_speed50", "bat_speed60",
  "bat_speed70", "bat_speed80", "bat_speed90",
  "GB30", "GB40", "GB50", "GB60", "GB70", "GB80", "GB90"
)

formulas <- sapply(features, function(x) {
  sprintf(
    "actual_rv_future ~ ns(%s, knots = c(mean(%s)), Boundary.knots = c(quantile(%s, 0.25), quantile(%s, 0.75)))",
    x, x, x, x
  )
})

formulas <- unname(formulas)

tune_grid = data.frame(formulas = formulas, rmse = rep(NA, length(formulas)))

for(row in seq_len(nrow(tune_grid))){
  if(!is.na(tune_grid$rmse[row])){next}
  
  print(paste0("beginning tuning grid row ", row, " out of ", nrow(tune_grid)))
  print(Sys.time())
  preds = data.frame()
  
  for(group in 1:5){
    model = lm(formula = tune_grid$formulas[row], data = df %>% filter(partition != group))
    pred_df <- df %>% filter(partition == group) %>% 
      mutate(pred = predict(model, .)) %>%  select(pred, actual_rv_future)
    
    preds = bind_rows(pred_df, preds)
    gc()
    
  }
  tune_grid$rmse[row] = sqrt(mean((preds$actual_rv_future - preds$pred)^2))

}

f = tune_grid$formulas[which.min(tune_grid$rmse)]
# > f
# [1] "actual_rv_future ~ ns(QC30, knots = c(mean(QC30)), 
#      Boundary.knots = c(quantile(QC30, 0.25), quantile(QC30, 0.75)))"

target_model = lm(formula = f, data = df)
pdp::partial(target_model, "QC30", plot = T)

df$reg_target = predict(target_model, df)


#### aggregated, downweighted/decayed data ####

for (x in seq(30, 90, 10)) {
  feature <- paste0("actual_rv_past", x)
  rsq <- summary(lm(df$actual_rv_future ~ df[[feature]]))$r.squared
  print(paste0(feature, ": ", rsq))
}

# [1] "actual_rv_past30: 0.134414891348586"
# [1] "actual_rv_past40: 0.134787264514618"
# [1] "actual_rv_past50: 0.134724306299944"
# [1] "actual_rv_past60: 0.134374515552707"
# [1] "actual_rv_past70: 0.133825076800551"
# [1] "actual_rv_past80: 0.133133041087461"
# [1] "actual_rv_past90: 0.132338369839441"

#### blending the past data with the skills model ####

tune_grid = data.frame(weight = seq(10,2000, 10),
                       rsq = NA_real_)


for(w in tune_grid$weight){
  df$crv_regressed = (df$actual_rv_past40*df$bip40 + df$reg_target*w)/(w+df$bip40)
  tune_grid$rsq[tune_grid$weight == w] = 
    MetricsWeighted::weighted_cor(df$actual_rv_future, df$crv_regressed, df$bip_future)
  
}

weight = tune_grid$weight[which.max(tune_grid$rsq)] # 690
df$crv_regressed = (df$actual_rv_past40*df$bip40 + df$reg_target*weight)/(weight+df$bip40)


#### aging curve ####





#### in-sample outputs ####







#### transforming to bayesian to get median projections ####







