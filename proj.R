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

bip = readRDS("bip.RDS")
run_values = readRDS("rv.RDS")
run_values$woba = c(0, 0.890,	1.261,	1.596,	2.049	)
# woba weights for comparison to savant model
# https://www.fangraphs.com/tools/guts?type=cn

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
            GB = mean((launch_angle < 10)),
            EV90 = quantile(launch_speed, 0.90),
            LA1030 = mean(ifelse(launch_angle > 10, ifelse(launch_angle < 30, 1, 0), 0))
  )

saveRDS(player_season, "player_season.RDS")

rm(bip)
gc()

player_season = readRDS("player_season.RDS")

#### regression target model ####





#### aggregated, downweighted/decayed data ####





#### blending the past data with the skills model ####





#### aging curve ####





#### in-sample outputs ####




