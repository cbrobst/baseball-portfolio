#### packages and initialization ####

#devtools::install_github(repo = "saberpowers/sabRmetrics")
library(sabRmetrics)
library(dplyr)
library(parallel)
setwd("baseball-portfolio")

ncores = detectCores()

#### download and clean batting data ####

for(y in 2016:2026){
  print(Sys.time())
  if(y < 2026){
    season = sabRmetrics::download_baseballsavant(start_date = as.Date(paste0(y,"-03-03")), 
                                                  end_date = as.Date(paste0(y,"-11-01"))) %>%
      select(game_id, year, event_index, pitch_number, bat_speed, swing_length, swing_path_tilt,
             launch_speed, launch_angle, delta_run_exp, attack_angle, attack_direction,
             expected_woba, iso_value, hit_coord_x, hit_coord_y, babip_value, events)
    
    if(ncores >= 10){
      print("first query done")
      cluster <- parallel::makeCluster(parallel::detectCores())
      season_pitch = sabRmetrics::download_statsapi(start_date = as.Date(paste0(y,"-03-03")), 
                                                    end_date = as.Date(paste0(y,"-11-01")),
                                                    cl = cluster) 
      parallel::stopCluster(cluster)
      season_pitch = season_pitch$pitch %>%
        select(game_id, event_index, pitch_number, play_id)#, hit_coord_x, hit_coord_y)
      season = inner_join(season, season_pitch, by = c("game_id","event_index","pitch_number"))
      
      season$video = sabRmetrics::get_video_url(season$play_id)
    }
    
    print("saving rds")
    
    saveRDS(season, paste0("savant_",y,".RDS"))
    rm(season)
    gc()
  } else {
    season = sabRmetrics::download_baseballsavant(start_date = as.Date(paste0(y,"-03-03")), 
                                                  end_date = as.Date(Sys.Date())) %>%
      select(game_id, year, event_index, pitch_number, bat_speed, swing_length, swing_path_tilt,
             launch_speed, launch_angle, delta_run_exp, attack_angle, attack_direction,
             expected_woba, iso_value, hit_coord_x, hit_coord_y, babip_value, events)
    
    if(ncores >= 10){
      print("first query done")
      cluster <- parallel::makeCluster(parallel::detectCores())
      season_pitch = sabRmetrics::download_statsapi(start_date = as.Date(paste0(y,"-03-03")), 
                                                    end_date = as.Date(Sys.Date()),
                                                    cl = cluster) 
      parallel::stopCluster(cluster)
      season_pitch = season_pitch$pitch %>%
        select(game_id, event_index, pitch_number, play_id)#, hit_coord_x, hit_coord_y)
      season = inner_join(season, season_pitch, by = c("game_id","event_index","pitch_number"))
      
      season$video = sabRmetrics::get_video_url(season$play_id)
    }
    
    print("saving rds")
    
    saveRDS(season, paste0("savant_",y,".RDS"))
    rm(season)
    gc()
  }
  
  print(paste0("done with "), y)
  rm(list = ls())
  gc()
}

#### load local data ####

if (file.exists("bip.RDS")) {
  bip = readRDS("bip.RDS")
} else {
  
  df = data.frame()
  
  for(y in 2016:2026){
    temp = readRDS(paste0("savant_",y,".RDS")) 
    df = bind_rows(temp, df)
  }
  
  bip = df %>% 
    filter(description == "hit_into_play")
  
  if(ncores >= 10){
    bip = bip %>% 
      filter(if_all(-video, ~ !is.na(.)))
  } else {
    bip = bip %>% 
      filter(if_all(everything(), ~ !is.na(.)))
  }
  
  print(paste0("nrow of bip: ", nrow(bip)))
  
  saveRDS(bip, "bip.RDS")
}



#### engineer more features ####

# fix hit coordinates
# https://rdrr.io/github/bdilday/GeomMLBStadiums/src/R/mlb_xy_transformation.R
# big thanks to bdilday for figuring out a reasonable conversion


bip = bip %>% mutate(hc_x = 2.495671*(hit_coord_x-125), 
                     hc_y = 2.495671*(199-hit_coord_y))

bip = bip %>% 
  mutate(spray_angle = atan2(hc_x, hc_y) * 180 / pi,
         landing_dist = sqrt(hc_x*hc_x + hc_y*hc_y))

bip$TB = bip$iso_value + ifelse(bip$iso_value == 3, 1, bip$babip_value)
bip %>% group_by(TB, events) %>% summarise(count = n())

#### get avg run values ####

run_values = bip %>% 
  group_by(TB) %>% 
  summarise(run_value = mean(delta_run_exp)) #, n_bip = n())

bip = inner_join(bip, run_values, by = "TB")

saveRDS(bip, "bip.RDS")
