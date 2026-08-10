#### packages and initialization ####

library(caret)
library(xgboost)
library(dplyr)
library(parallel)
setwd("baseball-portfolio")

ncores = detectCores()



#### begin modeling ####

bip = readRDS("bip.RDS")

test_set = bip %>% filter(year == 2026)
train_set = bip %>% filter(year < 2026)

set.seed(2005)
train_set$partition = sample(1:10,nrow(train_set), replace = T)

train_set %>% group_by(partition) %>% summarise(count = n())

for(group in 1:10){
  
  
  
  
}


