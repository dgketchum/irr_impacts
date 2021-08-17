#!/usr/bin/env Rscript

# Title     : TODO
# Objective : TODO
# Created by: dgketchum
# Created on: 6/25/21

library(MASS)

# df = read.csv(file = '/media/research/IrrigationGIS/gages/merged_q_ee/12352500.csv')

aic_glm = function(file){
  df = read.csv(file)
  
  t = 1: nrow(df)
  q = df$q
  qb = df$qb
  cc = df$cc
  irr = df$irr
  ppt = df$ppt
  etr = df$etr
  ppt_lt = df$ppt_lt
  etr_lt = df$etr_lt

  response = qb
  
  #full = glm(response ~ cc + irr + pr + etr + cc:pr, family = Gamma(link = 'log'))
  full = lm(response ~ cc + irr + ppt + etr + ppt_lt + etr_lt)

  summary(full)
  
  model = stepAIC(full, direction = 'both')
  
  summary = summary(model)
  
  plot(response, predict(model, type = 'response'))
  abline(0,1)
  
  return(as.data.frame(summary$coefficients))
}

# aic_glm('/media/research/IrrigationGIS/gages/merged_q_ee/12352500.csv')
