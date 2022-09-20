library(Synth)
library(MSCMT)
library(parallel)
library(tidyverse)
library(data.table)


fsample_person_period = function(pdf_prep, k_controls){

  # We sample only control cases #
  # We use the function "sample_n" with replacement #
    sample_control = pdf_prep %>%
        filter(treated == 0) %>%
        select(pidp) %>%
        distinct %>%
        sample_n(k_controls, replace = T)
  #

  # We create fake pid using #
  sample_control$pid_sampled = 2:(nrow(sample_control)+1)

  # We now bind with treated case unit #
  samp = rbind(pid_treated_df, sample_control)

  # We merge #
  df_treat_sampled = suppressMessages(left_join(samp, pdf_prep))
  #
  df_treat_sampled$pid_char = as.character(df_treat_sampled$pid_sampled)
  #
  df_treat_sampled
  #
  return(df_treat_sampled)
}


# We create a function that will re-weight / balance the control cases characteristics based on their synthetic weights #
f_summarise_controls = function(x){
    x %>%
    group_by(timing_new, n_controls) %>%

    summarise(sum_zero_wt = sum(mscmt_w == 0), # how many control cases with a weight of 0 #

      earnings_unwt = mean(earnings, na.rm = T), # we compute the average unweighted mean of the controls in order to compare the improvement of the SC #
      earnings_synth = earnings %*% mscmt_w, # the weighted earnings #

      # same thing wit the other co-variates #
      age_unwt = mean(age, na.rm = T),
      age_synth = age %*% mscmt_w,

      age_left_school_unwt = mean(age_left_school, na.rm = T),
      age_left_school_synth = age_left_school %*% mscmt_w,

      year_unwt = mean(year, na.rm = T),
      year_synth = year %*% mscmt_w)
}
