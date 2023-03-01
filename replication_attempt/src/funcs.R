library(Synth)
library(MSCMT)
library(parallel)
library(tidyverse)
library(data.table)

set.seed(42) # setting global initial state for reproducibility

fsample_person_period = function(pdf_prep, pid_treated_df, k_controls){

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


####
#Individual synthetic control calculation
####

ics <- function(ms_df=ms_df,
                k_controls=k_controls,
                times.depms=times.depms,
                times.predms=times.predms,
                synth_obj_save=synth_obj_save,
                sample=isc_sample,
                n_control=c_id,
                g=g)
{

    res <- mclapply(1:r, function(i)
        mscmt(ms_df[[i]], treatment.identifier = as.character(1),
              controls.identifier = as.character(2:(k_controls+1)),
              times.dep = times.depms, times.pred = times.predms, seed=1) )

    synth_obj_save[[i]] = res
    treated_w = data.frame(pidp = p_id, mscmt_w = 1)
    dff_treated_w_full = merge( isc_sample[[i]] , treated_w, by = 'pidp')
    setDT(dff_treated_w_full)
    setorder(dff_treated_w_full, pidp, timing_new, age)
    synth_w = lapply(1:r, function(i) data.frame(pid_sampled = as.numeric(names(res[[i]]$w)),
                                                 mscmt_w = res[[i]]$w, n_controls = length(unique(c_id)) ) )

    dff_controls_w_full = lapply(1:r, function(i) merge(synth_w[[i]], g[[i]], by = c('pid_sampled')) )
    controls_avrFULL = lapply(1:r, function(i) dff_controls_w_full[[i]] %>% f_summarise_controls )

    treated_avrFULL = dff_treated_w_full %>% group_by(pidp, timing_new) %>%
        summarise(earnings = mean(earnings, na.rm = T),
                  age = mean(age, na.rm = T),
                  age_left_school = mean(age_left_school, na.rm = T),
                  age = median(age),
                  year = mean(year))
    
    all_synth_w_full =  lapply(1:r, function(i) merge(treated_avrFULL, controls_avrFULL[[i]], by = c('timing_new')) )
    all_synth_w_full = rbindlist(all_synth_w_full, idcol = 'boot')
    setDT(all_synth_w_full)

    all_synth_w_full[, post_treatment_period := ifelse(timing_new >= 0,1,0), pidp]

    all_synth_w_full %>% group_by(post_treatment_period) %>%
        summarise(earnings = mean(earnings), earnings_synth = mean(earnings_synth)) %>%
        mutate(diff = round(earnings - earnings_synth)) # the difference is the average individual causal effect #
    all_synth_w_full = all_synth_w_full %>% select(boot, pidp, age, timing_new, post_treatment_period, everything())
    synth_w_df[[i]] = all_synth_w_full
}
