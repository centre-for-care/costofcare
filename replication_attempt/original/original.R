# Written by Giacomo Vagni
# Oxford, 2020-2021
# CC BY-NC-SA

########################################################################################################################
########################################################################################################################

# Librairies you will need #
library(Synth)
library(MSCMT)
library(parallel)
library(tidyverse)
library(data.table)
#

########################################################################################################################
########################################################################################################################

# The sample dataset is a list of 2 treated (1 in each list) with a set of potential control cases.
# The following variables are in data:

# pidp = personal identifier
# treated = is the case treated
# treatement_period = the treatment period (dummy variable)
# timing_new # the timing relative to the treatment (-4,-3,-2,-1,0,1,2,3,4,...)
# year # the year of the survey
# age # the age of the respondent
# age left school # (fixed characteristics)
# earnings # the outcome (y)

# In this code, I will use the age of the respondent and the age left school as covariates in the Synthetic Control estimation.
# In our paper, we use more variable, but the routine takes significantly longer the more covariates you have.

#
load('./isc_sample')
#


# The treated case in list 1 is
isc_sample[[1]] %>% filter(treated == 1)
# there are 200 control cases #
n_distinct(isc_sample[[1]]$pidp)
#

# The treated case in list 2 is
isc_sample[[2]] %>% filter(treated == 1)
# there are 141 control cases #
n_distinct(isc_sample[[2]]$pidp)
#

########################################################################################################################
# Data preparation #
########################################################################################################################

# For this code to run, you need to prepare your dataset in a list, with 1 treated per list and the controls.
# There are several ways to construct potential treatment periods for the controls.
# The issue is that in the ISC, the treatment period varies with each treated case. Some mothers will have a child at 25 and others at 30, etc.

# The best way is to use the age and year of the survey of the treated case to determine potential treatment periods for the control cases.
# You take the treated case and look at what age she had a child, let's say 28 years old in 2000. Then you select as potential controls, all
# women who where 28 years old in 2000, and you attribute them a potential timing (-2 in 1998, -1 in 1999, 0 in 2000, etc).
# You need to determine the acceptable level of pre-treatment periods. We tested 8 pre-treatment periods, and 4 pre-treatment periods.
# Then you will only keep cases with that many pre-treatment period .

# In this sample, only the year and the number of pre-treatment periods (4 years prior birth) was used to create potential control treatment time.
# The age is then used in the Synthetic Control estimate as a covariate. This is a good solution for small samples.

########################################################################################################################
########################################################################################################################

# In this code, we have 3 main functions

# fsample_person_period -- a function to for re-sampling #
# f_summarise_controls -- a function for summarising the weighted control characteristics
# The main loop running the Synth Control routine (which can be wrapped in a function for convenience)

########################################################################################################################
########################################################################################################################

#
fsample_person_period = function(pdf_prep, k_controls){

  # We sample only control cases #
  # We use the function "sample_n" with replacement #
  sample_control = pdf_prep %>% filter(treated == 0) %>% select(pidp) %>% distinct %>% sample_n(k_controls, replace = T)
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
f_summarise_controls = function(x){ x %>%
    group_by(timing_new, n_controls) %>%

    summarise(

      sum_zero_wt = sum(mscmt_w == 0), # how many control cases with a weight of 0 #

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

############################################################
############################################################

# initialise the output list of the function where you will store the isc results
synth_w_df = list()
# initialise a list to save the original mscmt output
synth_obj_save = list()
#

# lenght of the loop, depends on the number of treated cases #
l = length(isc_sample)
#

# select the number of control re-sampling
r = 10 # number of bootstrap # 30-500 # can quickly take a very long time to run #
#

############################################################
# The main loop function
# we have i in 1:2 because we have 2 treated cases. You can adjust the lenght of the loop according the number of treated case you have #
############################################################

#
for(i in 1:2){

  tryCatch({ # try catching error, to avoid the loop to stop #
    print(i)

    # select case i #
    pdf_prep = isc_sample[[i]]

    # ------------------------ #
    # create an mscmt object #
    # ------------------------ #

    #
    pdf_prep$pid_char = as.character(pdf_prep$pidp)
    #

    # We need to create a new timing variable because mscmt has issues with timing starting at 0 #
    pdf_prep$timing_new2 = pdf_prep$timing_new + 1500 # we just add a random starting value to our original timing variable -- e.g. 1500
    #
    u = unique(pdf_prep$timing_new2[pdf_prep$treatment_period == 0 & pdf_prep$treated == 1] )
    #
    # for reference #
    table(pdf_prep$timing_new2, pdf_prep$timing_new)

    # the individual identifier of the treated case #
    p_id = unique(pdf_prep$pidp[pdf_prep$treated == 1] )
    # the individual identifiers of the control cases #
    c_id = unique(pdf_prep$pidp[pdf_prep$treated == 0] )
    #

    ########################################################################
    ########################################################################

    #
    pid_treated_df = data.frame(pidp = unique( pdf_prep$pidp[pdf_prep$treated == 1]) )
    pid_treated_df$pid_sampled = 1

    # vector of control id #
    control_id = pdf_prep$pidp[pdf_prep$treated == 0]
    control_id = unique(control_id)
    #

    # the number of controls that this treated case has #
    k_controls = length(control_id)
    print(k_controls)
    #

    ########################################################################
    ########################################################################

    # bootsrap here the number of controls #
    # r is the number of boostrap #
    g = lapply(1:r, function(p) fsample_person_period(pdf_prep, k_controls = k_controls)  )
    # we end up with a lists of bootstraped samples (g) #

    ########################################################################
    ########################################################################
    # MSCMT #
    ########################################################################
    ########################################################################

    #
    # msctm will processe ALL the variables in the dataset, so you need to subset which co-variates you want! #

    # we need the following variables in our example #
    pdf_mscmt = lapply(1:r, function(i) g[[i]] [,c('pid_char', 'pid_sampled', 'earnings', 'timing_new2', 'age_left_school', 'age')] )

    # lenght of boot #
    # r number of lists
    length(pdf_mscmt)
    #

    # We use mclapply for parallel computation, make sure to load the paralell package #
    # this case take very long depending on your dataset #
    # We create the mscmt object -- using its function listFromLong #
    # we use mclapply because we have to run the synthetic control in each of the bootstrap sample! #
    ms_df <- mclapply(1:r, function(i)
      listFromLong(pdf_mscmt[[i]], unit.variable="pid_sampled",
                   time.variable="timing_new2",
                   unit.names.variable="pid_char") )
    #

    # We create dependant variable with the periods we want #
    times.depms  <- cbind("earnings" = c(min(u),max(u)) )
    #

    # The predictors #
    times.predms <- cbind("earnings"          = c(min(u),max(u)),
                          "age"           = c(min(u),max(u)),
                          "age_left_school"   = c(min(u),max(u)))

    #

    ########################################################################
    # The synth control computation
    ########################################################################

    # Again mclapply to run the mscmt function #
    # for each boot samples #
    res <- mclapply(1:r, function(i)
      mscmt(ms_df[[i]], treatment.identifier = as.character(1),
            controls.identifier = as.character(2:(k_controls+1)),
            times.dep = times.depms, times.pred = times.predms, seed=1) )
    #

    ########################################################################
    ########################################################################

    # plot to inspect the result #
    plot(res[[1]], type = 'comparison')
    # the timing variable is here #
    table(pdf_prep$timing_new2, pdf_prep$timing_new)
    #

    # Save the result #
    synth_obj_save[[i]] = res
    #

    ####################################################################################
    ####################################################################################

    # Give a weight of 1 to the treated case #
    treated_w = data.frame(pidp = p_id, mscmt_w = 1)
    #

    # We then have the individual counterfactual dataset #
    dff_treated_w_full = merge( isc_sample[[i]] , treated_w, by = 'pidp') # pdf_full
    #

    setDT(dff_treated_w_full)
    setorder(dff_treated_w_full, pidp, timing_new, age)

    ####################################################################################
    ####################################################################################

    # We now retrieve the synthetic weights for the controls #
    synth_w = lapply(1:r, function(i) data.frame(pid_sampled = as.numeric(names(res[[i]]$w)), mscmt_w = res[[i]]$w, n_controls = length(unique(c_id)) ) )

    # merge back with the control characteristics
    dff_controls_w_full =  lapply(1:r, function(i) merge(synth_w[[i]], g[[i]], by = c('pid_sampled')) )
    #

    ####################################################################################
    # Re-weighting using the synthetic weights #
    ####################################################################################

    # retrieve the control averages #
    controls_avrFULL = lapply(1:r, function(i) dff_controls_w_full[[i]] %>% f_summarise_controls )

    ####################################################################################
    ####################################################################################

    # retrieve the treated case characteristics #
    treated_avrFULL = dff_treated_w_full %>% group_by(pidp, timing_new) %>%
      summarise(earnings = mean(earnings, na.rm = T),
                age = mean(age, na.rm = T),
                age_left_school = mean(age_left_school, na.rm = T),
                age = median(age),
                year = mean(year))
    #

    ####################################################################################
    ####################################################################################

    # merge the treated case and the controls
    all_synth_w_full =  lapply(1:r, function(i) merge(treated_avrFULL, controls_avrFULL[[i]], by = c('timing_new')) )
    #

    # we haev the individual counterfactual dataset here with the weighted and un-weighted controls, which will serve as a comparison for
    # how much did the synthetic control balanced the data
    all_synth_w_full

    ####################################################################################
    ####################################################################################

    # retrieve all the bootsrapped datasets #
    all_synth_w_full = rbindlist(all_synth_w_full, idcol = 'boot')
    #
    setDT(all_synth_w_full)
    #

    # create a post-treatmetn variable #
    all_synth_w_full[, post_treatment_period := ifelse(timing_new >= 0,1,0), pidp]

    # average individual causal effect - combining all the bootstrapped #
    # the bootstrapped values with correct the SE #
    all_synth_w_full %>% group_by(post_treatment_period) %>%
      summarise(earnings = mean(earnings), earnings_synth = mean(earnings_synth)) %>%
      mutate(diff = round(earnings - earnings_synth)) # the difference is the average individual causal effect #
    #

    ####################################################################################
    ####################################################################################

    # select the output #
    all_synth_w_full = all_synth_w_full %>% select(boot, pidp, age, timing_new, post_treatment_period, everything())

    # save #
    synth_w_df[[i]] = all_synth_w_full
  }, error=function(e){cat("treated id :",i, "error")} )
}
#

# We now have a list with individual causal effects for each treated case #
synth_w_df[[1]]
#

#
synth_w_df[[1]] %>% group_by(pidp, timing_new) %>%
  summarise(earnings = mean(earnings), synthetic = mean(earnings_synth)) %>% # average the bootsrap samples for each treated #
  ggplot() +
  geom_line(aes(timing_new, earnings, colour = 'observed')) +
  geom_line(aes(timing_new, synthetic, colour = 'synthetic'))
#

#
synth_w_df[[2]] %>% group_by(pidp, timing_new) %>%
  summarise(earnings = mean(earnings), synthetic = mean(earnings_synth)) %>% # average the bootsrap samples for each treated #
  ggplot() +
  geom_line(aes(timing_new, earnings, colour = 'observed')) +
  geom_line(aes(timing_new, synthetic, colour = 'synthetic'))
#

####################################################################################
####################################################################################

# function for computing the average causal effect for each treated #
avr_causal = function(x) x %>% group_by(post_treatment_period, timing_new) %>% summarise(earnings = mean(earnings), earnings_synth = mean(earnings_synth)) %>% mutate(diff = round(earnings - earnings_synth))
#

# number of treated cases #
l = 2
# check the individual effects #
lapply(1:l, function(i) synth_w_df[[i]] %>% avr_causal )
#

####################################################################################
####################################################################################

# put together all the treated cases and their respective bootstrap samples #
isc_df = bind_rows(synth_w_df)
#

# order the dataset
setorder(isc_df, pidp, boot, timing_new)
isc_df[1:20, ]
#

####################################################################################
####################################################################################

# individual average causal effect #
isc_df %>%
  mutate(diff = earnings - earnings_synth) %>%
  group_by(pidp, post_treatment_period) %>%
  summarise(m = round(mean(diff)) ) %>% as.data.frame()
#

# average causal effect #
isc_df %>%
  mutate(diff = earnings - earnings_synth) %>%
  group_by(pidp, post_treatment_period) %>%
  summarise(m = mean(diff)) %>%
  group_by(post_treatment_period) %>%
  summarise(round( mean(m) ))  %>% as.data.frame()
#

# average causal effect by treatment period #
isc_df %>%
  mutate(diff = earnings - earnings_synth) %>%
  group_by(pidp, timing_new, post_treatment_period) %>%
  summarise(m = mean(diff)) %>%
  group_by(timing_new, post_treatment_period) %>%
  summarise(round( mean(m) ))  %>% as.data.frame()
#

# average counterfactual penalties #
isc_df %>%
  group_by(pidp, post_treatment_period) %>%
  summarise(m = mean(earnings), s = mean(earnings_synth)) %>%
  group_by(post_treatment_period) %>%
  summarise(m = mean(m), s = mean(s)) %>% mutate( round ( (m - s)/s, 3) )
#

####################################################################################
# Check the balance of co-variates #
####################################################################################

#
isc_df %>% group_by(post_treatment_period) %>%
  summarise(age_avr = mean(age), age_synth_avr = mean(age_synth),
            age_left_school_avr = mean(age_left_school),
            age_left_school_synth_avr = mean(age_left_school_synth),
            year_avr = mean(year), year_synth_avr = mean(year_synth),
            mean(earnings), mean(earnings_synth)) %>% mutate_all(funs(round(.,1))) %>% as.data.frame()
#

####################################################################################
####################################################################################
