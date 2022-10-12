library(Synth)
library(MSCMT)
library(parallel)
library(tidyverse)
library(data.table)

source('./replication_attempt/funcs.R')

load('./replication_attempt/data/isc_sample')

# The treated case in list 1 is
isc_sample[[1]] %>% filter(treated == 1)
n_distinct(isc_sample[[1]]$pidp)

# The treated case in list 2 is
isc_sample[[2]] %>% filter(treated == 1)
n_distinct(isc_sample[[2]]$pidp)


# lenght of the loop, depends on the number of treated cases #
l = length(isc_sample)

run <- function(r=50){
    # initialise the output list of the function where you will store the isc results
    synth_w_df = list()
    # initialise a list to save the original mscmt output
    synth_obj_save = list()
    #
    for(i in 1:2){
                                        # try catching error, to avoid the loop to stop #
            print(i)

                                        # select case i #
            pdf_prep = isc_sample[[i]]
            pdf_prep$pid_char = as.character(pdf_prep$pidp)
            pdf_prep$timing_new2 = pdf_prep$timing_new + 1500 # 1500 random int, replace by argument of randomint func
            
            u = unique(pdf_prep$timing_new2[pdf_prep$treatment_period == 0 & pdf_prep$treated == 1] )

            #table(pdf_prep$timing_new2, pdf_prep$timing_new)

            p_id = unique(pdf_prep$pidp[pdf_prep$treated == 1] )

            c_id = unique(pdf_prep$pidp[pdf_prep$treated == 0] )
            
            pid_treated_df = data.frame(pidp = unique( pdf_prep$pidp[pdf_prep$treated == 1]) )
            pid_treated_df$pid_sampled = 1

            control_id = pdf_prep$pidp[pdf_prep$treated == 0]
            control_id = unique(control_id)

            k_controls = length(control_id)
            #print(k_controls)

            g = lapply(1:r, function(p) fsample_person_period(pdf_prep,
                                                              pid_treated_df = pid_treated_df,
                                                              k_controls = k_controls)  )
                                        # we end up with a lists of bootstraped samples (g) #

            pdf_mscmt = lapply(1:r, function(i) g[[i]] [,c('pid_char',
                                                           'pid_sampled',
                                                           'earnings',
                                                           'timing_new2',
                                                           'age_left_school',
                                                           'age')] )

            #length(pdf_mscmt)

            ms_df <- mclapply(1:r, function(i)
                listFromLong(pdf_mscmt[[i]], unit.variable="pid_sampled",
                             time.variable="timing_new2",
                             unit.names.variable="pid_char") )

            times.depms  <- cbind("earnings" = c(min(u),max(u)) )

            times.predms <- cbind("earnings"          = c(min(u),max(u)),
                                  "age"           = c(min(u),max(u)),
                                  "age_left_school"   = c(min(u),max(u)))


########################################################################
                                        # The synth control computation
########################################################################

            res <- mclapply(1:r, function(i)
                mscmt(ms_df[[i]], treatment.identifier = as.character(1),
                      controls.identifier = as.character(2:(k_controls+1)),
                      times.dep = times.depms, times.pred = times.predms, seed=1) )

            #plot(res[[1]], type = 'comparison')

            #table(pdf_prep$timing_new2, pdf_prep$timing_new)

            synth_obj_save[[i]] = res

            treated_w = data.frame(pidp = p_id, mscmt_w = 1)

            dff_treated_w_full = merge( isc_sample[[i]] , treated_w, by = 'pidp')

            setDT(dff_treated_w_full)
            setorder(dff_treated_w_full, pidp, timing_new, age)

            synth_w = lapply(1:r, function(i) data.frame(pid_sampled = as.numeric(names(res[[i]]$w)),
                                                         mscmt_w = res[[i]]$w, n_controls = length(unique(c_id)) ) )

            dff_controls_w_full =  lapply(1:r, function(i) merge(synth_w[[i]], g[[i]], by = c('pid_sampled')) )

            controls_avrFULL = lapply(1:r, function(i) dff_controls_w_full[[i]] %>% f_summarise_controls )

            treated_avrFULL = dff_treated_w_full %>% group_by(pidp, timing_new) %>%
                summarise(earnings = mean(earnings, na.rm = T),
                          age = mean(age, na.rm = T),
                          age_left_school = mean(age_left_school, na.rm = T),
                          age = median(age),
                          year = mean(year))

            all_synth_w_full =  lapply(1:r, function(i) merge(treated_avrFULL, controls_avrFULL[[i]], by = c('timing_new')) )

            #all_synth_w_full

            all_synth_w_full = rbindlist(all_synth_w_full, idcol = 'boot')
            setDT(all_synth_w_full)

            all_synth_w_full[, post_treatment_period := ifelse(timing_new >= 0,1,0), pidp]

            all_synth_w_full %>% group_by(post_treatment_period) %>%
                summarise(earnings = mean(earnings), earnings_synth = mean(earnings_synth)) %>%
                mutate(diff = round(earnings - earnings_synth)) # the difference is the average individual causal effect #

            all_synth_w_full = all_synth_w_full %>% select(boot, pidp, age, timing_new, post_treatment_period, everything())
            synth_w_df[[i]] = all_synth_w_full
        }
    return(synth_w_df)
}

synth_w_df <- run(r=50)


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
ggsave('./replication_attempt/figure_1.png')
#

#
synth_w_df[[2]] %>% group_by(pidp, timing_new) %>%
  summarise(earnings = mean(earnings), synthetic = mean(earnings_synth)) %>% # average the bootsrap samples for each treated #
  ggplot() +
  geom_line(aes(timing_new, earnings, colour = 'observed')) +
  geom_line(aes(timing_new, synthetic, colour = 'synthetic'))
ggsave('./replication_attempt/figure_2.png')
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
