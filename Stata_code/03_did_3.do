***********************ANALYSIS*************************************************
********************************************************************************

clear all

//Maria work laptop
global username "`c(username)'"
dis "$username" // Displays your user name on your computer

if "$username" == "if1mp" { 
global data 	"C:\Users\if1mp\Desktop\project2"
global output 	"C:\Users\if1mp\Desktop\project2"
global log 		"C:\Users\if1mp\Desktop\project2"
global graph  	"C:\Users\if1mp\Desktop\project2"
global table 	"C:\Users\if1mp\Desktop\project2"

}


use "$data\generated_new"
format pidp %10.0f
duplicates tag pidp year, generate(duple)
xtset pidp year
*keep if reindex>=-5
*drop if reindex>5


 *****INDIVIDUAL INCOME
  *low_intensity *****************************************************************
 csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_low_intensity_ok) method(drimp )
  estat event, wboot estore(event1)  window(-8 6)
 csdid_plot, name(m1,replace) title("Low Intensity") graphregion(fcolor(white))
 * Estimates the chi2 statistic of the null hypothesis that ALL pretreatment ATTGT's are statistically equal to zero.
estat pretrend
  * Estimates the chi2 statistic of the null hypothesis that ALL pretreatment ATTGT's within window are statistically equal to zero.
 *csdid_stats pretrend, window (#1 #2)
*medium low_intensity *****************************************************************
  csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_low_intensity_ok) method(drimp )
   estat event, wboot estore(event2) window(-8 6)
 csdid_plot, name(m2,replace) title("Medium Low Intensity") graphregion(fcolor(white))
estat pretrend
*medium high_intensity *****************************************************************
  csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_high_intensity_ok) method(drimp )
   estat event, wboot estore(event3) window(-8 6)
 csdid_plot, name(m3,replace) title("Medium High Intensity") graphregion(fcolor(white))
estat pretrend
 *high intensity *****************************************************************
   csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_high_intensity_ok) method(drimp )
    estat event, wboot estore(event4) window(-8 6)
 csdid_plot, name(m4,replace) title("High Intensity") graphregion(fcolor(white))
estat pretrend


 grc1leg  m4 m3 m2 m1, xcommon scale(0.8) legendfrom(m1) graphregion(fcolor(white))



 
 ****HOUSEHOLD INCOME
   *low_intensity ***************************************************************** 
 csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_low_intensity_ok) method(drimp )
  estat event, wboot estore(event5)  window(-8 6)
 csdid_plot, name(h1,replace) title("Low Intensity") graphregion(fcolor(white))
estat pretrend
*medium low_intensity *****************************************************************
  csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_low_intensity_ok) method(drimp )
   estat event, wboot estore(event6) window(-8 6)
 csdid_plot, name(h2,replace) title("Medium Low Intensity") graphregion(fcolor(white))
estat pretrend
*medium high_intensity *****************************************************************
  csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_high_intensity_ok) method(drimp )
   estat event, wboot estore(event7) window(-8 6)
 csdid_plot, name(h3,replace) title("Medium High Intensity") graphregion(fcolor(white))
estat pretrend
 *high intensity *****************************************************************
   csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_high_intensity_ok) method(drimp )
    estat event, wboot estore(event8) window(-8 6)
 csdid_plot, name(h4,replace) title("High Intensity") graphregion(fcolor(white))
estat pretrend


 grc1leg  h4 h3 h2 h1, xcommon scale(0.8) legendfrom(h1) graphregion(fcolor(white))
 esttab event5 event6 event7 event8 using hh_est.rtf, replace nogaps
 


 
 

