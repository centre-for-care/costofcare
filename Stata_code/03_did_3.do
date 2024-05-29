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
  *low_intensity 
 csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_low_intensity_ok) method(drimp )
  estat event, wboot estore(event1)  window(-8 6)
 csdid_plot, name(m1,replace) title("Low Intensity") graphregion(fcolor(white))
*medium low_intensity
  csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_low_intensity_ok) method(drimp )
   estat event, wboot estore(event2) window(-8 6)
 csdid_plot, name(m2,replace) title("Medium Low Intensity") graphregion(fcolor(white))
*medium high_intensity
  csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_high_intensity_ok) method(drimp )
   estat event, wboot estore(event3) window(-8 6)
 csdid_plot, name(m3,replace) title("Medium High Intensity") graphregion(fcolor(white))
 *high intensity
   csdid ind_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_high_intensity_ok) method(drimp )
    estat event, wboot estore(event4) window(-8 6)
 csdid_plot, name(m4,replace) title("High Intensity") graphregion(fcolor(white))


 grc1leg  m4 m3 m2 m1, xcommon scale(0.8) legendfrom(m1) graphregion(fcolor(white))

 
 ****HOUSEHOLD INCOME
   *low_intensity 
 csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_low_intensity_ok) method(drimp )
  estat event, wboot estore(event5)  window(-8 6)
 csdid_plot, name(h1,replace) title("Low Intensity") graphregion(fcolor(white))
*medium low_intensity
  csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_low_intensity_ok) method(drimp )
   estat event, wboot estore(event6) window(-8 6)
 csdid_plot, name(h2,replace) title("Medium Low Intensity") graphregion(fcolor(white))
*medium high_intensity
  csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_medium_high_intensity_ok) method(drimp )
   estat event, wboot estore(event7) window(-8 6)
 csdid_plot, name(h3,replace) title("Medium High Intensity") graphregion(fcolor(white))
 *high intensity
   csdid hh_inc_deflated male dvage married asian black mixed  white  intermediate_education advanced_education employed_d hhsize [weight=weight_yearx], ivar(pidp) time(year) gvar(first_high_intensity_ok) method(drimp )
    estat event, wboot estore(event8) window(-8 6)
 csdid_plot, name(h4,replace) title("High Intensity") graphregion(fcolor(white))


 grc1leg  h4 h3 h2 h1, xcommon scale(0.8) legendfrom(h1) graphregion(fcolor(white))
 esttab event5 event6 event7 event8 using hh_est.rtf, replace nogaps
 

 
 
 
*  csdid ind_inc_deflated dvage married asian black mixed  white  intermediate_education advanced_education [weight=weight_yearx], ivar(pidp) time(year) gvar(first_treat) method(drimp ) saverif(rif_example)   replace 
  *use rif_example
  
*estimate att by event, group or cohorts  
estat pretrend
estat all
estat simple
estat calendar
estat group 
estat event
*csdid_stats simple 
*csdid_stats calendar
*csdid_stats group
*csdid_stats event

*** if similiar graphs to isc are needed: 
*The CSDID plot can be done via -coefplot-
*smoothed lines can be done using vc_reg and vc_graph (ssc install vc_reg). Also look from -event_plot-

