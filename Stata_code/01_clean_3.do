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


* cleaning variables 
clear all

use "$data\treatment_control_clean_tr.dta"

label variable ind_inc_deflated "Monthly income (deflated)"

*intensity of care
encode aidhrs, gen(hours)
gen care_intensity=.
replace care_intensity=0 if hours==1
replace care_intensity=1 if hours==2| hours==3| hours==9
replace care_intensity=2 if hours==5|hours==6|hours==7| hours==8
replace care_intensity=3 if hours==4|hours==10| hours==11

*low intensity from 0 to 4
*medium low intensity from 5 to 19
*medium high intensity from 20 to 49
*high intensity from 50+

gen first_treat=year_treated
bysort pidp: egen treated_ok=max(treated)
gen match=1 if year_treated==year
 
**** did sant'anna variables
*low_intensity
gen first_low_intensity=year_treated if care_intensity==0 & match==1
replace first_low_intensity=0 if treated_ok==0
 bysort pidp: egen first_low_intensity_ok=max(first_low_intensity)
*medium_low_intensity
gen first_medium_low_intensity=year_treated if care_intensity==1 & match==1
replace first_medium_low_intensity=0 if treated_ok==0
 bysort pidp: egen first_medium_low_intensity_ok=max(first_medium_low_intensity)
*medium_high_intensity
gen first_medium_high_intensity=year_treated if care_intensity==2 & match==1
replace first_medium_high_intensity=0 if treated_ok==0
 bysort pidp: egen first_medium_high_intensity_ok=max(first_medium_high_intensity)
*high_intensity
gen first_high_intensity=year_treated if care_intensity==3 & match==1
replace first_high_intensity=0 if treated_ok==0
 bysort pidp: egen first_high_intensity_ok=max(first_high_intensity)

drop first_low_intensity  first_medium_low_intensity  first_medium_high_intensity first_high_intensity


* care intensity defined as per year t
gen care_intensity_t=.
replace care_intensity_t=0 if care_intensity==0 & match==1
replace care_intensity_t=1 if care_intensity==1 & match==1
replace care_intensity_t=2 if care_intensity==2 & match==1
replace care_intensity_t=3 if care_intensity==3 & match==1
bysort pidp: egen care_intensity_tm= max(care_intensity_t)

drop care_intensity_t
rename care_intensity_tm care_intensity_t
 
label define intensity_t2 0 "0-4 hours" 1 "5-19 hours" 2 "10-19 hours" 3"20-49 hours" 4 "50 plus" 
label values   care_intensity_t intensity_t2

tab care_intensity_t, gen(x)
rename x1 low_int1
rename x2 medium_low_int1
rename x3 medium_high_int1
rename x4 high_int1


*label variable 
la var ind_inc_deflated "Ind. Income"
la var  hh_inc_deflated "Household Income"
la var inc_share "Income share (%)"
la var employed_d "Employed"
la var dvage "Age"
la var male "Male"
la var married "Married"
la var asian "Asian"
la var black "Black"
la var white "White"
la var other_mixed "Mixed" 
la var lower_education "Lower education"
la var intermediate_education "Intermediate education"
la var advanced_education "Advanced education"

 save "$data\generated.dta", replace
