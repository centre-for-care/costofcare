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

***control*********************************************************************
clear all
import excel "C:\Users\if1mp\Desktop\project2\control_cases.xlsx", sheet("control_cases") firstrow
save "C:\Users\if1mp\Desktop\project2\control_tr.dta", replace
clear all
use "control_tr", replace

*CLEANING CONTROL
*variable have to be recoded as missing in stata are different from those in python 
drop A
*occupation
encode jbnssec8_dv, gen (job8)
replace job8=. if job8==9 | job8==10
tab job8, gen (s)
rename s1 higher_professional
rename s2 intermediate_occ 
rename s3 large_employers
rename s4 lower_management
rename s5 lower_supervisory
rename s6 routine
rename s7 semi_routine
rename s8 small_employers

*education
tab edu_6, gen(x)
rename x1 level_1
rename x2 level_2
rename x3 level_3
rename x4 level_4
rename x5 level_5
rename x6 level_6

tab edu_3, gen(y)
rename y1 lower_education
rename y2 intermediate_education
rename y3 advanced_education
encode edu_3, gen(education3)
label define education 1"lower" 2 "intermediate" 3 "advanced"
label values education3 education 

*sex
rename sex_recoded male
label define male 0"female"1"male"
label values male male

*married
label define married 0 "single" 1 "married"
label values  mastat_recoded married
rename mastat_recoded married

*ethnicity
gen other_mixed=other
replace other_mixed=1 if mixed==1

*employed
encode employed, gen(employed1)
replace employed1=. if employed1==2
gen employed2=0 if employed1==3
replace employed2=1 if employed1==1
label define employed 0 "unemployed" 1 "employed"
label values employed2 employed
drop employed1
rename employed2 employed1
tab employed1, gen(x)
rename x1 unemployed_d
rename x2 employed_d

save "$data\control_clean_tr.dta", replace

****treatment****************************************************************** 
clear all


clear all
import excel "C:\Users\if1mp\Desktop\project2\treated_cases.xlsx", sheet("treated_cases") firstrow
save "C:\Users\if1mp\Desktop\project2\treatment_tr.dta", replace
clear all
use "$data/treatment_tr", replace

drop aidhrs_recoded_6
*CLEANING TREATMENT 
drop A
*occupation
encode jbnssec8_dv, gen (job8)
replace job8=. if job8==9 | job8==10
tab job8, gen (s)
rename s1 higher_professional
rename s2 intermediate_occ 
rename s3 large_employers
rename s4 lower_management
rename s5 lower_supervisory
rename s6 routine
rename s7 semi_routine
rename s8 small_employers


*education
tab edu_6, gen(x)
rename x1 level_1
rename x2 level_2
rename x3 level_3
rename x4 level_4
rename x5 level_5
rename x6 level_6

tab edu_3, gen(y)
rename y1 lower_education
rename y2 intermediate_education
rename y3 advanced_education
encode edu_3, gen(education3)
label define education 1"lower" 2 "intermediate" 3 "advanced"
label values education3 education 

*sex
rename sex_recoded male
label define male 0"female"1"male"
label values male male


*married
label define married 0 "single" 1 "married"
label values  mastat_recoded married
rename mastat_recoded married

*ethnicity
gen other_mixed=other
replace other_mixed=1 if mixed==1

*employed
encode employed, gen(employed1)
replace employed1=0 if employed1==2
label define employed 0 "unemployed" 1 "employed"
label values employed1 employed
tab employed1, gen(x)
rename x1 unemployed_d
rename x2 employed_d


*recode aidhrs_recoded_3
encode aidhrs_recoded_3, gen(aidhrs_recoded_3k)
drop aidhrs_recoded_3
rename aidhrs_recoded_3k aidhrs_recoded_3
drop aidhrs_recoded_4
save "$data\treatment_clean_tr.dta", replace

*APPEND CONTROL AND TREATMENT**************************************************

append using "$data\control_clean_tr"

save "$data\treatment_control_clean_tr_new.dta", replace

********************************************************************************

* cleaning variables - care 
clear all
*use "$data\treatment_control_clean_tr.dta"
use "$data\treatment_control_clean_tr_new.dta"

label variable ind_inc_deflated "Monthly income (deflated)"

*intensity of care
encode aidhrs, gen(hours)
gen care_intensity=.
*0-4
replace care_intensity=0 if hours==1|hours==2
*5-19
replace care_intensity=1 if hours==3| hours==4| hours==5| hours==14| hours==15
*20-49
replace care_intensity=2 if hours==8|hours==9|hours==10| hours==11| hours==12| hours==13
*50+
replace care_intensity=3 if hours==6|hours==7| hours==16| hours==17| hours==18

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
 
label define intensity_t2 0 "0-4 hours" 1 "5-19 hours" 2 "20-49 hours" 3"50+hours" 
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

 save "$data\generated_new.dta", replace
