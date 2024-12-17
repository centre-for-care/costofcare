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

*label variables
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
la var hhsize "household size"

*Figure 2 :  Care Intensity and Income Profiles.
lowess ind_inc_deflated dvage if care_intensity_t==0, generate(level1) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==1, generate(level2) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==2, generate(level3) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==3, generate(level4) nograph 
lowess ind_inc_deflated dvage if treated_ok==0, generate(ctr_all) nograph 
graph twoway  line  level1 level2 level3 level4 ctr_all dvage, ///
  msymbol(O) mcolor(#41558C) mcolor(#6E9887) mcolor(#E89818) mcolor(#CF202A) mcolor(#000000)  ///
  xtitle(Age) ytitle(Individual Income) legend( label(1 " Low Intensity") label(2 "Medium-Low Intensity") label(3 "Medium-High Intensity") label(4 "High-Intensity") label(5 "Control")) graphregion(color(white)) saving(level_intensity2) sort

*Table 1 : Descriptive statistics.
global list0 ind_inc_deflated hh_inc_deflated inc_share employed_d dvage male married  asian black white other_mixed hhsize lower_education intermediate_education advanced_education 
eststo drop *
eststo: estpost summarize $list0 if care_intensity_t==0 [aw=weight_yearx]
eststo: estpost summarize $list0 if care_intensity_t==1 [aw=weight_yearx]
eststo: estpost summarize $list0 if care_intensity_t==2 [aw=weight_yearx]
eststo: estpost summarize $list0 if care_intensity_t==3 [aw=weight_yearx]
eststo: estpost summarize $list0 if treated_ok==0 [aw=weight_yearx]
esttab using summary.rtf, cells("mean(fmt(2))sd(fmt(2)) ")   wide nodepvar  title({\b Table 2.} {\i Descriptive statistics treatment vs control group }) compress replace 
esttab using summary1.rtf, cells("mean(fmt(2))")   label wide nodepvar  title({\b Table 2.} {\i Descriptive statistics treatment vs control group }) compress replace 







