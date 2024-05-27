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

*Figure 1- thresholds as in the analysis
lowess ind_inc_deflated dvage if care_intensity_t==0, generate(level1) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==1, generate(level2) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==2, generate(level3) nograph 
lowess ind_inc_deflated dvage if care_intensity_t==3, generate(level4) nograph 
lowess ind_inc_deflated dvage if treated_ok==0, generate(ctr_all) nograph 
graph twoway  line  level1 level2 level3 level4 ctr_all dvage, ///
  msymbol(O) mcolor(#41558C) mcolor(#6E9887) mcolor(#E89818) mcolor(#CF202A) mcolor(#000000)  ///
  xtitle(Age) ytitle(Individual Income) legend( label(1 " Low Intensity") label(2 "Medium-Low Intensity") label(3 "Medium-High Intensity") label(4 "High-Intensity") label(5 "Control")) graphregion(color(white)) saving(level_intensity2) sort

*Table 1
tab reindex care_intensity_t if reindex>=-8 & reindex<=8 [aw=weight_yearx], col row
tabout reindex care_intensity_t using table2.text if reindex>=-8 & reindex<=8 [aw=weight_yearx], c(freq) f(0c ) font(bold) replace


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

*Table 2 high vs low vs control - background characteristics
global list0 ind_inc_deflated hh_inc_deflated inc_share employed_d dvage male married  asian black white other_mixed hhsize lower_education intermediate_education advanced_education 
*global list1 ind_inc_deflated hh_inc_deflated inc_share employed_d dvage male married  asian black white other_mixed lower_education intermediate_education advanced_education higher_professional intermediate_occ large_employers lower_management lower_supervisory routine semi_routine
*global list2 ind_inc_deflated hh_inc_deflated inc_share employed_d dvage male married  asian black white other_mixed lower_education intermediate_education advanced_education higher_professional intermediate_occ large_employers lower_management lower_supervisory routine semi_routine
eststo drop *
eststo: estpost summarize $list0 if low_int1==1 [aw=weight_yearx]
eststo: estpost summarize $list0 if medium_low_int1==1 [aw=weight_yearx]
eststo: estpost summarize $list0 if medium_high_int1==1 [aw=weight_yearx]
eststo: estpost summarize $list0 if high_int1==1 [aw=weight_yearx]
eststo: estpost summarize $list0 if treated==0 [aw=weight_yearx]
esttab using summary.rtf, cells("mean(fmt(2))sd(fmt(2)) ")   wide nodepvar  title({\b Table 2.} {\i Descriptive statistics treatment vs control group }) compress replace 
esttab using summary1.rtf, cells("mean(fmt(2))")   label wide nodepvar  title({\b Table 2.} {\i Descriptive statistics treatment vs control group }) compress replace 







