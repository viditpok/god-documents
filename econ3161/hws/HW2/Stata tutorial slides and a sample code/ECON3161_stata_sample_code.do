*This code is for ECON3161 Class coding illustration

* change working directory
cd C:\Users\xyan306\Desktop\stataDataset

* read data
use wage2.dta, replace

*generate variable logwage you can also use gen as a short-hand notation
generate logwage=log(wage)


* export the dataset to .xls file
export excel using "wage", firstrow(variables) replace

* brief summary statistics of the data or sum command for short-hand notation
summarize

summarize wage,d

* more detailed summary statistics
sum,detail

describe 

* doing a regression analysis we could also use reg (short-hand notation)
regress wage educ


*export your results 
*esttab using example.tex, title (Regression table)
reg wage educ

* predict wage
predict wagehat

*drop variable wagehat
drop wagehat


*plot the data
scatter wage educ, title("Plot of wage against educ") ytitle("earnings") xtitle("education")


*count how many people have 12 years education
count if educ==12

summarize, detail
estpost tabstat wage educ exper age black married south urban, c(stat)  stat(mean sd min max n)
esttab using "summary.tex", replace cells ("mean sd min max count") title("Summary Statistics") nonumber


