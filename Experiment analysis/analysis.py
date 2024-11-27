import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

def load_and_preprocess_data():
    data = pd.read_csv("./Experiment analysis/data.csv")

    # Pipeline to transform the data from the CSV file to a usable form
    data = data.drop(data.columns[[0,1,2,3,15]], axis = 1)
    data.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'VOICE_GENDER', 'PART_GENDER', 'GROUP' ]
    data["OVERALL_SAT"] = (data["Q1"] + data["Q8"] + (6 - data["Q3"]) + (6 - data["Q5"])) / 4 
    data["SYSTEM_COMPR"] = (data["Q4"] + (6 - data["Q2"]) ) / 2
    data["EXPLAINABILITY"] = data["Q6"] 
    data["VOICE_SAT"] = (6 - data["Q7"])
    data["SATISFACTION"] = (data["OVERALL_SAT"] + data["VOICE_SAT"])/2
    data = data.drop(data.columns[[0,1,2,3,4,5,6,7]], axis = 1)

    # The columnheaders are now: [Q8 VOICE_GENDER PART_GENDER GROUP OVERALL_SAT VOICE_SAT]

    # check group assignments
    data = data[~((data['VOICE_GENDER'] == 'Male') & (data['PART_GENDER'] == 'Male') & (data['GROUP'] != 'A'))]
    data = data[~((data['VOICE_GENDER'] == 'Female') & (data['PART_GENDER'] == 'Male') & (data['GROUP'] != 'B'))]
    data = data[~((data['VOICE_GENDER'] == 'Male') & (data['PART_GENDER'] == 'Female') & (data['GROUP'] != 'C'))]
    data = data[~((data['VOICE_GENDER'] == 'Female') & (data['PART_GENDER'] == 'Female') & (data['GROUP'] != 'D'))]

    return data

def descriptive_statistics(data: pd.DataFrame, variables: list[str]):
    print("---------------------------------------------")
    print("Overall descriptive statistics")
    for variable in variables:
        print(f"The mean of {variable} is {data[variable].mean()} and its standard deviation {data[variable].std()}.")
        print(f"A 95% CI for {variable} is {scipy.stats.t.interval(confidence=0.95, df=len(data[variable])-1    , loc=data[variable].mean(), scale=data[variable].std())}")
    print("---------------------------------------------")
    print("Correlation analysis")
    print(data[variables].corr())
    print("---------------------------------------------")
    print("Descriptive statistics by part gender")
    print(pd.pivot_table(data, values=variables, index='PART_GENDER', aggfunc=[np.mean, np.std, np.median]))
    print("---------------------------------------------")
    print("Descriptive statistics by voice gender")
    print(pd.pivot_table(data, values=variables, index='VOICE_GENDER', aggfunc=[np.mean, np.std, np.median]))
    print("---------------------------------------------")
    print("Descriptive statistics by both part and voice gender")
    print(pd.pivot_table(data, values=variables, index=['PART_GENDER', 'VOICE_GENDER'], aggfunc=[np.mean, np.std, np.median]))

def violin_plots_by_group(data: pd.DataFrame, variables:list[str]):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle("Violin Plot of OVERALL_SAT and VOICE_SAT")  

    sns.set(style="whitegrid")  # Set the style (optional)
    axes[0,0].set_title("Group A")
    sns.violinplot(ax=axes[0,0] , data=data[data['GROUP'] == "A"][variables])
    axes[0,1].set_title("Group B")
    sns.violinplot(ax=axes[0,1] , data=data[data['GROUP'] == "B"][variables])
    axes[1,0].set_title("Group C")
    sns.violinplot(ax=axes[1,0] , data=data[data['GROUP'] == "C"][variables])
    axes[1,1].set_title("Group D")
    sns.violinplot(ax=axes[1,1] , data=data[data['GROUP'] == "D"][variables])

    plt.xlabel("Category") 
    plt.ylabel("Value")  
    plt.show() 


def violin_plots_by_dependent_variables(data: pd.DataFrame, variables):
    palette = sns.color_palette()
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(5, 6*len(variables)))
    fig.suptitle("Violin plot of satisfaction measures by group")  
    for k, variable in enumerate(variables):
        axes[k].set_title(f"{variable} between groups")
        sns.violinplot(ax=axes[k] , data=data, x="GROUP", y=variable, order=["A", "B", "C", "D"], color=palette[k])
    plt.show()

def normality_test(data: pd.DataFrame, variable:str, plot:bool=False):
    data[variable] = (data[variable] - data[variable].mean())/data[variable].std()
    print("---------------------------------------------")
    print(f"Normality test for {variable}")
    statistic, pvalue = scipy.stats.shapiro(data[variable])
    print(f"We have a W statistic of {statistic} and a p-value of {pvalue}.")
    if plot:
        fig = sm.qqplot(data[variable], line='45')
        plt.show()
    return pvalue


def comparison_participant_gender_groups(data: pd.DataFrame, dependant_variable: str):
    part_male_data = data[data['PART_GENDER'] == 'Male']
    part_female_data = data[data['PART_GENDER'] == 'Female']

    levene_result = scipy.stats.levene(part_male_data[dependant_variable], part_female_data[dependant_variable])
    t_result= scipy.stats.ttest_ind(part_male_data[dependant_variable], part_female_data[dependant_variable])
    print("---------------------------------------------")
    print(f"The variance test for {dependant_variable} between participant gender groups gives statistic of {levene_result.statistic} and p-value of {levene_result.pvalue}.")
    print("---------------------------------------------")
    print(f"The {dependant_variable} comparison test between participant gender groups gives a T-statistic of {t_result.statistic} and p-value of {t_result.pvalue}.")


def comparison_voice_gender_groups(data: pd.DataFrame, dependant_variable: str, significance: float=0.05, not_sure: bool=True):
    df = data
    if not_sure:
        group_notsure = df[(df['VOICE_GENDER'] == 'Not sure') & (df['PART_GENDER'] == 'Male')][dependant_variable]
        assert len(df[(df['VOICE_GENDER'] == 'Not sure') & (df['PART_GENDER'] == 'Female')]) == 0  # Im our data there is no female that was not sure about voice gender.
    else:
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'A') | (df["GROUP"] == 'C'))), "VOICE_GENDER"] = 'Male'
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'B') | (df["GROUP"] == 'D'))), "VOICE_GENDER"] = 'Female'

    group_male = data[data['VOICE_GENDER'] == 'Male'][dependant_variable]
    group_female = data[data['VOICE_GENDER'] == 'Female'][dependant_variable]

    if not_sure:
        levene_result = scipy.stats.levene(group_male, group_female, group_notsure)
    else:
        levene_result = scipy.stats.levene(group_male, group_female)
    print("---------------------------------------------")
    print(f"The variance test for {dependant_variable} between voice gender groups gives statistic of {levene_result.statistic} and p-value of {levene_result.pvalue}.")
    if levene_result.pvalue < significance:
        print("WARNING: Homoscedascidity may not be warranted.")

    if normality_test(df, variable=dependant_variable)>significance:
        if not_sure:
            print("---------------------------------------------")
            print(f"{dependant_variable} one-way ANOVA test between user assigned voice gender groups.")
            model = ols(f'{dependant_variable} ~ C(VOICE_GENDER)', data=data).fit() 
            result = sm.stats.anova_lm(model, type=1) 
        else:
            print("---------------------------------------------")
            print(f"{dependant_variable} T-test between voice gender groups (not sure answers considered correctly classified).")
            result = scipy.stats.ttest_ind(group_male, group_female)
    else:
        if not_sure:
            print("---------------------------------------------")
            print(f"{dependant_variable} Kruskal-Wallis test between user assigned voice gender groups.")
            result = scipy.stats.kruskal(group_male, group_female, group_notsure)
        else:
            print("---------------------------------------------")
            print(f"{dependant_variable} Wilcoxon test between voice gender groups (not sure answers considered correctly classified).")
            result = scipy.stats.wilcoxon(group_male, group_female)
    print(result)

def comparison_voice_and_participant_gender_groups(data: pd.DataFrame, dependant_variable: str, significance: float=0.05, not_sure: bool=True):
    df = data
    if not_sure:
        group_notsure = df[(df['VOICE_GENDER'] == 'Not sure') & (df['PART_GENDER'] == 'Male')][dependant_variable]
        assert len(df[(df['VOICE_GENDER'] == 'Not sure') & (df['PART_GENDER'] == 'Female')]) == 0
    else:
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'A') | (df["GROUP"] == 'C'))), "VOICE_GENDER"] = 'Male'
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'B') | (df["GROUP"] == 'D'))), "VOICE_GENDER"] = 'Female'

    group_A = df[(df['VOICE_GENDER'] == 'Male') & (df['PART_GENDER'] == 'Male')][dependant_variable]
    group_B = df[(df['VOICE_GENDER'] == 'Male') & (df['PART_GENDER'] == 'Female')][dependant_variable]
    group_C = df[(df['VOICE_GENDER'] == 'Female') & (df['PART_GENDER'] == 'Male')][dependant_variable]
    group_D = df[(df['VOICE_GENDER'] == 'Female') & (df['PART_GENDER'] == 'Female')][dependant_variable]

    
    if normality_test(df, variable=dependant_variable)>significance:
        print("---------------------------------------------")
        print(f"{dependant_variable} two-way ANOVA test between part and voice gender groups")
        model = ols(f'{dependant_variable} ~ C(VOICE_GENDER) + C(PART_GENDER) + C(VOICE_GENDER):C(PART_GENDER)', data=df).fit() 
        result = sm.stats.anova_lm(model, type=2) 
    else:
        print("---------------------------------------------")
        print(f"{dependant_variable} Kruskal-Wallis test between part and voice gender groups")
        if not_sure:
            result = scipy.stats.kruskal(group_A, group_B, group_C, group_D, group_notsure)
        else:
            result = scipy.stats.kruskal(group_A, group_B, group_C, group_D)
    print(result)

    

def two_way_anova(data: pd.DataFrame, dependant_variable: str, not_sure: bool=True):
    df = data
    if not not_sure:
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'A') | (df["GROUP"] == 'C'))), "VOICE_GENDER"] = 'Male'
        df.loc[ ((df["VOICE_GENDER"] == 'Not sure') & ((df["GROUP"] == 'B') | (df["GROUP"] == 'D'))), "VOICE_GENDER"] = 'Female'

    print("---------------------------------------------")
    print(f"{dependant_variable} two-way anova test between part and voice gender groups")
    model = ols(f'{dependant_variable} ~ C(VOICE_GENDER) + C(PART_GENDER) + C(VOICE_GENDER):C(PART_GENDER)', data=df).fit() 
    result = sm.stats.anova_lm(model, type=2) 
    print(result)

def non_parametric_analysis(data: pd.DataFrame, dependant_variable: str):
    group_A_data = data[data['GROUP'] == 'A'][dependant_variable]
    group_B_data = data[data['GROUP'] == 'B'][dependant_variable]
    group_C_data = data[data['GROUP'] == 'C'][dependant_variable]
    group_D_data = data[data['GROUP'] == 'D'][dependant_variable]

    print("---------------------------------------------")
    print(f"{dependant_variable} comparison test between part and voice gender groups with non-parametric test")
    result = scipy.stats.kruskal(group_A_data, group_B_data, group_C_data, group_D_data)
    print(f"The {dependant_variable} comparison test between voice gender groups gives a statistic of {result.statistic} and p-value of {result.pvalue}.")

def main():
    data = load_and_preprocess_data()
    question_variables = ['OVERALL_SAT', 'VOICE_SAT', 'SYSTEM_COMPR']
    dependant_variables = ['OVERALL_SAT', 'VOICE_SAT', 'SATISFACTION']
    control_variables = ['SYSTEM_COMPR', 'EXPLAINABILITY']
    descriptive_statistics(data, variables=dependant_variables)
    #violin_plots_by_dependent_variables(data, variables=dependant_variables)
    #violin_plots_by_group(data, variables=dependant_variables)
    print(len(data))
    for var in dependant_variables:
        print("---------------------------------------------")
        print(f"STATISTICAL TEST FOR {var}")
        comparison_voice_gender_groups(data, dependant_variable=var, not_sure=True)
        comparison_voice_and_participant_gender_groups(data, dependant_variable=var, not_sure=True)
    

if __name__ == "__main__":
    main()