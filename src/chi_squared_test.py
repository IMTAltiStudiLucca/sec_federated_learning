""""
    File name: chi_squared_test.py
    Description: This is a test used to check whether two different classifications of the same data set are
                 independent of each other. The result of the test is a test statistic that has a Chi-Squared
                 distribution and can be interpreted to reject or fail to reject the assumption or null hypothesis that
                 the observed and expected frequencies are the same, that the classifications are independent of
                 each other.
                 H0 (null hypothesis): the two classification criteria are independent of each other
                 H1: The two criteria are related to each other.
    Credits:  Statistical Methods for Machine Learning (book), J. Brownlee
              https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
"""

from scipy.stats import chi2_contingency
from scipy.stats import chi2

def chi_squared_test(table):
    stat, p, dof, expected = chi2_contingency(table)
    
    return p

""" Contingency Table
This table summarizes the collected observations with one variable corresponding to columns and another variable 
corresponding to rows. For instance, one variable is the couple of two digits and the other is another couple of
digits.
Each cell in the table corresponds to the count or frequency of observations that correspond to the row and column 
categories. 

Example: We would verify it the couple 1-7 is independent from the couple 3-8
X = [digit-1  digit-7];  X = [bit-0 bit-1]
Y = [digit-3  digit-8];  Y = [bit-0 bit-1]

and if we have a table as follows

table = [[10, 20],
         [6, 9]]
10 corresponds to the times that I observed the couple digit-1 with digit-3.
"""

table = [[10, 20],
         [6, 9]]
print(table)

stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)

""" Test statics
    - Test Statistic ≥ Critical Value: significant result, reject null hypothesis, dependent (H1).
        -- p-value ≤ alpha: significant result, reject null hypothesis, dependent (H1)
􏰠    - Test Statistic < Critical Value: not significant result, fail to reject null hypothesis, independent (H0).
        -- p-value > alpha: not significant result, fail to reject null hypothesis, independent (H0).
"""


""" Interpret test-statistic
"""
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


""" Interpret p-value
The p-value helps to understand if the difference between the observed result and the hypothesized one 
is due to the randomness introduced by the sampling, or if this difference is statistically significant, that is, 
difficult to explain through the randomness due to the sampling.
"""
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
