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
"""

# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2

""" Contingency Table

Given a data set (e.g., MNIST) and two classification criteria X and Y (e.g., might be the digits), a contingency table 
is defined as a table showing the values of the frequencies of the values (BASELINES) of the data set with respect to 
the different classes of criteria X and Y.

Each cell in the table corresponds to the count or frequency of observations that correspond to the row and column 
categories.
"""

table = [[10, 20, 30],
         [6, 9, 17]]
print(table)

stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)

# interpret test-statistic
# Significance level indicates that with 95% the two classification criteria cannot be considered independent.
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
