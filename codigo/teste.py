
# Create data
import scipy.stats as stats
 
group1 = [456, 564, 54, 554, 54, 51, 1, 12, 45, 5]
group2 = [65, 87, 456, 564, 456, 564, 564, 6, 4, 564]
 
# conduct the Wilcoxon-Signed Rank Test
print(len(group1), len(group2))
print(stats.wilcoxon(group1, group2))   