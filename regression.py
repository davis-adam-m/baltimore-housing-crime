# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2

#  load dataset
df = pd.read_csv('clean_data/master.csv')

# create derived column to calculate population density, rescale to square miles
df['pop_density'] = df['pop'] / (df['area_sqft'] / 27878400)

# rescale median income to 1k increments for interpretability
df['income_thousands'] = df['income'] / 1000

# review descriptive stats for entire dataset and distribution of IV and DV via histograms
print(df.describe())
df.hist(column=['vacancy_count', 'homicide_count'])
plt.show()

# check homicide_count variation/mean ratio to determine appropriate stat model for regression
mean_homicide = df['homicide_count'].mean()
var_homicide = np.var(df['homicide_count'], ddof=1)

print(f"Mean: {mean_homicide}")
print(f"Variance: {var_homicide}")
print(f"Variance/Mean ratio: {var_homicide / mean_homicide}")

# high variance ratio indicates negative binomial regression model is most appropriate

# check for multicollinearity amongst predictors
x = df[['vacancy_count', 'income_thousands', 'pop_density']]
print(x.corr())

# set DV and IVs
y = df['homicide_count']
x = df[['vacancy_count', 'income_thousands', 'pop_density']]

# add intercept term
x = sm.add_constant(x)

# fit the negative binomial regression model with test alpha values to determine best fit
#for test_alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
     #nb_model = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=test_alpha))
     #nb_results = nb_model.fit()
     #print(f"alpha={test_alpha}: log-likelihood={nb_results.llf:.2f}, AIC={nb_results.aic:.2f}")

# fit the negative binomial regression model with manually selected alpha value
nb_model = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=0.2))
nb_results = nb_model.fit()
print(nb_results.summary())

# convert coefficients to IRRs for interpretability
irr = np.exp(nb_results.params)
conf_int = np.exp(nb_results.conf_int())
conf_int.columns = ['2.5%', '97.5%']

irr_table = pd.concat([irr, conf_int], axis=1)
irr_table.columns = ['IRR', '2.5%', '97.5%']
print(irr_table)

# results indicate vacancy is statistically significant; vacancy_count +1 = 0.0737% increase in predicted homicide count
# calculate effect of 100 vacancies
effect_100 = np.exp(nb_results.params['vacancy_count'] * 100)
print(effect_100)

# 100 vacancies = 1.07 homicides. that is a potential death. compelling results

# income is also significant; +$1000 median income = 1.8% decrease in predicted homicide count

# pop density is significant, but has a very small effect per unit. calculate effect per 1000 people:
effect_1000 = np.exp(nb_results.params['pop_density'] * 1000)
print(effect_1000)

# +1000 people per sq mile = +1.03 expected homicides. potential reason to reduce vacancies, spread dense population

# run a null test to rule out random chance
X_null = sm.add_constant(pd.DataFrame(index=df.index))
null_model = sm.GLM(y, X_null, family=sm.families.NegativeBinomial(alpha=0.2))
null_results = null_model.fit()
print(null_results.llf)

# check mcfadden r2 for a more robust measure of significance
mcfadden_r2 = 1 - (nb_results.llf / null_results.llf)
print(mcfadden_r2)

# 0.2099 indicates the NB model explains variation in homicide counts with 21% improvement over null model

# calculate p-value NB model vs null model
lr_stat = 2 * (nb_results.llf - null_results.llf)
p_value = chi2.sf(lr_stat, df=3)
print(lr_stat, p_value)

# p < 0.001 indicates very significant improvement of NB model over null model, LR 97.4

# produce visualizations

# scatterplot: vacancy_count vs homicide_count, plot binomial trendline
# create a smooth range of vacancy_count values
vacancy_range = np.linspace(df['vacancy_count'].min(), df['vacancy_count'].max(), 200)

# hold other predictors at their mean
mean_income = df['income_thousands'].mean()
mean_density = df['pop_density'].mean()

plot_df = pd.DataFrame({
    'vacancy_count': vacancy_range,
    'income_thousands': mean_income,
    'pop_density': mean_density
})

plot_df_const = sm.add_constant(plot_df, has_constant='add')
plot_df['predicted'] = nb_results.predict(plot_df_const[['const', 'vacancy_count', 'income_thousands', 'pop_density']])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(data=df, x='vacancy_count', y='homicide_count', alpha=0.6, s=60, ax=axes[0])
sns.lineplot(data=plot_df, x='vacancy_count', y='predicted', color='firebrick', ax=axes[0])
axes[0].set_xlim(0, 600)
axes[0].set_title('Zoomed: 0–600 Vacant Buildings')

sns.scatterplot(data=df, x='vacancy_count', y='homicide_count', alpha=0.6, s=60, ax=axes[1])
sns.lineplot(data=plot_df, x='vacancy_count', y='predicted', color='firebrick', ax=axes[1])
axes[1].set_title('Full Range')

for ax in axes:
    ax.set_xlabel('Vacant Building Count')
    ax.set_ylabel('Homicide Count')

fig.suptitle('Vacant Building Count vs. Homicide Count by CSA', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('vacancy_homicide_scatter_dual.png', dpi=300)
plt.show()

# IRR/Coefficient plot
scale_factors = {'vacancy_count': 100, 'income_thousands': 1, 'pop_density': 1000}

scaled_irr = {}
for var, factor in scale_factors.items():
    coef = nb_results.params[var]
    ci = nb_results.conf_int().loc[var]
    scaled_irr[var] = {
        'IRR': np.exp(coef * factor),
        'ci_low': np.exp(ci[0] * factor),
        'ci_high': np.exp(ci[1] * factor)
    }
# nicer labels for the plot
label_map = {
    'vacancy_count': 'Vacant Buildings',
    'income_thousands': 'Median Income ($1,000s)',
    'pop_density': 'Population Density (per sq mi)'
}

scaled_irr_df = pd.DataFrame(scaled_irr).T.reset_index()
scaled_irr_df.columns = ['predictor', 'IRR', 'ci_low', 'ci_high']
scaled_irr_df['predictor'] = scaled_irr_df['predictor'].map(label_map)
scaled_irr_df['effect_magnitude'] = (scaled_irr_df['IRR'] - 1).abs()
scaled_irr_df = scaled_irr_df.sort_values('effect_magnitude', ascending=True)

plt.figure(figsize=(8, 5))

# error bars representing the 95% CI
plt.errorbar(
    x=scaled_irr_df['IRR'],
    y=scaled_irr_df['predictor'],
    xerr=[
        scaled_irr_df['IRR'] - scaled_irr_df['ci_low'],
        scaled_irr_df['ci_high'] - scaled_irr_df['IRR']
    ],
    fmt='o',
    color='firebrick',
    ecolor='gray',
    elinewidth=2,
    capsize=5,
    markersize=8
)

# reference line at IRR = 1 (no effect)
plt.axvline(x=1, color='black', linestyle='--', linewidth=1)

plt.xlabel('Incidence Rate Ratio (IRR)')
plt.title('Effect of Predictors on Expected Homicide Count\n(Negative Binomial Regression, 95% CI)', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('irr_plot.png', dpi=300)
plt.show()