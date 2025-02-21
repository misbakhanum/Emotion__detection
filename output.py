import matplotlib.pyplot as plt
import numpy as np

# Data
states = ['Assam', 'Odisha', 'Haryana', 'Telangana', 'Rajasthan']
crime_rates = [168.3, 137.8, 119.7, 111.2, 105.4]
cases_2019 = [30025, 23183, 14483, 20406, 21065]
cases_2020 = [22153, 41590, 12791, 31532, 31452]
cases_2021 = [23153, 43593, 13455, 32653, 40738]

bar_width = 0.25
x = np.arange(len(states))

# Plotting bars
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, cases_2019, width=bar_width, label='Cases in 2019', color='red')
plt.bar(x, cases_2020, width=bar_width, label='Cases in 2020', color='blue')
plt.bar(x + bar_width, cases_2021, width=bar_width, label='Cases in 2021', color='green')

# Adding crime rate as text above the bars
for i, rate in enumerate(crime_rates):
    plt.text(x[i], max(cases_2019[i], cases_2020[i], cases_2021[i]) + 2000, f'{rate}', 
             ha='center', color='black', fontsize=9)

# Customizing the plot
plt.xticks(x, states)
plt.xlabel('States')
plt.ylabel('Number of Cases')
plt.title('States with Highest Rate of Crime Against Women in 2021')
plt.legend()
plt.tight_layout()

# Show plot
plt.show()

