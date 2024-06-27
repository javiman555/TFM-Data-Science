import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the date range
date_range = pd.date_range(start="2024-04-01", end="2024-06-25")

# Generate random data for each category
investigation = [2, 3, 1, 2, 1, 4, 4,
                 0, 0, 0, 0, 0, 3, 2,
                 0, 0, 0, 0, 0, 2, 3,
                 0, 0, 0, 0, 0, 1, 3,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0,
                 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 1, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 1, 0,
                 1, 0, 0, 0, 0, 0, 1,
                 2, 2]
programming = [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 0, 1, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 4, 5, 3, 5,
                 1, 1, 1, 5, 1, 3, 2,
                 1, 1, 8, 6, 8, 8, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1,
                 2, 2]
thesis =        [0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 1, 0, 0,
                 0, 3, 0, 0, 0, 1, 0,
                 0, 0, 2, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0, 0,
                 4, 5, 4, 2, 6, 3, 4,
                 2, 1, 4, 2, 2, 1, 2,
                 2, 2]

# Create a DataFrame to hold the data
data = pd.DataFrame({
    'Date': date_range,
    'Investigation': investigation,
    'Programming': programming,
    'Thesis': thesis
})

# Set the Date column as the index
data.set_index('Date', inplace=True)

# Plot the data using a bar plot
plt.figure(figsize=(14, 8))

# Calculate cumulative sums for each category
cumulative_investigation = data['Investigation'].cumsum()
cumulative_programming = data['Programming'].cumsum()
cumulative_thesis = data['Thesis'].cumsum()

# Plotting
plt.bar(data.index, cumulative_investigation, label='Investigation', color='blue')
plt.bar(data.index, cumulative_programming, bottom=cumulative_investigation, label='Programming', color='green')
plt.bar(data.index, cumulative_thesis, bottom=cumulative_investigation + cumulative_programming, label='Thesis', color='red')

# Customize the plot
plt.title('Timeline')
plt.xlabel('Date')
plt.ylabel('Cumulative Hours')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
