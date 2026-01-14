import pandas as pd
import matplotlib.pyplot as plt

#Read CSV using pyarrow engine and store in dataframe
titanic_data = pd.read_csv('data/titanic.csv' , engine='pyarrow')
#print(titanic_data.head(8))

#Log raw file data info for audits
print(f"Columns in raw file received::{titanic_data.columns.tolist()}")
print(f"Total rows in raw file received::{len(titanic_data)}")

#Immediately convert to parquet for better performance in future reads
titanic_data.to_parquet('data/titanic.parquet')
print("Converted raw CSV to Parquet format for better performance in future reads.")

#Analyze data: How many people survived the Titanic disaster and plot a pie chart
plt.pie(titanic_data['Survived'].value_counts(), labels=['Did not survive', 'Survived'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Survival Distribution on Titanic')
plt.show()

#Get subset of data for people aged less than 15 and greater than 65 (who are considered vulnerable) and only keep Name, Age and Survived columns
subset_data = titanic_data[(titanic_data['Age'] < 15) & (titanic_data['Age'] > 65)]
vulnerable_people_data = subset_data[['Name', 'Age', 'Survived']]
subset_data.to_parquet('data/vulnerable_people_data.parquet')
print("Extracted vulnerable people data and saved to Parquet format.")
