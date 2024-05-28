# BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1

# **Final Project: Machine Learning Implementation**

# **Topic: CSST104 Data Science Salaries 2023 Analysis**
 
**Submitted By:**

**Magpantay, Niño Jandel C.**

**Esguerra, Nashrudin Maverick A.**

**Bautista, Neil Bryan**

**BSCS-3B**


# **PROJECT OVERVIEW 01**

**Key user attributes:** Work year, Experience Level,
Employment Type, Job Title, Salary, Salary Currency, Salary
in Used, Employee Residence, Remote Ratio, Company
Location, Company Size.

# **LIBRARIES AND DATA HANDLING 02**

**Libraries Used:** Pandas, Matplotlib, Seaborn.
**Data Loading and Preprocessing:** Loading from CSV, data
cleaning, handling dates and categorical data.

# **DATA ANALYSIS TECHNIQUE 03**

**Descriptive Statistics:** Mean, Median, Count, Standard
Deviation. 
**Visualization Methods:** Bar charts, Pie charts,
heatmaps, count and distribution plots.

# **KEY FINDINGS 04**

**User Demographics:** Age and Gender distribution, regional
preferences.
**Device Usage:** Popular devices from user
segment, device-based viewing patterns.
**Subscription Details:** Preferences for subscription plans,
impact on user engagement.

# **ADVANCE ANALYSIS 05**
**Geographical Insights:** Categorization in the
continents, regional analysis. 
**Temporal trends:** Signup trends over months, seasonal patterns.

# **MACHINE LEARNING 06**

**Regression Model:** Powerful statistical method for predicting
a continuous variable. In the context of our Data Science
Salaries 2023 Analysis.

# **VISUAL INSIGHTS 07**

**Gender Distribution:** Count plots by country, Device
preference by country. 
**Subscription type popularity:**Visualization of plan popularity.

# **CONCLUSION 08**

Summary of insights derived, implications for future
strategic decisions.

# **APPENDIX**


**Code Snippets:** Provide Python code used for loading,
cleaning, transforming data, and generating visualizations.

**Google Colab Link:** https://colab.research.google.com/drive/1AeeqrgNwKjYq6nzJ
uGc4c0Pwd1e9ne3Y

**Datasets:** Sample dataset of Data Science Salaries 2023
Analysis.

**Additional References:** Referenced any external datasets or
tools used during the analysis process.

**Github Website Link:** [https://github.com/NashEsguerra/CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS-](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB)

# **Data Analysis and Machine Learning Implementation
Project Documentation Template**


**I. Project Overview**

The analysis aims to explore the determinants of salary in the data science domain by 
examining various attributes related to data scientists' employment. Through comprehensive 
analysis, the project seeks to uncover insights into salary trends, geographical variations, and 
factors influencing compensation levels within the industry. By investigating key attributes 
such as work year, experience level, and job title, the project aims to provide actionable 
insights for both data scientists and employers, guiding career decisions and compensation 
strategies. Additionally, the analysis will shed light on the impact of employment type, remote 
work arrangements, and company characteristics on salary disparities, facilitating a deeper 
understanding of the data science labor market dynamics.

1. **Work Year:** Analyzing the distribution of data scientists across different years 
provides insights into industry growth and demand for data science expertise over 
time. Understanding how salary trends evolve year by year can help anticipate future 
market conditions and inform hiring and budgeting decisions for employers. For data 
scientists, insights into salary trends over time can aid in strategic career planning and 
negotiation strategies.

2. **Experience Level:** Examining salary variations based on experience level helps 
identify how experience impacts earning potential within the data science field. 
Understanding the relationship between experience and salary can guide professional 
development efforts, such as acquiring additional skills or pursuing advanced degrees. 
Employers can use this information to establish competitive compensation structures 
and retention strategies tailored to different experience levels.

3. **Employment Type:** Analyzing salary differences between full-time, part-time, and 
contract positions sheds light on employment preferences and their influence on 
compensation. Understanding how different employment arrangements impact salary 
can inform workforce planning and recruitment strategies for employers. For data 
scientists, insights into the trade-offs between employment types and compensation 
can guide career decisions and lifestyle choices.

4. **Job Title:** Exploring salary disparities across job titles reveals how specific roles or 
responsibilities correlate with salary levels, guiding career progression and skill 
development. Understanding the value assigned to different job titles within the data 
science field can inform job seekers' career trajectories and negotiation strategies. 
Employers can use this information to benchmark salaries for various roles and ensure 
equitable compensation across their organization.

5. **Salary:** The primary focus of the analysis, examining salary distributions, trends, and 
factors influencing variations in compensation. By identifying factors that contribute 
to salary discrepancies, such as education, skills, and industry experience, the analysis 
aims to provide actionable insights for both job seekers and employers. 
Understanding the factors driving salary variations can help data scientists negotiate 
competitive compensation packages and employers develop effective hiring and 
retention strategies.

6. **Salary Currency & Salary in USD:** Converting salaries to a common currency 
(USD) facilitates uniform comparison and analysis across different regions and 
currencies. By standardizing salary data, the analysis aims to remove currency-related 
biases and provide insights that are applicable globally. Understanding salary conversions also helps in assessing the true value of compensation packages and 
comparing them across international job markets.

7. **Employee Residence:** Analyzing salary discrepancies based on employee residence 
provides insights into geographical factors influencing compensation levels. 
Understanding how salaries vary across different regions helps both job seekers and 
employers make informed decisions regarding relocation, remote work opportunities, 
and cost-of-living adjustments. Insights into regional salary trends also inform talent 
acquisition strategies, allowing employers to target regions with competitive salary 
offerings.

8. **Remote Ratio:** Understanding how remote work arrangements impact salary helps 
in assessing the value of flexibility and remote work policies. By analyzing salary 
differentials between remote and non-remote positions, the analysis aims to quantify 
the impact of remote work on compensation levels. Insights into remote work 
compensation can inform decisions regarding remote work policies, workforce 
distribution, and talent acquisition strategies.

9. **Company Location:** Exploring salary discrepancies across different company 
locations offers insights into regional variations in compensation and cost of living 
adjustments. Understanding how salaries differ between urban and rural areas, as well
as across different countries or states, informs talent acquisition strategies and 
compensation benchmarking for employers. For data scientists, insights into regional 
salary trends help in evaluating job opportunities and negotiating competitive 
compensation packages.

10. **Company Size:** Analyzing salary trends based on company size illuminates how 
organizational scale influences salary structures and employment opportunities. By 
comparing salaries across small, medium, and large companies, the analysis aims to
identify potential salary premiums associated with company size and industry 
competitiveness. Understanding how company size impacts compensation can guide 
job seekers' career decisions and employers' talent acquisition strategies.


By delving into these attributes, stakeholders in the data science domain can refine hiring 
practices and compensation structures, fostering a more equitable and competitive labor 
market. This meticulous analysis empowers both data scientists and employers with 
insights crucial for career advancement and organizational growth. Such data-driven 
decision-making not only optimizes resource allocation and talent acquisition but also 
cultivates a more vibrant and sustainable data science ecosystem, poised for continuous 
innovation and excellence.


# **II. Libraries and Data Handling**

**Libraries Used:** Pandas for data manipulation, Matplotlib and Seaborn for data visualization.

1. **Pandas (import pandas as pd):** Pandas is a powerful data manipulation and 
analysis library in Python. It provides data structures like DataFrame for handling 
structured data efficiently.

2. **Seaborn (import seaborn as sns):** Seaborn is a statistical data visualization library 
built on top of Matplotlib. It provides a high-level interface for drawing attractive and 
informative statistical graphics.

3. **Matplotlib** (import matplotlib.pyplot as plt): Matplotlib is a comprehensive 
library for creating static, animated, and interactive visualizations in Python.
NumPy (import numpy as np): NumPy is a fundamental package for scientific 
computing with Python. It provides support for arrays, matrices, and 
mathematical functions.

4. **Scikit-learn** (from sklearn.model_selection import train_test_split, from 
sklearn.linear_model import LinearRegression, from sklearn.metrics import 
mean_squared_error, r2_score): Scikit-learn is a machine learning library that 
provides simple and efficient tools for data mining and data analysis. It includes 
various algorithms for classification, regression, clustering, and more.

5. **Plotly (import plotly.express as px):** Plotly is an interactive, open-source plotting 
library that supports over 40 chart types and renders charts in various formats, 
including HTML and PNG images.

6. **PyCountry (import pycountry):** PyCountry is a package providing ISO databases 
and information about countries

**Data Handling Processes:** Data handling processes involve the collection, storage, 
manipulation, analysis, and dissemination of data to extract meaningful insights and support 
decision-making

**• Loading Data:** The code starts by loading a dataset from a CSV file using Pandas' 
read_csv() function.

**• Exploratory Data Analysis (EDA):** Various EDA techniques are employed to 
understand the dataset's structure and characteristics. This includes examining the 
first and last few rows of the dataset (head() and tail()), checking the shape and 
columns of the dataset (shape and columns attributes), exploring data types (dtypes), 
checking unique values in a column (unique()), getting summary statistics (describe()), 
and handling missing values (using isnull() and visualization via a heatmap).

**• Data Visualization:** Seaborn and Matplotlib are used extensively for data 
visualization. This includes creating count plots, bar plots, pie charts, histograms, 
boxplots, line plots, and more to visualize various aspects of the dataset such as salary 
distribution, job titles, salary trends over the years, salary by different categorical 
variables, etc.

**• Data Preprocessing:** Before modeling, the dataset is preprocessed. This involves 
dropping irrelevant columns, encoding categorical variables using OneHotEncoder, 
and splitting the data into training and testing sets using train_test_split().

**• Modeling:** Linear regression modeling is performed using Scikit-learn. The model is 
trained on the encoded training data, and predictions are made on the test data.

**• Evaluation:** Model performance is evaluated using mean squared error 
(mean_squared_error()) and R^2 score (r2_score()).

Encoding or label encoding is a crucial step in preprocessing categorical variables for 
machine learning algorithms. This transformation converts categorical data into numerical 
form, facilitating better prediction by ML models. By assigning unique numerical labels to 
each category, it allows algorithms to interpret categorical features effectively. These steps 
are foundational in any data analysis workflow involving Python, laying the groundwork for 
structured exploration and visualization of user data. Meticulously handling these processes 
ensures that the dataset is well-prepared for more advanced analyses, leading to the 
discovery of actionable insights and informed decision-making


# **III. Data Analysis Techniques**

**Descriptive Statistics:**
Summary statistics like mean, median, count, etc., are used to understand the distribution 
of data. Descriptive statistics summarize and provide a quick overview of the data through 
metrics such as mean, median, count, standard deviation, minimum, and maximum values. 
Here’s how they help in the context of Data Science Salaries 2023 Analysis:

**• Mean and Median:** These measures provide insights into the central tendency of 
numerical data, such as the average salary or years of experience, helping to 
understand typical values.

**• Count:** Counting occurrences of categorical variables like job titles or employee 
residence countries offers a summary of the dataset's composition.

**• Standard Deviation:** This metric quantifies the dispersion or variability of numerical 
data around the mean, indicating the spread of salaries or other variables in the 
dataset.

**Inferential Statistics:**
In the context of the provided code snippet, inferential statistics techniques might not be 
directly evident. However, if applied, they could include hypothesis testing or confidence 
interval estimation to make predictions or inferences about the population based on sample 
data.

**Predictive Modeling:**

**• Linear Regression:** By fitting a linear model between independent variables like 
experience level and company size with the dependent variable, salary, linear 
regression predicts salary based on these factors.

**• Mean Squared Error (MSE):** This metric quantifies the average squared difference 
between the predicted and actual salaries, providing a measure of predictive accuracy.

**• R-squared (R2) Score:** R2 score indicates the proportion of variance in the 
dependent variable (salary) explained by the independent variables, helping to assess 
the goodness-of-fit of the regression model.

**Visualization Techniques:**
**• Bar Chart:** Visualizing job title counts or other categorical variables offers a clear 
understanding of the distribution and frequency of different categories.

**• Pie Chart:** Representing proportions of categorical variables like job titles or remote 
ratios provides a concise overview of categorical data distribution.

**• Heat Map:** Mapping missing values in the dataset highlights areas of data 
incompleteness, aiding in data cleaning and imputation decisions.

**• Count Plot:** Count plots are used to display the frequency of observations in 
categorical variables, such as the count of employees at each experience level or in 
each company size category.

**• Distribution Plots (Histograms and Box Plots):** Histograms and box plots 
visualize the distribution of numerical variables like salary, showing the range, 
central tendency, and spread of the data. These plots help in identifying patterns 
such as skewness, outliers, and central tendency in the dataset.

# **IV. Key Findings**

**User Demographics:** The dataset likely includes information about the demographics of 
data scientists such as age, gender, education level, etc. However, these variables are not 
explicitly analyzed in the provided code.

**• Employee Residence:** The analysis examines the average salary by employee 
residence, providing insights into the geographical distribution of salaries. It shows 
that average adjusted salaries vary based on the country where the employee resides. 
This suggests that the location of the employee can significantly impact their earning 
potential.

**• Experience Level:** Salary trends are explored concerning experience levels. The 
count plot illustrates the distribution of data scientists across different experience 
levels. This insight can help in understanding the demand for data scientists at various 
career stages and how salary scales with experience.

**• Company Location:**The analysis investigates the average salary by company 
location. It reveals variations in average adjusted salaries across different countries 
where companies are located. This indicates that the geographical location of the 
company influences the compensation offered to data scientists.

**• Salary:** The analysis includes visualizations depicting salary distributions, trends over 
the years, and comparisons based on factors like experience level, remote ratio, 
company size, and employment type. These insights provide a comprehensive 
understanding of salary dynamics within the dataset.

**These findings can influence job title decisions in several ways:**

**• Recruitment Strategy:** Understanding salary trends based on demographics, 
experience level, and location can guide recruitment strategies. For example, if the 
goal is to attract experienced data scientists, offering competitive salaries based on 
market trends can be crucial.

**• Market Positioning:** Companies can use salary insights to position themselves 
competitively in the market. Offering salaries that align with industry standards and 
considering geographical variations can help attract top talent.

**• Retention Strategies:** Knowing how salary correlates with factors like experience 
and location can inform retention strategies. Companies can adjust compensation 
packages to retain skilled employees, especially in regions or industries where talent 
is in high demand.

**• Job Title Design:** Based on salary insights and demographic trends, companies can 
design job titles that appeal to their target audience. For example, if the analysis 
indicates a high concentration of mid-level data scientists in certain regions, creating 
job titles tailored to this demographic can attract more candidates.

The advanced analysis delves deeper into the dataset's insights, considering various factors 
such as demographics, experience levels, company and employee locations, and salary 
dynamics. By examining how these variables interplay with salary trends, the analysis 
provides valuable insights into recruitment strategies, market positioning, retention 
initiatives, and job title design. By understanding the nuanced relationships between these 
factors, companies can make informed decisions to attract, retain, and motivate top talent in 
the competitive landscape of data science. This advanced analysis goes beyond surface-level 
observations, offering actionable intelligence to optimize human resource strategies and 
enhance organizational performance.

# **V. Advanced Analysis**

**Geographical Insights:**

• The dataset contains information about both company locations and employee 
residences. By analyzing the average adjusted salary by company location and 
employee residence, we can gain insights into geographical salary discrepancies.
• Using choropleth maps and bar plots, we visualize the average adjusted salary across 
different countries. These visualizations help in understanding regional variations in 
salary levels, potentially indicating differences in economic development or cost of 
living

**Temporal Trends:**

• Analyzing salary trends over time can reveal valuable insights into the dynamics of 
the job market and economic conditions.
• We have explored salary trends over the years, visualizing the average salary against 
work year. This helps in understanding how salaries have evolved over time and 
whether there are any significant trends or fluctuations.
• Additionally, the inflation-adjustment analysis provides a more nuanced view of 
salary changes over time, considering the impact of inflation rates on salary 
adjustments. This allows for a more accurate comparison of salary levels across 
different years.

**Machine Learning Model:**

• We have applied a linear regression model to predict salaries based on various 
features. This goes beyond basic descriptive analysis and allows for the exploration 
of relationships between salary and other factors in a predictive modeling context.
• By encoding categorical variables and fitting a linear regression model, we aim to 
understand the factors that influence salary levels and potentially uncover insights 
that can inform hiring strategies or compensation policies.

**Feature Importance:**

• After fitting the model, we can analyze the feature importance to understand which 
factors have the most significant impact on salary predictions. This can provide 
actionable insights for organizations to prioritize certain factors in their hiring or 
retention strategies.

**Dynamics and Seasonal Patterns:**

• Advanced time series analysis techniques can be applied to uncover seasonal 
patterns or cyclical trends in salary data. This could involve decomposition methods 
or autoregressive models to identify recurring patterns and understand their drivers.
• Exploring correlations between economic indicators and salary levels can provide 
insights into broader market dynamics and how they impact compensation trends.

By employing these advanced analytical techniques, we can gain a deeper understanding of 
the factors influencing salary levels, identify geographical disparities, uncover temporal 
trends, and ultimately derive actionable insights to inform decision-making processes in 
human resources management and organizational strategy.

# **VI. Machine Learning Implementation**

Discuss the data preparation, Data Selection, Data Cleaning and Feature Scaling 
implementation. Process of building the machine learning model. Including the training and 
testing sets.

# **VII. Visual Insights**

**Linear Regression Model**

Utilizing the Linear Regression model, we aim to predict salaries based on various features 
such as experience level, company size, and remote ratio. After preparing the data by 
selecting relevant features, cleaning it to handle missing values and outliers, and scaling 
features for uniformity, we proceed to build the model. This involves training the model on a 
portion of the data, splitting it into training and testing sets, and fitting the Linear Regression 
model to the training data. Evaluation of the model's performance includes assessing metrics
like Mean Squared Error (MSE) and R-squared (R²) score, as well as conducting residual 
analysis to understand the accuracy and potential biases of the model's predictions


**Preparing the Data for Linear Regression:**

**• Data Selection:** Identify relevant features that contribute to predicting salaries, such 
as experience level, company size, and remote ratio.
**• Data Cleaning:** Handle missing values, outliers, and ensure data types are 
appropriate for analysis.
• **Feature Scaling:** Normalize or standardize numerical features to ensure all features 
contribute equally to the model

**Building the Linear Regression Model:**

**• Model Training:** Split the dataset into training and testing sets to train the model on 
a portion of the data and evaluate its performance on unseen data.
**• Splitting Data:** Use techniques like train-test split to divide the dataset into training 
and testing sets while preserving the distribution of data.
**• Model Fitting:** Fit the Linear Regression model to the training data to learn the 
relationship between features and target variable (salary).

**Model Evaluation:**

**• Performance Metrics:** Assess the model's performance using metrics like Mean 
Squared Error (MSE) and R-squared (R²) score to understand how well the model 
predicts salary.
**• Residual Analysis:** Analyze the residuals (the differences between actual and 
predicted values) to check for patterns or biases in the model's predictions and 
identify areas for improvement.

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/a0ebe305-428d-4a69-a6ba-eb4d52a844a5)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/e7bda298-4752-41f1-b718-92b38deb72b4)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/fc74c7de-ebf7-4c7f-b53c-7773cc263ad8)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/aac68273-88cf-4c73-84a9-6a7c7eda819f)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/f42c006f-6e31-44ca-81e9-42e67b6e584d)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/74a4e54c-b800-4e6e-a14f-79784a2beeda)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/28c4a83e-925f-44ce-b9ac-25d36b4cc87f)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/70bf3e10-982a-456c-a7e8-36e279ec4aa6)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/1166885a-e5f3-441a-ac27-fba879aa67cd)

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/173f1355-f31c-409a-bcd1-fd4fd8b82593)





**Implementing the Model with Code Example**

Here’s how you can implement these steps in Python using scikit-learn, assuming the data 
has already been preprocessed:

**1. Import Libraries**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/10ea0585-ebc5-4a31-af07-ce8833a29788)


**2. Numerical Data**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/0d8c391e-c54a-4ac0-b1a6-1735ae4eefbe)


**3. Splitting the dataset into the training set and test set**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/d19031aa-8371-44de-9b74-17fe892a0962)


**4. Initialize the Linear Regression Model**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/fbda06cc-42c0-4f46-bd05-d3562d12e069)


**5. Fit the model**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/582db6d2-c289-4bb2-9716-28b0d6c47a8b)


**6. Predict on the testing set**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/981cadde-b9c0-4f28-9a49-938a0b2345e3)


**7. Evaluate the model**

![image](https://github.com/NashEsguerra/BSCS-CSST104-DATA-SCIENCE-SALARIES-2023-ANALYSIS.-EMB1/assets/145514134/a62cb7d6-8815-409c-9122-788b7d6cdd3b)



# **VIII. Conclusion**

In this analysis, we delved into a dataset containing information about salaries of data 
scientists, exploring various factors influencing salary trends. Through visualization and 
statistical analysis, we gained valuable insights that can significantly impact businesses and 
organizations employing data scientists.
Firstly, we observed the distribution of salaries across different experience levels, remote 
work ratios, company sizes, and employment types. This analysis provided a comprehensive 
understanding of how these factors correlate with salary variations. Furthermore, we 
identified the top 10 job titles and their respective salary distributions, shedding light on the 
roles that command higher compensation in the data science field. Moreover, we investigated 
the relationship between salary and other variables such as work year, experience level, 
remote work ratio, and company size. These analyses revealed nuanced insights into how 
these factors collectively influence salary levels.


Additionally, we addressed the impact of inflation on salaries, providing a method to adjust 
salaries based on inflation rates, thereby ensuring fair compensation over time. Finally, 
leveraging geographical insights, we explored the average salaries by company location and 
employee residence, highlighting regional disparities in data scientist salaries. Overall, this 
analysis underscores the importance of data-driven decision-making in human resources and 
organizational planning. By understanding the factors influencing data scientist salaries, 
businesses can make informed decisions regarding recruitment, retention, and compensation 
strategies.
Moving forward, further analysis could explore additional factors such as educational 
background, specific skill sets, industry domains, and certification levels to gain a more 
granular understanding of salary determinants in the data science landscape. Additionally, 
longitudinal studies tracking salary trends over time could provide valuable insights into 
industry dynamics and economic shifts

**Appendix**

**Data Sources:**
• The dataset used for analysis and modeling is sourced from 
"DataScientiestSalaries.csv".
• Inflation rates used for adjusting salaries are provided within the code as dictionaries 
(us_inflation_rates and global_inflation_rates).

**Contributor Details:**
• Analysis, visualization, and modeling code written by [Magpantay,Esguerra,Bautista].
• Data cleaning, preprocessing, and manipulation may have been performed by 
**[Magpantay,Esguerra,Bautista].**


# **Data Science Salaries 2023 Analysis Using Python**

**Program:**  https://colab.research.google.com/drive/1AeeqrgNwKjYq6nzJuGc4c0Pwd1e9ne3Y#scrollTo=VCI2JIUBux9q&printMode=true

**Datasets:** https://drive.google.com/file/d/13P7JQj1vPx6D92X1KuMReBr8DseoVSnO/view?usp=sharing
