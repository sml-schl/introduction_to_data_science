# Data Science Course: Jupyter Notebooks

Welcome to the Data Science Course! This package contains comprehensive Jupyter notebooks covering 4 days of hands-on data science training.

## 📚 Course Overview

This course provides a complete introduction to data science, from basic concepts to machine learning model evaluation. Each day includes:
- **Student Notebook**: Exercises with TODO markers for hands-on learning
- **Solutions Notebook**: Complete solutions with detailed explanations

### Course Structure (4 Days × 90 minutes each)

#### **Day 2: Introduction to Data Science**
Learn the fundamentals of data science and visualization.

**Topics Covered:**
- The three pillars of data science (Domain Expertise, Statistics, Computer Science)
- Big Data characteristics (5 V's: Volume, Velocity, Variety, Veracity, Value)
- Data visualization with Plotly
- Correlation vs causation
- Identifying poor visualizations
- Data storytelling

**Files:**
- `day2_intro_to_data_science_student.ipynb`
- `day2_intro_to_data_science_solutions.ipynb`

---

#### **Day 4: Data Preparation & Feature Engineering**
Master the art of cleaning and preparing data for analysis.

**Topics Covered:**
- Data quality assessment and the knowledge hierarchy
- Structured vs unstructured data
- Handling missing data (MCAR, MAR, MNAR)
- Imputation strategies (mean, median, group-based)
- Outlier detection (IQR, Z-score, Isolation Forest)
- Normalization techniques (Min-Max, Z-score standardization)
- Categorical encoding (one-hot encoding)
- Feature engineering (creating meaningful new features)

**Files:**
- `day4_data_preparation_student.ipynb`
- `day4_data_preparation_solutions.ipynb`

---

#### **Day 6: Introduction to Machine Learning**
Build your first machine learning models.

**Topics Covered:**
- What is machine learning? (Supervised, Unsupervised, Reinforcement)
- Train-test split methodology
- Classification algorithms (K-NN, Decision Trees, Logistic Regression)
- Regression for continuous predictions
- K-means clustering
- Neural network basics (architecture, activation functions, hyperparameters)
- Cross-validation
- Model comparison

**Files:**
- `day6_machine_learning_student.ipynb`
- `day6_machine_learning_solutions.ipynb`

---

#### **Day 8: Model Evaluation & Assessment**
Learn to rigorously evaluate machine learning models.

**Topics Covered:**
- Bias-variance tradeoff
- Overfitting vs underfitting
- Regression metrics (MAE, MSE, RMSE, R²)
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrices
- ROC/AUC curves
- Type I vs Type II errors
- Cross-validation for robust evaluation
- Ethical AI and Trustworthy AI principles
- Bias detection in models

**Files:**
- `day8_model_evaluation_student.ipynb`
- `day8_model_evaluation_solutions.ipynb`

---

## 📊 Dataset

All notebooks use the **Titanic Dataset**, which contains passenger information from the RMS Titanic disaster.

### Why Titanic?
- Perfect for learning: manageable size (~891 passengers)
- Rich feature set: numerical and categorical variables
- Real missing values and outliers to practice handling
- Interesting for both classification and regression tasks
- Historical context makes insights engaging

### Features:
- **PassengerId**: Unique identifier
- **Survived**: Survival (0 = No, 1 = Yes) - TARGET VARIABLE
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare (£)
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The dataset is automatically loaded from seaborn's built-in datasets, so no manual download is required!

---

## 🛠️ Setup Instructions

### Prerequisites

**Python Version:** Python 3.8 or higher recommended

### Installation

1. **Install Jupyter:**
   ```bash
   pip install jupyter
   ```

2. **Install Required Libraries:**
   ```bash
   pip install pandas numpy plotly seaborn scikit-learn scipy matplotlib
   ```

   Or install all at once:
   ```bash
   pip install pandas numpy plotly seaborn scikit-learn scipy matplotlib jupyter
   ```

### Verify Installation

Run this in a Python terminal to verify all libraries are installed:

```python
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn import __version__ as sklearn_version
print(f"✓ All libraries installed!")
print(f"Scikit-learn version: {sklearn_version}")
```

---

## 🚀 Getting Started

### Starting Jupyter Notebook

1. **Navigate to the notebook directory:**
   ```bash
   cd /path/to/ds_course_notebooks
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Your browser will open with the Jupyter interface**

4. **Open any notebook to start learning!**

### Recommended Learning Path

1. **Start with Day 2 Student Notebook**
   - Work through exercises
   - Fill in TODO markers
   - Try to solve problems before checking solutions

2. **Check Solutions When Needed**
   - Solutions provide complete code
   - Detailed explanations for each exercise
   - Multiple approaches where applicable

3. **Progress Sequentially**
   - Day 2 → Day 4 → Day 6 → Day 8
   - Each day builds on previous concepts

4. **Experiment and Explore**
   - Modify code and see what happens
   - Try bonus challenges
   - Apply concepts to your own datasets

---

## 📖 Learning Tips

### For Students

✅ **DO:**
- Read all markdown cells carefully
- Try to complete exercises before checking solutions
- Experiment with the code
- Run cells multiple times with different parameters
- Ask questions and discuss with peers
- Complete bonus challenges if time permits

❌ **DON'T:**
- Skip to solutions without attempting exercises
- Copy-paste code without understanding it
- Rush through the material
- Ignore error messages (they teach you!)

### For Instructors

**Timing Guidance:**
- Each notebook is designed for 90 minutes
- Adjust based on student pace and questions
- Bonus challenges for faster students
- Reflection questions foster discussion

**Teaching Tips:**
- Demonstrate concepts before letting students try
- Encourage pair programming
- Walk around and help with errors
- Have students present their visualizations
- Discuss ethical implications in Day 8

---

## 🔧 Troubleshooting

### Common Issues

**1. Module Not Found Error**
```python
ModuleNotFoundError: No module named 'plotly'
```
**Solution:** Install the missing library:
```bash
pip install plotly
```

**2. Kernel Died / Notebook Won't Run**
**Solution:** Restart the kernel: `Kernel > Restart Kernel`

**3. Visualizations Not Showing**
**Solution:**
- For Plotly: Make sure you're using a modern browser
- Try: `pip install plotly --upgrade`

**4. Dataset Won't Load**
**Solution:**
```python
# The Titanic dataset loads automatically with seaborn
import seaborn as sns
df = sns.load_dataset('titanic')
```

**5. Out of Memory Errors**
**Solution:** Close other applications, restart Jupyter

---

## 📚 Additional Resources

### Documentation
- **Pandas:** https://pandas.pydata.org/docs/
- **NumPy:** https://numpy.org/doc/
- **Plotly:** https://plotly.com/python/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Seaborn:** https://seaborn.pydata.org/

### Interactive Learning
- **Kaggle Learn:** https://www.kaggle.com/learn
- **DataCamp:** https://www.datacamp.com/
- **Coursera:** Machine Learning courses

### Books
- "Python Data Science Handbook" by Jake VanderPlas
- "Hands-On Machine Learning" by Aurélien Géron
- "Storytelling with Data" by Cole Nussbaumer Knaflic

### Communities
- **Stack Overflow:** https://stackoverflow.com/questions/tagged/pandas
- **Reddit:** r/datascience, r/machinelearning
- **Kaggle Forums:** https://www.kaggle.com/discussion

---

## 🎯 Learning Objectives Summary

By the end of this course, students will be able to:

✓ Understand the fundamentals of data science and its applications
✓ Create effective data visualizations that tell stories
✓ Clean and prepare messy real-world datasets
✓ Handle missing values and outliers appropriately
✓ Engineer meaningful features to improve model performance
✓ Build machine learning models for classification and regression
✓ Evaluate models using appropriate metrics
✓ Detect and mitigate bias in AI systems
✓ Apply ethical principles to data science projects

---

## 📝 Course Assessment

### Evaluation Criteria

**Participation (30%):**
- Active engagement with exercises
- Asking questions
- Helping peers

**Exercise Completion (40%):**
- Completing TODO items
- Producing working code
- Quality of visualizations

**Understanding (30%):**
- Reflection question answers
- Explaining concepts
- Applying learned techniques

---

## 🤝 Support & Feedback

### Getting Help

1. **Check the solutions notebook** for your current day
2. **Review error messages** carefully - they often explain the problem
3. **Search online** - many others have encountered similar issues
4. **Ask your instructor** during class
5. **Collaborate with peers** - teaching others reinforces learning

### Providing Feedback

We'd love to hear from you:
- What worked well?
- What was confusing?
- Suggestions for improvement?
- Additional topics you'd like to learn?

---

## 📜 License & Usage

These notebooks are designed for educational purposes. Feel free to:
- Use for personal learning
- Share with students
- Modify for your needs
- Build upon the content

---

## 🎓 About This Course

This course was designed to provide a practical, hands-on introduction to data science for beginners. The focus is on understanding core concepts through real-world examples rather than mathematical theory.

**Philosophy:**
- Learn by doing
- Real datasets, real problems
- Build intuition before diving into math
- Ethics integrated throughout, not as an afterthought

**Target Audience:**
- Students new to data science
- Professionals transitioning to data roles
- Anyone curious about machine learning
- Python basics helpful but not required

---

## 🚀 Next Steps After This Course

**Continue Learning:**
1. **Advanced Topics:** Deep Learning, NLP, Computer Vision
2. **Specialized Areas:** Time Series, Recommender Systems, A/B Testing
3. **Big Data:** Spark, Hadoop, Cloud Computing
4. **MLOps:** Model deployment, monitoring, CI/CD

**Practice Projects:**
1. Kaggle competitions
2. Personal datasets (fitness, finance, hobbies)
3. Open source contributions
4. Build a portfolio on GitHub

**Certifications:**
- Google Data Analytics Certificate
- IBM Data Science Professional Certificate
- AWS Certified Machine Learning

---

## 📊 Quick Reference: Common Commands

### Jupyter Shortcuts
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell, stay on current
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Convert to markdown
- `Y`: Convert to code

### Pandas Essentials
```python
df.head()              # View first 5 rows
df.info()              # Data types and missing values
df.describe()          # Summary statistics
df.isnull().sum()      # Count missing values
df.groupby('col').mean()  # Group and aggregate
```

### Plotly Basics
```python
px.scatter(df, x='col1', y='col2')  # Scatter plot
px.bar(df, x='col1', y='col2')      # Bar chart
px.histogram(df, x='col')           # Histogram
px.box(df, x='col')                 # Box plot
```

---

## ✨ Acknowledgments

- **Dataset:** Titanic passenger data from Seaborn library
- **Libraries:** Built with pandas, NumPy, Plotly, scikit-learn
- **Inspiration:** Real-world data science education needs

---

**Happy Learning! 🎉**

Remember: Every data scientist started exactly where you are now. The key is consistent practice and curiosity. Don't be afraid to experiment, make mistakes, and learn from them!

*Good luck on your data science journey!*
