# Attention Span Prediction System

A Machine Learning-based web application that predicts user attention span based on content characteristics and engagement behavior. This project analyzes how users interact with different types of content and provides actionable insights to improve content strategy.


## Overview

With the rapid growth of digital content, understanding user attention has become crucial. This project:
- Predicts attention span using Machine Learning  
- Analyzes user behavior across devices, platforms, and time  
- Provides insights to improve engagement and content strategy  


## Purpose

The aim of this project is to understand digital behavior patterns and provide a predictive system that helps analyze which types of content keep users engaged.


## Project Structure

It uses a trained Machine Learning model integrated with a Flask web app to generate real-time predictions through a simple and clean UI.


## Features

- Predicts attention span based on content metadata
- Clean Flask-based web interface
- Fully functional ML pipeline: preprocessing → model → prediction
- Lightweight and easy to run locally
- Ideal for research, analytics, and user-behavior modeling


## Tech Stack

- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn, Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, VS Code  


## How It Works

- User inputs content-related details through the web interface
- Data is preprocessed and scaled
- Trained ML model predicts attention span
- Results are displayed on the UI  


## Key Insights

### Content Engagement
- Interactive and infographic content types show the highest engagement
- Text-based content has the lowest average attention
  
### Device Usage Trends
- Mobile and Desktop are the most used devices
- Android users show highest engagement, especially during evening and night
- iPads, iPhones, and Smart TVs have minimal usage
  
### Time-Based Behavior
- Highest engagement observed at night
- Afternoon shows the lowest attention levels

### Weekly Trends
- Sunday has the highest engagement
- Monday records the lowest

 ### Content Strategy
- Longer content increases attention span
- Extremely long content leads to inconsistent engagement
  
### User Interaction
- Higher scroll depth strongly correlates with increased time spent


## Future Enhancements

- Improve model accuracy and feature engineering
- Add user login & history tracking
- Integrate Deep Learning model
- Deploy full version on cloud
- Add interactive analytics dashboard
- Convert UI into React Native


## Running the Project Locally

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd attention-span-prediction
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
```
http://127.0.0.1:5000/
```

---

### Notes
- Make sure Python (3.x) is installed  
- Ensure all `.pkl` model files are present in the correct directory  
- If port 5000 is busy, change the port in `app.py`  

---

Thanks for checking out this project! Feel free to share your thoughts.
