# AMR_Surveillance_Dashboard
 A comprehensive interactive dashboard to analyze and visualize global Antimicrobial Resistance (AMR) trends. Built with Streamlit, this project integrates public health data, predictive modeling, and unsupervised learning to support antimicrobial stewardship.

 ---

 ## ❓ Why This Project
 Antimicrobial resistance is a growing global health crisis, leading to prolonged hospital stays, higher costs, and increased mortality. Healthcare systems need dynamic surveillance tools to:
 - Monitor pathogen resistance trends
 - Predict emerging threats
 - Align antibiotic usage with resistance patterns

 This dashboard was built to serve that purpose — helping public health analysts, infection control teams, and researchers make data-driven decisions.

 ---

 ## 🔍 What It Does

 ### 1. **Resistance Trends**
 Track and compare AMR patterns in adults vs. pediatric populations, with:
 - Top 5 pathogen trends
 - Year-over-year percent changes
 - Cumulative and proportional burden

 ### 2. **Forecasting Resistance**
 Using Prophet, forecast resistance levels for any selected pathogen with:
 - Visual trend projections
 - Decomposed seasonal patterns
 - Model evaluation (MSE)

 ### 3. **AMU vs Resistance**
 Explore correlations between antimicrobial usage and resistance:
 - Predefined and custom drug-pathogen comparisons
 - Pearson correlation + regression analysis
 - Dual-axis time trend overlays

 ### 4. **Mechanism Clustering**
 Cluster AMR genes based on their resistance mechanisms:
 - PCA + KMeans clustering
 - Cluster labeling by dominant mechanism & drug class

 ### 5. **ML Feature Importance**
 Understand which biological features most influence resistance clustering:
 - Random Forest classifier
 - Bar chart of top predictive features

 ---

 ## ⚙️ How to Use It

 ### 1. Clone the repo
 ```bash
 git clone https://github.com/your-username/amr-surveillance-dashboard.git
 cd amr-surveillance-dashboard
 ```

 ### 2. Set up environment
 ```bash
 pip install -r requirements.txt
 ```

 ### 3. Launch the dashboard
 ```bash
 streamlit run amr_dashboard.py
 ```

 ---

 ## 🔬 Data Sources
 - **CDC NHSN AMR Dataset** (Adult & Pediatric)
 - **WHO GLASS** Antimicrobial Usage Data
 - **CARD** – Comprehensive Antibiotic Resistance Database

 ---

 ## 🧠 Skills Highlighted
 - Data analysis & transformation with `pandas`
 - Visual storytelling with `matplotlib`, `seaborn`, `plotly`
 - Time series modeling with `Prophet`
 - Machine learning with `scikit-learn`
 - Web app development with `streamlit`

 ---

 ## 💡 Author
 Saurav Ajit Kulkarni  
 📧 sauravkulkarni9@gmail.com  
 🔗 [LinkedIn](https://linkedin.com/in/sauravkulkarni)

 ---

 ## 📃 License
 This project is intended for educational and portfolio use. Please cite data sources if reused.
