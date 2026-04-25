# AI-Based Financial Audit System



#### Overview



The AI-Based Financial Audit System is a smart web application developed to make financial auditing faster, easier, and more accurate. Instead of manually checking large numbers of financial transactions, the system automatically analyzes transaction data and identifies suspicious activities using both rule-based auditing and Machine Learning.



The project uses the Isolation Forest algorithm to detect unusual transaction patterns and possible fraud cases. It also uses SHAP (Explainable AI) to help explain why certain transactions are marked as suspicious, making the results easier to understand for auditors and users.







#### Features



1. Rule-based anomaly detection
2. Machine Learning-based anomaly detection using Isolation Forest
3. Explainable AI using SHAP
4. Interactive dashboard visualizations
5. Downloadable audit reports
6. Email-based report sharing
7. Fraud similarity detection
8. Custom audit rule support





#### 

#### Technologies Used



* &#x20;Python
* &#x20;Flask
* &#x20;Pandas
* &#x20;NumPy
* &#x20;Scikit-learn
* &#x20;Plotly
* &#x20;SHAP







### &#x20;How to Run the Project

##### 

##### Step 1: Clone the Repository



git clone https://github.com/Aditi-0303/Ai-audit-webapp.git







##### Step 2: Open the Project Folder

##### 

cd Ai-audit-webapp





##### Step 3: Create a Virtual Environment



python -m venv .venv







##### Step 4: Activate the Virtual Environment



For Windows:



..venv\\Scripts\\activate







##### Step 5: Install Required Libraries



pip install flask pandas numpy scikit-learn plotly shap







\## Email Setup for Sending Audit Reports



This project includes an email feature that allows audit reports to be shared directly through Gmail using SMTP.



##### Step 6: Enable 2-Step Verification in Gmail



Open:



https://myaccount.google.com/security



Enable:



\* 2-Step Verification





##### Step 7: Generate a Gmail App Password



Go to:



https://myaccount.google.com/apppasswords



Generate a 16-character App Password.



Example:



abcdefghijklmnop







##### Step 8: Configure SMTP Environment Variables



Run these commands in PowerShell:



$env:SMTP\_HOST="smtp.gmail.com"



$env:SMTP\_PORT="587"



$env:SMTP\_USER="\[yourgmail@gmail.com](mailto:yourgmail@gmail.com)"



$env:SMTP\_PASS="your\_16\_character\_app\_password"



$env:EMAIL\_FROM="\[yourgmail@gmail.com](mailto:yourgmail@gmail.com)"



Replace the email address and app password with your own Gmail credentials.







##### Step 9: Run the Application



python app.py





##### Step 10: Open the Application

##### 

Open your browser and visit:



http://127.0.0.1:5000







#### Machine Learning Used



This project uses the Isolation Forest algorithm, an unsupervised Machine Learning technique used to detect unusual or suspicious financial transactions by identifying patterns that differ from normal transaction behavior.

#### 

#### Author



Aditi.U.K





