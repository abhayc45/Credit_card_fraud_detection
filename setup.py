from setuptools import setup, find_packages

setup(
    name="credit_card_fraud_detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "xgboost",
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "joblib",
        # Add any other required packages
    ],
    entry_points={
        "console_scripts": [
            "fraud-app=app:main",  # Optional CLI if needed
        ]
    }
)
