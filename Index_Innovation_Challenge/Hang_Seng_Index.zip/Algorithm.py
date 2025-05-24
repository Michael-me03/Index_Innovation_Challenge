import pandas as pd
import requests
from io import BytesIO
import pdfplumber
import re
from sklearn.preprocessing import MinMaxScaler

# Load CSV Data
def load_csv_data(file_path):
    data = pd.read_csv(file_path, sep=';', encoding='utf-8')
    data.columns = data.columns.str.strip()  # Clean column names
    data['Free-float %'] = pd.to_numeric(data['Free-float %'].str.replace(',', '.').str.strip(), errors='coerce')
    return data

# Extract text from PDF files with error handling
def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_text = ""
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text()
        return pdf_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_all_pdfs(data):
    data['Financial Report Text'] = data['Financial report / Allotment'].apply(extract_text_from_pdf)
    return data

# Extract financial metrics from text
def extract_financial_metrics(text):
    metrics = {}
    revenue_match = re.search(r'Revenue[:\s]+([\d,]+)', text)
    
    metrics['Revenue'] = revenue_match.group(1) if revenue_match else '0'
    
    return metrics

def apply_financial_extraction(data):
    data['Financial Metrics'] = data['Financial Report Text'].apply(extract_financial_metrics)
    return data

# Normalize and build index based on 50% free-float and 50% revenue
def build_financial_index(data):
    # Convert financial metrics to numeric format
    data['Revenue'] = pd.to_numeric(data['Financial Metrics'].apply(lambda x: x['Revenue'].replace(',', '')), errors='coerce').fillna(0)
    
    # Normalize free-float % and revenue using MinMaxScaler
    scaler = MinMaxScaler()
    data[['Free-float %', 'Revenue']] = scaler.fit_transform(data[['Free-float %', 'Revenue']])
    
    # Build the final index with 50% Free-float and 50% Revenue
    data['Final Index'] = 0.5 * data['Free-float %'] + 0.5 * data['Revenue']
    
    return data

# Normalize the financial index to sum to 100%
def normalize_index_to_100(data):
    total_index = data['Final Index'].sum()
    if total_index > 0:
        data['Normalized Financial Index'] = (data['Final Index'] / total_index) * 100
    else:
        data['Normalized Financial Index'] = 0
    
    # Ensure no stock has 0% by adding a small adjustment if needed
    min_adjustment = 0.1  # Set the minimum threshold to 0.1%
    data['Normalized Financial Index'] = data['Normalized Financial Index'].apply(lambda x: x if x > min_adjustment else min_adjustment)
    
    # Re-normalize to ensure the total is 100%
    total_normalized = data['Normalized Financial Index'].sum()
    data['Normalized Financial Index'] = (data['Normalized Financial Index'] / total_normalized) * 100
    return data

def main():

    # Load the CSV data
    csv_file_path = './Hang_Seng_Index.zip/Data/Batch 3 Data.csv'  # Adjust file path as needed
    data = load_csv_data(csv_file_path)
    
    # Extract Financial Data from PDFs
    data = extract_all_pdfs(data)
    
    # Apply Financial Metric Extraction from PDFs
    data = apply_financial_extraction(data)
    
    # Build Financial Index based on Free-float and Revenue
    data = build_financial_index(data)
    
    # Normalize Index to Sum to 100% and avoid 0% values
    data = normalize_index_to_100(data)
    
    # Output Results
    print(data[['RIC', 'Normalized Financial Index']])  # Include the stock ID (RIC) in the output
    
    # Save results as CSV in the requested format: ID,output
    output_data = data[['RIC', 'Normalized Financial Index']].copy()
    output_data.columns = ['ID', 'output']  # Rename columns to match the format
    output_data.to_csv('output_index.csv', index=False)

    print("output_index.csv created successfully!")

if __name__ == "__main__":
    main()
