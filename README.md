# AppaltiGPT Evaluation Tool

Streamlit app for running and evaluating extraction tasks (requirements, main info, products) on tender documents.

## Features

- Run multiple extraction runs in parallel
- Export results to Excel for analysis
- Compare extraction consistency across runs
- Track processing times and costs

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

3. Configure environment variables:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_SERVICE_KEY`: Your Supabase service key
   - `API_URL`: URL of the FastAPI backend (default: http://localhost:8000)

## Local Development

Run the Streamlit app:
```bash
streamlit run evals.py
```

The app will be available at http://localhost:8501

## Deployment on Railway

1. Create a new Railway service
2. Connect this repository
3. Set environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `API_URL` (your backend URL on Railway)
4. Railway will automatically deploy using:
   ```
   streamlit run evals.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

## Usage

1. Select a tender from the sidebar
2. Choose the number of extraction runs (1-10)
3. Click "Run Extractions" for your desired extraction type:
   - **Requirements Extraction**: Extract tender requirements
   - **Main Info Extraction**: Extract tender main information
   - **Products Extraction**: Extract products from Capitolato Tecnico
4. Download results as Excel file

## Notes

- This is an internal tool for evaluation purposes
- All extractions run in evaluation mode (no database saves, no verification)
- Requires the backend API to be running and accessible
