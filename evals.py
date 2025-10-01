#!/usr/bin/env python3
"""
Extraction Evaluation Tool

This Streamlit app runs multiple extractions (requirements or main info) and exports results to Excel.
"""

import streamlit as st
import json
import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from io import BytesIO

# Import Supabase client (still needed for loading tenders)
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
supabase: Client = create_client(supabase_url, supabase_service_key)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")  # FastAPI backend URL

# Page config
st.set_page_config(
    page_title="Extraction Evaluation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_tender' not in st.session_state:
    st.session_state.selected_tender = None
if 'extraction_runs' not in st.session_state:
    st.session_state.extraction_runs = []
if 'main_info_runs' not in st.session_state:
    st.session_state.main_info_runs = []
if 'products_runs' not in st.session_state:
    st.session_state.products_runs = []
if 'available_tenders' not in st.session_state:
    st.session_state.available_tenders = []

def load_tenders():
    """Load available tenders from Supabase."""
    try:
        response = supabase.table('tenders').select('id, name, company_id, info').eq('company_id', '47').execute()
        if response.data:
            st.session_state.available_tenders = response.data
            return response.data
        return []
    except Exception as e:
        st.error(f"Error loading tenders: {e}")
        return []

def run_products_extraction(tender_id: str, company_id: str) -> Dict:
    """Run a single products extraction for the given tender."""
    try:
        # Ensure tender_id and company_id are strings
        tender_id = str(tender_id)
        company_id = str(company_id)

        # Prepare request payload for products extraction
        payload = {
            "tender_id": tender_id,
            "company_id": company_id,
            "is_eval": True
        }

        # Start extraction job
        response = requests.post(
            f"{API_URL}/extract-products",
            json=payload
        )

        if response.status_code != 200:
            st.error(f"Failed to start products extraction: {response.text}")
            return None

        job_data = response.json()
        job_id = job_data.get("job_id")

        # Poll for completion
        progress_bar = st.progress(0, text="Starting products extraction...")
        start_time = time.time()

        while True:
            status_response = requests.get(f"{API_URL}/job-status/{job_id}")

            if status_response.status_code != 200:
                st.error(f"Failed to get job status: {status_response.text}")
                return None

            status_data = status_response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)

            # Update progress bar
            progress_bar.progress(progress / 100, text=f"Products extraction progress: {progress}%")

            if status == "completed":
                progress_bar.progress(100, text="Products extraction completed!")
                result = status_data.get("result")
                return result
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                st.error(f"Products extraction failed: {error}")
                return None

            # Timeout after 10 minutes
            if time.time() - start_time > 600:
                st.error("Products extraction timed out after 10 minutes")
                return None

            time.sleep(2)

    except Exception as e:
        st.error(f"Error during products extraction: {e}")
        return None


def run_main_info_extraction(tender_id: str, company_id: str) -> Dict:
    """Run a single main info extraction for the given tender."""
    try:
        # Ensure tender_id and company_id are strings
        tender_id = str(tender_id)
        company_id = str(company_id)

        # Prepare request payload for main info extraction
        payload = {
            "tender_id": tender_id,
            "company_id": company_id,
            "is_eval": True
        }

        # Start extraction job
        response = requests.post(
            f"{API_URL}/extract-main-info",
            json=payload
        )

        if response.status_code != 200:
            st.error(f"Failed to start main info extraction: {response.text}")
            return None

        job_data = response.json()
        job_id = job_data.get("job_id")

        # Poll for completion
        progress_bar = st.progress(0, text="Starting main info extraction...")
        start_time = time.time()

        while True:
            status_response = requests.get(f"{API_URL}/job-status/{job_id}")

            if status_response.status_code != 200:
                st.error(f"Failed to get job status: {status_response.text}")
                return None

            status_data = status_response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)

            # Update progress bar
            progress_bar.progress(progress / 100, text=f"Main info extraction progress: {progress}%")

            if status == "completed":
                progress_bar.progress(100, text="Main info extraction completed!")
                result = status_data.get("result")
                return result
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                st.error(f"Main info extraction failed: {error}")
                return None

            # Timeout after 10 minutes
            if time.time() - start_time > 600:
                st.error("Main info extraction timed out after 10 minutes")
                return None

            time.sleep(2)

    except Exception as e:
        st.error(f"Error during main info extraction: {e}")
        return None


def run_extraction(tender_id: str, company_id: str) -> Dict:
    """Run a single extraction for the given tender."""
    try:
        # Ensure tender_id and company_id are strings
        tender_id = str(tender_id)
        company_id = str(company_id)

        # Prepare request payload
        payload = {
            "tender_id": tender_id,
            "company_id": company_id,
            "is_eval": True  # Set to True for evaluation mode
        }

        # Start extraction job
        response = requests.post(
            f"{API_URL}/extract-requirements",
            json=payload
        )

        if response.status_code != 200:
            st.error(f"Failed to start extraction: {response.text}")
            return None

        job_data = response.json()
        job_id = job_data.get("job_id")

        # Poll for completion
        progress_bar = st.progress(0, text="Starting extraction...")
        start_time = time.time()

        while True:
            status_response = requests.get(f"{API_URL}/job-status/{job_id}")

            if status_response.status_code != 200:
                st.error(f"Failed to get job status: {status_response.text}")
                return None

            status_data = status_response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)

            # Update progress bar
            progress_bar.progress(progress / 100, text=f"Extraction progress: {progress}%")

            if status == "completed":
                progress_bar.progress(100, text="Extraction completed!")
                result = status_data.get("result")

                return result
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                st.error(f"Extraction failed: {error}")
                return None

            # Timeout after 10 minutes
            if time.time() - start_time > 600:
                st.error("Extraction timed out after 10 minutes")
                return None

            time.sleep(2)

    except Exception as e:
        st.error(f"Error during extraction: {e}")
        return None

def products_to_excel(products_runs: List[Dict]) -> BytesIO:
    """Convert products extraction results to Excel file with one sheet per run."""
    if not products_runs:
        return None

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for run in products_runs:
            run_num = run['run_number']
            products_data = run.get('products', {})
            products_list = products_data.get('prodotti', [])

            # Create main products DataFrame
            if products_list:
                df_products = pd.DataFrame(products_list)

                # Reorder columns for better readability
                column_order = ['nome_prodotto', 'prezzo_unitario', 'quantita_richieste',
                              'lotto', 'certificazioni_richieste',
                              'caratteristiche_tecniche']

                # Only include columns that exist in the data
                existing_columns = [col for col in column_order if col in df_products.columns]
                other_columns = [col for col in df_products.columns if col not in column_order]
                final_columns = existing_columns + other_columns

                df_products = df_products[final_columns]

                # Clean up data for display
                for col in df_products.columns:
                    if col == 'lotto':
                        # Handle lotto which might be array or string
                        df_products[col] = df_products[col].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else ''
                        )
                    else:
                        df_products[col] = df_products[col].fillna('')

            # Write to Excel
            sheet_name = f"Products_{run_num}"

            # Write products data directly without summary
            if products_list:
                df_products.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            else:
                # Write "No products found" message
                no_products_df = pd.DataFrame([["No products found"]], columns=["Message"])
                no_products_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)

            # Try to format the sheet
            try:
                worksheet = writer.sheets[sheet_name]
                # Set column widths
                worksheet.column_dimensions['A'].width = 25
                worksheet.column_dimensions['B'].width = 40
                if products_list and 'C' in worksheet.column_dimensions:
                    worksheet.column_dimensions['C'].width = 20
                if products_list and 'D' in worksheet.column_dimensions:
                    worksheet.column_dimensions['D'].width = 15
                if products_list and 'E' in worksheet.column_dimensions:
                    worksheet.column_dimensions['E'].width = 50
            except:
                pass

    output.seek(0)
    return output


def main_info_to_excel(main_info_runs: List[Dict]) -> BytesIO:
    """Convert main info extraction results to Excel file with one sheet per run."""
    if not main_info_runs:
        return None

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for run in main_info_runs:
            run_num = run['run_number']
            info = run.get('main_info', {})

            # Create multiple DataFrames for different sections
            data_sections = []

            # CIG Information
            cig_data = []
            for cig_item in info.get('cig', []):
                cig_data.append({
                    "CIG": cig_item.get('cig', ''),
                    "Lotto": cig_item.get('lotto', ''),
                    "Valore Lotto": cig_item.get('valore_lotto', 0)
                })
            if cig_data:
                df_cig = pd.DataFrame(cig_data)
                data_sections.append(("CIG Information", df_cig))

            # General Information
            general_data = [{
                "Campo": "Oggetto",
                "Valore": info.get('oggetto', '')
            }, {
                "Campo": "Stazione Appaltante",
                "Valore": info.get('stazione_appaltante', '')
            }, {
                "Campo": "Link Stazione Appaltante",
                "Valore": info.get('link_stazione_appaltante', '')
            }, {
                "Campo": "Documento",
                "Valore": info.get('documento', '')
            }]
            df_general = pd.DataFrame(general_data)
            data_sections.append(("General Information", df_general))

            # Dates
            dates = info.get('date', {})
            date_data = [{
                "Tipo Data": "Pubblicazione",
                "Data": dates.get('pubblicazione', '') if dates.get('pubblicazione') else 'null'
            }, {
                "Tipo Data": "Termine Chiarimenti",
                "Data": dates.get('termine_chiarimenti', '')
            }, {
                "Tipo Data": "Termine Presentazione Offerte",
                "Data": dates.get('termine_presentazione_offerte', '')
            }]
            df_dates = pd.DataFrame(date_data)
            data_sections.append(("Dates", df_dates))

            # Contacts
            contatti = info.get('contatti', {})
            contact_data = [{
                "Campo": "RUP",
                "Valore": contatti.get('rup', '')
            }, {
                "Campo": "Email",
                "Valore": contatti.get('email', '')
            }, {
                "Campo": "Telefono",
                "Valore": contatti.get('telefono', '')
            }]
            df_contacts = pd.DataFrame(contact_data)
            data_sections.append(("Contacts", df_contacts))

            # Procedure
            procedura = info.get('procedura', {})
            punteggi = procedura.get('punteggi_aggiudicazione', {})
            proc_data = [{
                "Campo": "Tipo Procedura",
                "Valore": procedura.get('tipo_procedura', '')
            }, {
                "Campo": "Criterio Aggiudicazione",
                "Valore": procedura.get('criterio_aggiudicazione', '')
            }, {
                "Campo": "Codici CPV",
                "Valore": ', '.join(procedura.get('codice_cpv', []))
            }, {
                "Campo": "Punti Tabellari",
                "Valore": punteggi.get('punti_tabellari', '')
            }, {
                "Campo": "Punti Quantitativi",
                "Valore": punteggi.get('punti_quantitativi', '')
            }, {
                "Campo": "Punti Discrezionali",
                "Valore": punteggi.get('punti_discrezionali', '')
            }]
            df_procedure = pd.DataFrame(proc_data)
            data_sections.append(("Procedure", df_procedure))

            # Location
            ubicazione = info.get('ubicazione', {})
            location_data = [{
                "Campo": "Comune",
                "Valore": ubicazione.get('comune', '')
            }, {
                "Campo": "Provincia",
                "Valore": ubicazione.get('provincia', '')
            }, {
                "Campo": "Regione",
                "Valore": ubicazione.get('regione', '')
            }]
            df_location = pd.DataFrame(location_data)
            data_sections.append(("Location", df_location))

            # Amounts
            importi = info.get('importi', {})
            amount_data = [{
                "Campo": "Importo Base",
                "Valore": importi.get('importo_base', 0)
            }, {
                "Campo": "Importo Complessivo",
                "Valore": importi.get('importo_complessivo', 0)
            }]
            df_amounts = pd.DataFrame(amount_data)
            data_sections.append(("Amounts", df_amounts))

            # Contract Duration
            durata = info.get('durata_contratto', {})
            duration_data = [{
                "Campo": "Durata (mesi)",
                "Valore": durata.get('mesi', 0)
            }, {
                "Campo": "Prorogabile",
                "Valore": "Sì" if durata.get('prorogabile', False) else "No"
            }, {
                "Campo": "Proroga Massima (mesi)",
                "Valore": durata.get('proroga_massima_mesi', '') if durata.get('proroga_massima_mesi') else 'null'
            }]
            df_duration = pd.DataFrame(duration_data)
            data_sections.append(("Contract Duration", df_duration))

            # Sopralluogo
            sopralluogo = info.get('sopralluogo', {})
            sopralluogo_data = [{
                "Campo": "Previsto",
                "Valore": "Sì" if sopralluogo.get('previsto', False) else "No"
            }]
            df_sopralluogo = pd.DataFrame(sopralluogo_data)
            data_sections.append(("Sopralluogo", df_sopralluogo))

            # Write all sections to a single sheet with spacing
            sheet_name = f"MainInfo_{run_num}"
            current_row = 0

            for section_name, df in data_sections:
                # Write section header
                df_header = pd.DataFrame([[section_name]])
                df_header.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                 index=False, header=False)
                current_row += 2

                # Write data
                df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                current_row += len(df) + 3  # Add spacing between sections

            # Try to format the sheet
            try:
                worksheet = writer.sheets[sheet_name]
                # Set column widths
                worksheet.column_dimensions['A'].width = 30
                worksheet.column_dimensions['B'].width = 50
                if 'C' in worksheet.column_dimensions:
                    worksheet.column_dimensions['C'].width = 30
            except:
                pass

    output.seek(0)
    return output


def requirements_to_excel(extraction_runs: List[Dict]) -> BytesIO:
    """Convert extraction results to Excel file with one sheet per run."""
    if not extraction_runs:
        return None

    output = BytesIO()

    # Use default engine (will auto-detect available engine)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for run in extraction_runs:
            run_num = run['run_number']
            requirements = run['requirements']

            # Create a list of rows for this run
            data = []

            # Requisiti ordine generale
            for req in requirements.get("requisiti_ordine_generale", []):
                data.append({
                    "Categoria": "Requisiti Ordine Generale",
                    "Sottocategoria": "",
                    "Requisito": req
                })

            # Requisiti ordine speciale
            ros = requirements.get("requisiti_ordine_speciale", {})

            # Idoneità professionale
            for albo in ros.get("idoneita_professionale", {}).get("albi_registri_richiesti", []):
                data.append({
                    "Categoria": "Requisiti Ordine Speciale",
                    "Sottocategoria": "Idoneità Professionale",
                    "Requisito": albo
                })

            # Capacità economico-finanziaria
            for fatt in ros.get("capacita_economico_finanziaria", {}).get("fatturato_minimo", []):
                data.append({
                    "Categoria": "Requisiti Ordine Speciale",
                    "Sottocategoria": "Capacità Economico-Finanziaria",
                    "Requisito": fatt
                })

            # Capacità tecnico-professionale
            ctp = ros.get("capacita_tecnico_professionale", {})
            for exp in ctp.get("esperienza_pregressa", []):
                data.append({
                    "Categoria": "Requisiti Ordine Speciale",
                    "Sottocategoria": "Esperienza Pregressa",
                    "Requisito": exp
                })
            for cert in ctp.get("certificazioni", []):
                # Handle certificazioni as objects with verification results
                if isinstance(cert, dict):
                    cert_text = cert.get("certificazione", "")
                    if cert.get("is_obbligatoria"):
                        cert_text += " (obbligatoria)"
                    else:
                        cert_text += " (migliorativa)"
                    if cert.get("note_obbligatorieta_certificazione"):
                        cert_text += f" - {cert['note_obbligatorieta_certificazione']}"

                    # Add verification status if present
                    if "presente" in cert:
                        cert_text += f" | VERIFICA: {'✓' if cert['presente'] else '✗'}"
                        if cert.get("note"):
                            cert_text += f" - {cert['note']}"
                else:
                    cert_text = str(cert)

                data.append({
                    "Categoria": "Requisiti Ordine Speciale",
                    "Sottocategoria": "Certificazioni",
                    "Requisito": cert_text
                })

            # Requisiti partecipazione
            for req in requirements.get("requisiti_partecipazione", []):
                data.append({
                    "Categoria": "Requisiti Partecipazione",
                    "Sottocategoria": "",
                    "Requisito": req
                })

            # Criteri valutazione
            criterio = requirements.get("criteri_valutazione", {}).get("tipo_criterio", "")
            if criterio:
                data.append({
                    "Categoria": "Criteri Valutazione",
                    "Sottocategoria": "",
                    "Requisito": criterio
                })

            # Create DataFrame and write to Excel
            df = pd.DataFrame(data)
            sheet_name = f"Estrazione_{run_num}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Try to auto-adjust column widths if openpyxl is available
            try:
                worksheet = writer.sheets[sheet_name]
                for column in df:
                    column_length = max(df[column].astype(str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_length + 2, 50)
            except:
                pass  # Skip column width adjustment if not supported

    output.seek(0)
    return output

def main():
    st.title("📊 Extraction Evaluation Tool")
    st.markdown("Run multiple extractions for requirements or main info and export results to Excel")

    # Create tabs for different extraction types
    tab1, tab2, tab3 = st.tabs(["Requirements Extraction", "Main Info Extraction", "Products Extraction"])

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Load tenders button
        if st.button("🔄 Refresh Tenders"):
            load_tenders()

        # Tender selection
        if not st.session_state.available_tenders:
            load_tenders()

        if st.session_state.available_tenders:
            tender_options = {
                f"{t['id']} - {t['name'][:50]}": t
                for t in st.session_state.available_tenders
            }

            selected_tender_key = st.selectbox(
                "Select Tender",
                options=list(tender_options.keys()),
                help="Choose a tender to extract requirements from"
            )

            if selected_tender_key:
                st.session_state.selected_tender = tender_options[selected_tender_key]

                # Display tender info
                st.info(f"""
                **Tender ID:** {st.session_state.selected_tender['id']}
                **Company ID:** {st.session_state.selected_tender['company_id']}
                **Name:** {st.session_state.selected_tender['name']}
                """)

        # Number of extraction runs
        num_runs = st.number_input(
            "Number of Extraction Runs",
            min_value=1,
            max_value=10,
            value=3,
            help="Run the extraction multiple times"
        )

    # Tab 1: Requirements Extraction
    with tab1:
        if st.session_state.selected_tender:
            st.header("🚀 Run Requirements Extractions")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                if st.button("🚀 Run Requirements Extractions", type="primary", use_container_width=True):
                    st.session_state.extraction_runs = []

                    # Create a container for progress updates
                    progress_container = st.container()

                    with progress_container:
                        st.info(f"Starting {num_runs} parallel requirements extraction runs...")

                    # Progress tracking
                    overall_progress = st.progress(0, text=f"Starting {num_runs} extractions...")
                    run_status_placeholders = {}

                    # Create placeholders for each run status
                    for i in range(num_runs):
                        run_status_placeholders[i] = st.empty()
                        run_status_placeholders[i].info(f"Run {i+1}: Pending...")

                    # Function to run extraction
                    def run_extraction_task(run_num, tender_id, company_id):
                        try:
                            result = run_extraction(
                                str(tender_id),
                                str(company_id)
                            )

                            if result and result.get("requirements"):
                                run_data = {
                                    "run_number": run_num,
                                    "timestamp": datetime.now().isoformat(),
                                    "requirements": result["requirements"],
                                    "processing_time": result.get("processing_time", 0),
                                    "cost_info": result.get("cost_info", {}),
                                    "status": "success"
                                }
                                return run_data
                            else:
                                return {"run_number": run_num, "status": "failed", "error": "No requirements returned"}
                        except Exception as e:
                            return {"run_number": run_num, "status": "error", "error": str(e)}

                    # Execute extractions in parallel
                    with ThreadPoolExecutor(max_workers=min(num_runs, 5)) as executor:
                        # Submit all tasks with context
                        futures = {}
                        for i in range(num_runs):
                            future = executor.submit(
                                run_extraction_task,
                                i + 1,
                                st.session_state.selected_tender['id'],
                                st.session_state.selected_tender['company_id']
                            )
                            # Add Streamlit context to the thread
                            add_script_run_ctx(future)
                            futures[future] = i + 1

                        completed_count = 0
                        # Process completed tasks
                        for future in as_completed(futures):
                            run_num = futures[future]
                            try:
                                result = future.result()

                                # Update UI based on result
                                if result.get("status") == "success":
                                    run_status_placeholders[run_num-1].success(f"✅ Run {run_num}: Completed successfully")
                                    st.session_state.extraction_runs.append(result)
                                elif result.get("status") == "failed":
                                    run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Failed - {result.get('error', 'Unknown error')}")
                                else:
                                    run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Error - {result.get('error', 'Unknown error')}")

                                completed_count += 1
                                # Update overall progress
                                overall_progress.progress(
                                    completed_count / num_runs,
                                    text=f"Completed {completed_count}/{num_runs} extractions"
                                )
                            except Exception as e:
                                logger.error(f"Run {run_num} failed with exception: {e}")
                                run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Exception - {str(e)}")
                                completed_count += 1
                                overall_progress.progress(
                                    completed_count / num_runs,
                                    text=f"Completed {completed_count}/{num_runs} extractions"
                                )

                    # Sort results by run number for consistent display
                    st.session_state.extraction_runs.sort(key=lambda x: x['run_number'])

                    # Final status
                    overall_progress.progress(1.0, text=f"All {num_runs} requirements extractions completed!")
                    st.success(f"Completed {len(st.session_state.extraction_runs)} successful requirements extractions out of {num_runs} runs")

                    # Generate and download Excel file automatically
                    if st.session_state.extraction_runs:
                        excel_data = requirements_to_excel(st.session_state.extraction_runs)
                        if excel_data:
                            st.download_button(
                                label="📥 Download Requirements Results (Excel)",
                                data=excel_data,
                                file_name=f"requirements_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel_requirements"
                            )

            with col2:
                st.metric("Completed Runs", len(st.session_state.extraction_runs))

            with col3:
                if st.session_state.extraction_runs:
                    avg_time = sum(r.get("processing_time", 0) for r in st.session_state.extraction_runs) / len(st.session_state.extraction_runs)
                    avg_cost = sum(r.get("cost_info", {}).get("total_cost", 0) for r in st.session_state.extraction_runs) / len(st.session_state.extraction_runs)
                    st.metric("Avg Processing Time", f"{avg_time:.1f}s")
                    st.metric("Avg Total Cost", f"${avg_cost:.4f}")

            # Display previous results if available
            if st.session_state.extraction_runs and not st.button("🚀 Run Requirements Extractions", key="check_runs"):
                st.divider()
                st.subheader("📊 Previous Requirements Extraction Results")

                excel_data = requirements_to_excel(st.session_state.extraction_runs)
                if excel_data:
                    st.download_button(
                        label="📥 Download Previous Requirements Results (Excel)",
                        data=excel_data,
                        file_name=f"requirements_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_requirements_previous"
                    )
            
                # Display detailed results for each run
                st.subheader("🔍 Detailed Results per Run")
                for run in st.session_state.extraction_runs:
                    with st.expander(f"Run {run['run_number']} - {run['timestamp'][:19]}"):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Processing Time", f"{run.get('processing_time', 0):.1f}s")

                        cost_info = run.get('cost_info', {})
                        with col2:
                            st.metric("Extraction Cost", f"${cost_info.get('extraction_cost', 0):.4f}")

                        with col3:
                            st.metric("Dedup Cost", f"${cost_info.get('deduplication_cost', 0):.4f}")

                        with col4:
                            total_cost = cost_info.get('total_cost', 0)
                            st.metric("Total Cost", f"${total_cost:.4f}")
                    
                        # Cost breakdown
                        if cost_info:
                            st.write("**Cost Breakdown:**")
                            st.write(f"- Extraction (Gemini 2.5 Pro): ${cost_info.get('extraction_cost', 0):.4f}")
                            st.write(f"- Deduplication (Gemini 2.5 Pro): ${cost_info.get('deduplication_cost', 0):.4f}")
                            st.write(f"- Summary (Gemini 2.5 Flash): ${cost_info.get('summary_cost', 0):.4f}")

                        # Requirements count
                        requirements = run.get('requirements', {})
                        if requirements:
                            req_counts = []
                            req_counts.append(f"Ordine Generale: {len(requirements.get('requisiti_ordine_generale', []))}")
                            req_counts.append(f"Partecipazione: {len(requirements.get('requisiti_partecipazione', []))}")

                            ros = requirements.get('requisiti_ordine_speciale', {})
                            if ros:
                                req_counts.append(f"Idoneità Prof.: {len(ros.get('idoneita_professionale', {}).get('albi_registri_richiesti', []))}")
                                req_counts.append(f"Capacità Econ.: {len(ros.get('capacita_economico_finanziaria', {}).get('fatturato_minimo', []))}")
                                # Count certificazioni properly (they could be objects or strings)
                                certificazioni = ros.get('capacita_tecnico_professionale', {}).get('certificazioni', [])
                                esperienza = ros.get('capacita_tecnico_professionale', {}).get('esperienza_pregressa', [])
                                req_counts.append(f"Capacità Tecn.: {len(esperienza) + len(certificazioni)}")

                            st.write("**Requirements Count:** " + " | ".join(req_counts))
        else:
            st.info("👈 Please select a tender from the sidebar to begin requirements extraction")

    # Tab 2: Main Info Extraction
    with tab2:
        if st.session_state.selected_tender:
            st.header("🚀 Run Main Info Extractions")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                if st.button("🚀 Run Main Info Extractions", type="primary", use_container_width=True):
                    st.session_state.main_info_runs = []

                    # Create a container for progress updates
                    progress_container = st.container()

                    with progress_container:
                        st.info(f"Starting {num_runs} parallel main info extraction runs...")

                        # Progress tracking
                        overall_progress = st.progress(0, text=f"Starting {num_runs} main info extractions...")
                        run_status_placeholders = {}

                        # Create placeholders for each run status
                        for i in range(num_runs):
                            run_status_placeholders[i] = st.empty()
                            run_status_placeholders[i].info(f"Run {i+1}: Pending...")

                        # Function to run main info extraction task
                        def run_main_info_task(run_num, tender_id, company_id):
                            try:
                                result = run_main_info_extraction(
                                    str(tender_id),
                                    str(company_id)
                                )

                                if result and result.get("main_info"):
                                    run_data = {
                                        "run_number": run_num,
                                        "timestamp": datetime.now().isoformat(),
                                        "main_info": result["main_info"],
                                        "processing_time": result.get("processing_time", 0),
                                        "cost_info": result.get("cost_info", {}),
                                        "status": "success"
                                    }
                                    return run_data
                                else:
                                    return {"run_number": run_num, "status": "failed", "error": "No main info returned"}
                            except Exception as e:
                                return {"run_number": run_num, "status": "error", "error": str(e)}

                        # Execute extractions in parallel
                        with ThreadPoolExecutor(max_workers=min(num_runs, 5)) as executor:
                            # Submit all tasks with context
                            futures = {}
                            for i in range(num_runs):
                                future = executor.submit(
                                    run_main_info_task,
                                    i + 1,
                                    st.session_state.selected_tender['id'],
                                    st.session_state.selected_tender['company_id']
                                )
                                # Add Streamlit context to the thread
                                add_script_run_ctx(future)
                                futures[future] = i + 1

                            completed_count = 0
                            # Process completed tasks
                            for future in as_completed(futures):
                                run_num = futures[future]
                                try:
                                    result = future.result()

                                    # Update UI based on result
                                    if result.get("status") == "success":
                                        run_status_placeholders[run_num-1].success(f"✅ Run {run_num}: Completed successfully")
                                        st.session_state.main_info_runs.append(result)
                                    elif result.get("status") == "failed":
                                        run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Failed - {result.get('error', 'Unknown error')}")
                                    else:
                                        run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Error - {result.get('error', 'Unknown error')}")

                                    completed_count += 1
                                    # Update overall progress
                                    overall_progress.progress(
                                        completed_count / num_runs,
                                        text=f"Completed {completed_count}/{num_runs} main info extractions"
                                    )
                                except Exception as e:
                                    logger.error(f"Run {run_num} failed with exception: {e}")
                                    run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Exception - {str(e)}")
                                    completed_count += 1
                                    overall_progress.progress(
                                        completed_count / num_runs,
                                        text=f"Completed {completed_count}/{num_runs} main info extractions"
                                    )

                        # Sort results by run number for consistent display
                        st.session_state.main_info_runs.sort(key=lambda x: x['run_number'])

                        # Final status
                        overall_progress.progress(1.0, text=f"All {num_runs} main info extractions completed!")
                        st.success(f"Completed {len(st.session_state.main_info_runs)} successful main info extractions out of {num_runs} runs")

                        # Generate and download Excel file automatically
                        if st.session_state.main_info_runs:
                            excel_data = main_info_to_excel(st.session_state.main_info_runs)
                            if excel_data:
                                st.download_button(
                                    label="📥 Download Main Info Results (Excel)",
                                    data=excel_data,
                                    file_name=f"main_info_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel_main_info"
                                )

            with col2:
                st.metric("Completed Runs", len(st.session_state.main_info_runs))

            with col3:
                if st.session_state.main_info_runs:
                    avg_time = sum(r.get("processing_time", 0) for r in st.session_state.main_info_runs) / len(st.session_state.main_info_runs)
                    avg_cost = sum(r.get("cost_info", {}).get("total_cost", 0) for r in st.session_state.main_info_runs) / len(st.session_state.main_info_runs)
                    st.metric("Avg Processing Time", f"{avg_time:.1f}s")
                    st.metric("Avg Total Cost", f"${avg_cost:.4f}")

            # Display previous results if available
            if st.session_state.main_info_runs and not st.button("🚀 Run Main Info Extractions", key="check_main_info_runs"):
                st.divider()
                st.subheader("📊 Previous Main Info Extraction Results")

                excel_data = main_info_to_excel(st.session_state.main_info_runs)
                if excel_data:
                    st.download_button(
                        label="📥 Download Previous Main Info Results (Excel)",
                        data=excel_data,
                        file_name=f"main_info_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_main_info_previous"
                    )

                # Display detailed results for each run
                st.subheader("🔍 Detailed Results per Run")
                for run in st.session_state.main_info_runs:
                    with st.expander(f"Run {run['run_number']} - {run['timestamp'][:19]}"):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Processing Time", f"{run.get('processing_time', 0):.1f}s")

                        cost_info = run.get('cost_info', {})
                        with col2:
                            st.metric("Extraction Cost", f"${cost_info.get('extraction_cost', 0):.4f}")

                        with col3:
                            st.metric("Dedup Cost", f"${cost_info.get('deduplication_cost', 0):.4f}")

                        with col4:
                            total_cost = cost_info.get('total_cost', 0)
                            st.metric("Total Cost", f"${total_cost:.4f}")

                        # Main info summary
                        main_info = run.get('main_info', {})
                        if main_info:
                            st.write("**Extracted Information:**")
                            st.write(f"- CIG Count: {len(main_info.get('cig', []))}")
                            st.write(f"- Oggetto: {main_info.get('oggetto', 'N/A')[:100]}...")
                            st.write(f"- Stazione Appaltante: {main_info.get('stazione_appaltante', 'N/A')}")
                            st.write(f"- Tipo Procedura: {main_info.get('procedura', {}).get('tipo_procedura', 'N/A')}")
                            st.write(f"- Criterio Aggiudicazione: {main_info.get('procedura', {}).get('criterio_aggiudicazione', 'N/A')}")
                            st.write(f"- Importo Base: €{main_info.get('importi', {}).get('importo_base', 0):,.2f}")
        else:
            st.info("👈 Please select a tender from the sidebar to begin main info extraction")

    # Tab 3: Products Extraction
    with tab3:
        if st.session_state.selected_tender:
            st.header("🚀 Run Products Extractions")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                if st.button("🚀 Run Products Extractions", type="primary", use_container_width=True):
                    st.session_state.products_runs = []

                    # Create a container for progress updates
                    progress_container = st.container()

                    with progress_container:
                        st.info(f"Starting {num_runs} parallel products extraction runs...")

                        # Progress tracking
                        overall_progress = st.progress(0, text=f"Starting {num_runs} products extractions...")
                        run_status_placeholders = {}

                        # Create placeholders for each run status
                        for i in range(num_runs):
                            run_status_placeholders[i] = st.empty()
                            run_status_placeholders[i].info(f"Run {i+1}: Pending...")

                        # Function to run products extraction task
                        def run_products_task(run_num, tender_id, company_id):
                            try:
                                result = run_products_extraction(
                                    str(tender_id),
                                    str(company_id)
                                )

                                if result and result.get("products"):
                                    run_data = {
                                        "run_number": run_num,
                                        "timestamp": datetime.now().isoformat(),
                                        "products": result["products"],
                                        "processing_time": result.get("processing_time", 0),
                                        "cost_info": result.get("cost_info", {}),
                                        "status": "success"
                                    }
                                    return run_data
                                else:
                                    return {"run_number": run_num, "status": "failed", "error": "No products returned"}
                            except Exception as e:
                                return {"run_number": run_num, "status": "error", "error": str(e)}

                        # Execute extractions in parallel
                        with ThreadPoolExecutor(max_workers=min(num_runs, 5)) as executor:
                            # Submit all tasks with context
                            futures = {}
                            for i in range(num_runs):
                                future = executor.submit(
                                    run_products_task,
                                    i + 1,
                                    st.session_state.selected_tender['id'],
                                    st.session_state.selected_tender['company_id']
                                )
                                # Add Streamlit context to the thread
                                add_script_run_ctx(future)
                                futures[future] = i + 1

                            completed_count = 0
                            # Process completed tasks
                            for future in as_completed(futures):
                                run_num = futures[future]
                                try:
                                    result = future.result()

                                    # Update UI based on result
                                    if result.get("status") == "success":
                                        run_status_placeholders[run_num-1].success(f"✅ Run {run_num}: Completed successfully")
                                        st.session_state.products_runs.append(result)
                                    elif result.get("status") == "failed":
                                        run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Failed - {result.get('error', 'Unknown error')}")
                                    else:
                                        run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Error - {result.get('error', 'Unknown error')}")

                                    completed_count += 1
                                    # Update overall progress
                                    overall_progress.progress(
                                        completed_count / num_runs,
                                        text=f"Completed {completed_count}/{num_runs} products extractions"
                                    )
                                except Exception as e:
                                    logger.error(f"Run {run_num} failed with exception: {e}")
                                    run_status_placeholders[run_num-1].error(f"❌ Run {run_num}: Exception - {str(e)}")
                                    completed_count += 1
                                    overall_progress.progress(
                                        completed_count / num_runs,
                                        text=f"Completed {completed_count}/{num_runs} products extractions"
                                    )

                        # Sort results by run number for consistent display
                        st.session_state.products_runs.sort(key=lambda x: x['run_number'])

                        # Final status
                        overall_progress.progress(1.0, text=f"All {num_runs} products extractions completed!")
                        st.success(f"Completed {len(st.session_state.products_runs)} successful products extractions out of {num_runs} runs")

                        # Generate and download Excel file automatically
                        if st.session_state.products_runs:
                            excel_data = products_to_excel(st.session_state.products_runs)
                            if excel_data:
                                st.download_button(
                                    label="📥 Download Products Results (Excel)",
                                    data=excel_data,
                                    file_name=f"products_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel_products"
                                )

            with col2:
                st.metric("Completed Runs", len(st.session_state.products_runs))

            with col3:
                if st.session_state.products_runs:
                    avg_time = sum(r.get("processing_time", 0) for r in st.session_state.products_runs) / len(st.session_state.products_runs)
                    avg_cost = sum(r.get("cost_info", {}).get("total_cost", 0) for r in st.session_state.products_runs) / len(st.session_state.products_runs)
                    st.metric("Avg Processing Time", f"{avg_time:.1f}s")
                    st.metric("Avg Total Cost", f"${avg_cost:.4f}")

            # Display previous results if available
            if st.session_state.products_runs and not st.button("🚀 Run Products Extractions", key="check_products_runs"):
                st.divider()
                st.subheader("📊 Previous Products Extraction Results")

                excel_data = products_to_excel(st.session_state.products_runs)
                if excel_data:
                    st.download_button(
                        label="📥 Download Previous Products Results (Excel)",
                        data=excel_data,
                        file_name=f"products_results_{st.session_state.selected_tender['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_products_previous"
                    )

                # Display detailed results for each run
                st.subheader("🔍 Detailed Results per Run")
                for run in st.session_state.products_runs:
                    with st.expander(f"Run {run['run_number']} - {run['timestamp'][:19]}"):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Processing Time", f"{run.get('processing_time', 0):.1f}s")

                        cost_info = run.get('cost_info', {})
                        with col2:
                            st.metric("Extraction Cost", f"${cost_info.get('extraction_cost', 0):.4f}")

                        with col3:
                            st.metric("Alignment Cost", f"${cost_info.get('alignment_cost', 0):.4f}")

                        with col4:
                            total_cost = cost_info.get('total_cost', 0)
                            st.metric("Total Cost", f"${total_cost:.4f}")

                        # Products summary
                        products_data = run.get('products', {})
                        products_list = products_data.get('prodotti', [])
                        if products_list:
                            st.write("**Extracted Products:**")
                            st.write(f"- Products Count: {len(products_list)}")

                            # Show first 3 products as examples
                            for i, product in enumerate(products_list[:3], 1):
                                st.write(f"- Product {i}: {product.get('nome_prodotto', 'N/A')}")
                                st.write(f"  - Price: {product.get('prezzo_unitario', 'N/A')}")
                                st.write(f"  - Quantity: {product.get('quantita_richieste', 'N/A')}")
                                st.write(f"  - Lot: {product.get('lotto', 'N/A')}")

                            if len(products_list) > 3:
                                st.write(f"- ... and {len(products_list) - 3} more products")
                        else:
                            st.write("**No products extracted**")
        else:
            st.info("👈 Please select a tender from the sidebar to begin products extraction")

if __name__ == "__main__":
    main()
