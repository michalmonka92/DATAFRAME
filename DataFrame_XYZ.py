# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:45:19 2026

@author: monka
"""  
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import py3Dmol
import pickle
import pandas as pd
import os
import glob
import io
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds, Draw
import subprocess
from rdkit.Chem import rdmolfiles  
import shutil
import re
from natsort import natsorted
from io import BytesIO
from PIL import Image
import py3Dmol
import streamlit.components.v1 as components
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
import numpy as np 
import gdown
from analiza_podstawnikow import wykonaj_analize_L2
from stworz_mol_z_optymalizacji import stworz_mol_z_XYZ
import matplotlib.pyplot as plt
import seaborn as sns
        
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]
    
st.set_page_config(layout="wide")
@st.cache_data
@st.cache_data(ttl=1, show_spinner=False)
@st.cache_data(show_spinner=False)
def load_my_data():

    filename_4 = "DataFrame_S0_Optimized_Structures_FULL.pkl" 
    file_id4 = '1aHKJUIERhSzwIEEs4EtcqGshnB8DaXDL' #zoptymalizowane S0 struktury


    filename_0 = "Starting_Structures.pkl" 
    file_id0 = '12vrT1chxTl7_-81GoAD1vNqfVQ5TUZDT'
        
    filename_1 = "wyniki_obliczen1.pkl"
    file_id1 = '1qFMH8GqQPHyO7BZxF-wJOXNWScBusRkU'
        
    filename_2 = "DataFrame_Energies_FULL.pkl"
    file_id2 = "1Rngl3VouuX9kYVneiUGaJAJcYe3sdDKd"

    filename_3 = "DataFrame_Dihedrals.pkl"
    file_id3 = "1SeTbsc_NOPAqBegcc8pAJIRzCKpyerzQ"

    # Funkcja pomocnicza do pobierania
    def download_file(file_id, output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        # Jeśli plik istnieje, usuwamy go, aby gdown pobrał nową wersję
        if os.path.exists(output_name):
            os.remove(output_name)
        
        try:
            # gdown czasami potrzebuje parametru fuzzy=True dla GDrive
            gdown.download(url, output_name, quiet=False)
        except Exception as e:
            st.error(f"Błąd pobierania {output_name}: {e}")

    # Pobieranie plików
    with st.spinner('Synchronizacja z Google Drive...'):
        download_file(file_id0, filename_0)
        download_file(file_id1, filename_1)
        download_file(file_id2, filename_2)
        download_file(file_id3, filename_3)
        download_file(file_id4, filename_4)

    # Wczytywanie
    try:
        data_0 = pd.read_pickle(filename_0)
        data = pd.read_pickle(filename_1)
        data2 = pd.read_pickle(filename_2)
        data3 = pd.read_pickle(filename_3)
        data4 = pd.read_pickle(filename_4)
        return data_0,data, data2, data3, data4
    except Exception as e:
        st.error(f"Błąd wczytywania pkl: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Wywołanie danych
df0,df, df2, df3, df4 = load_my_data()

df['S0_MOL_Opt'] = df.apply(lambda x: stworz_mol_z_XYZ(x['Starting_Structure_MOL'], x['S0_XYZ_Opt']), axis=1)

#%%---------------------------------------------------------------------------------------Title--------------------------------------------------------------------------------------------------------------------------------
kolor_ramki = "#ff9300"  # Twój kolor (np. niebieski)
kolor_tla = "#363636"    # Jasny odcień dla wypełnienia
pomarancz = "#ff9300"

st.markdown("""<style>
    /* Celujemy w tekst nagłówka expandera */
    .stExpander details summary p {
        color: cyan !important;
        font-weight: bold;
        font-size: 1.1rem;}
    </style>
    """, unsafe_allow_html=True)
st.markdown(f"""<hr style="height:5px;margin-top: -1px; border:none; color:{kolor_tla}; background-color:{kolor_tla};" />""", unsafe_allow_html=True)   
st.markdown(f"""
        <style>
        .moja-ramka1 {{ 
        border-radius: 14px;
        padding: 25px;
        background-color: {kolor_tla};
        color=#ffffff;
        text-align: center;
        height: 90px;
    }}
    .moja-ramka1 h4 {{
        color: "#ffffff";
        font-size: 18px;
        margin: -5px;
    }}
    </style>
    
    <div class="moja-ramka1">
        <h4>TADF dataset for Machine Learning</h4>
    </div>
    """, unsafe_allow_html=True) 
st.markdown(" ") # Bardzo mały odstęp
cola,colb=st.columns([4.7,5])
with cola:
           

        st.markdown(f"""
        <style>
        .moja-ramka {{ 
        border-radius: 10px;
        padding: 30px;
        background-color: {kolor_tla};
        text-align: center;
        height: 40px;
    }}
    .moja-ramka h4 {{
        color: {kolor_ramki};
        font-size: 18px;
        margin: 15px;
    }}
    </style>
    
    <div class="moja-ramka">
        <h4>Linker modifications</h4>
    </div>
    """, unsafe_allow_html=True)  
        st.markdown(f"""<hr style="height:5px;margin-top: -3px; border:none; color:{pomarancz}; background-color:{pomarancz};" />""", unsafe_allow_html=True)
        image = Image.open('linkers.jpg')
        st.image(image, use_container_width=True)     

with colb:
        st.markdown(f"""
        <style>
        .moja-ramka {{ 
        border-radius: 10px;
        padding: 0px;
        background-color: {kolor_tla};
        text-align: center;
        height: 40px;
    }}
    .moja-ramka h4 {{
        color: {kolor_ramki};
        font-size: 18px;
        margin: 0;
    }}
    </style>
    
    <div class="moja-ramka">
        <h4>Substituents</h4>
    </div>
    """, unsafe_allow_html=True)
        st.markdown(f"""<hr style="height:5px;margin-top: -3px; border:none; color:{pomarancz}; background-color:{pomarancz};" />""", unsafe_allow_html=True)

        image = Image.open('Subs.jpg')
        st.image(image, use_container_width=True)      


#%%-------------------------------------------------------------------------------------------------INPUT DATAFRAME STARTING STRUCTURES----------------------------------------------------------------------------------

st.markdown("""<hr style="height:5px; border:none; color:#444444; background-color:#444444;" />""", unsafe_allow_html=True)      
with st.expander("Input DataFrame: Starting Structures (from Dejan) ", expanded=False):
        cola,colb=st.columns([3,8])
        with cola:
                st.markdown('<span style="color: #ff9300; font-weight: bold;">Input DataFrame</span>', unsafe_allow_html=True)
                st.dataframe(df0)

                @st.cache_data
                def convert_df(df0):
                    return df0.to_csv(index=False, sep=';').encode('utf-8-sig') # utf-8-sig pomaga na polskie znaki

                csv = convert_df(df0)


                st.markdown("""
                    <style>
                    /* Celujemy w przycisk pobierania */
                    .stDownloadButton > button {
                        background-color: #FFA500; /* Twój pomarańczowy */
                        color: white;              /* Kolor tekstu */
                        font-weight: bold;         /* POGRUBIENIE */
                        font-size: 16px;           /* Wielkość czcionki */
                        padding: 0.6em 1.2em;
                        border-radius: 7px;       /* Zaokrąglone rogi */
                        border: none;
                        width: 100%;               /* Przycisk na pełną szerokość kolumny */
                        transition: 0.3s;          /* Płynna zmiana koloru */
                    }
                
                    /* Efekt po najechaniu myszką (hover) */
                    .stDownloadButton > button:hover {
                        background-color: #FF8C00; /* Ciemniejszy pomarańcz */
                        color: #424242;
                        border: none;
                    }
                    
                    /* Efekt po kliknięciu (active) */
                    .stDownloadButton > button:active {
                        background-color: #CC7000;
                        transform: scale(0.98);    /* Efekt delikatnego wciśnięcia */
                    }
                    </style>
                    """, unsafe_allow_html=True)
                
                buffer = io.BytesIO()
                pickle.dump(df0, buffer)
                pkl_data = buffer.getvalue()


                col1,col2,col3=st.columns([1.5,2,0.5])
                with col1:
                        st.download_button(
                    label="Download as *csv",
                    data=csv,
                    file_name='tadf_data.csv',
                    mime='text/csv',
                )
                with col2:
                        st.download_button(
                    label="Download as *pkl (Pickle)",
                    data=pkl_data,
                    file_name="tadf_data.pkl",
                    mime="application/octet-stream"  # Standardowy typ dla plików binarnych
                )
                with col3:
                        print()
                                

        
        with colb:
                st.markdown('<span style="color: #ff9300; font-weight: bold;">Description and Summary</span>', unsafe_allow_html=True)
                st.text("""
This dataset contains starting structures (pre-optimized) of TADF emitters. All data points are aggregated into the `Starting_Structures.pkl` dataframe, which you can download as csv or pkl. 
""")
                stats = df0.describe()
                stats=stats.drop(['top','freq'], axis=0)

                stats_ = stats.style.set_table_styles([
                    {'selector': 'th',
                     'props': [
                     ('background-color', '#ff9300'), # Ciemne tło nagłówka
                     ('color', 'black'),             # POMARAŃCZOWY KOLOR CZCIONKI
                     ('font-family', 'cambria'),
                     ('font-weight', 'bold'),
                     ('border', '1px solid #363636'),
                     ('font-size', '14px')
                     ]
                    }
                ])
                st.table(stats_)

                with st.expander("2D & 3D Structure Preview", expanded=False):
                        col_a,col_b,col_c,col_d=st.columns([5,0.1,2,0.1])
                        with col_a:    
                                if 'Linker' in df0.columns:
                                        available_ls = df0['Linker'].unique()
                                        available_ls = natsorted(available_ls)
                                        selected_l = st.selectbox(
                                            "Select the linker mod:", 
                                            available_ls, 
                                            key="selectbox_linker_main" # Dodaj unikalny ciąg znaków
                                        )
                                        
                                        df0_filtered = df0[df0['Linker'] == selected_l].copy()
                                        df0_filtered['sort_key'] = df0_filtered['ID'].apply(natural_sort_key)
                                        df0_filtered = df0_filtered.sort_values(by='sort_key').drop(columns=['sort_key'])
                                else:
                                        df0_filtered = df0.head(16)
                                
                                current_id0 = st.session_state.get('selected_id0', df['ID'].iloc[0])
                                selected_row0 = df0[df0['ID'] == current_id0].iloc[0] 
                                
                        
                                n_cols_gal = 8  # 4 kolumny wewnątrz lewego panelu
                                    
                                    # Grupowanie wierszy galerii
                                gallery_rows = [df0_filtered[i:i + n_cols_gal] for i in range(0, len(df0_filtered), n_cols_gal)]
                                    
                                for row_data in gallery_rows:
                                        cols = st.columns(n_cols_gal)
                                        for i, (idx, row) in enumerate(row_data.iterrows()):
                                            with cols[i]:
                                                m = row['Starting_Structure_MOL']
                                                if m:
                                                    # 1. Przygotowanie obrazka
                                                    m_2d = Chem.Mol(m)
                                                    m_2d = Chem.RemoveHs(m_2d)
                                                    AllChem.Compute2DCoords(m_2d)
                                                    img = Draw.MolToImage(m_2d, size=(300, 300))
                                                    img = img.rotate(90, expand=True)
                                                    st.image(img, use_container_width=True)
                                                    
                                                    # 2. Podpis ID
                                                    st.markdown(f'<div style="text-align:center; font-size:14px;font-weight:bold; color:{pomarancz};">{row["ID"]}</div>', unsafe_allow_html=True)
                                                    is_active0 = (row['ID'] == current_id0)
                                                    # KOMPLETNY CSS: kolorowanie aktywnego oraz zmniejszenie wysokości
                                                    st.markdown("""
                                                        <style>
                                                        /* 1. Styl dla przycisku wybranego (Zielony) */
                                                        div[data-testid="stButton"] button[kind="primary"] {
                                                            background-color:  #ff9300 !important;
                                                            color: white !important;
                                                            border-color: #ff9300 !important;
                                                        }
                                                
                                                        /* 2. Zmniejszenie wysokości wszystkich przycisków w galerii */
                                                        div[data-testid="stButton"] button {
                                                            padding-top: 0px !important;
                                                            padding-bottom: 0px !important;
                                                            height: 24px !important; /* Tutaj kontrolujesz wysokość w osi Y */
                                                            min-height: 24px !important;
                                                            line-height: 24px !important;
                                                        }
                                                        
                                                        /* 3. Wycentrowanie ikony ptaszka w mniejszym przycisku */
                                                        div[data-testid="stButton"] button p {
                                                            font-size: 14px !important;
                                                            margin-top: -2px !important;
                                                        }
                                                        </style>
                                                    """, unsafe_allow_html=True)

                                                    # Jeśli aktywny, dajemy type="primary" (zadziała Twój CSS), jeśli nie - "secondary"
                                                    btn_type = "primary" if is_active0 else "secondary"
                                                    
                                                    if st.button("✔", key=f"gal_{row['ID']}", use_container_width=True, type=btn_type):
                                                        st.session_state['selected_id0'] = row['ID']
                                                        st.rerun()
        
                                        
                                st.text(" ")
                        with col_b:
                                pass
                        with col_c:        
                                st.markdown(f"""
                                <div style="font-size: 16px; font-weight: bold; margin-bottom: 15px;">Structure: <span style="color: #ff9300;">{current_id0}</span>
                                  </div>
                                    """, unsafe_allow_html=True)

                                st.markdown(f"""
                                <div style="
                                    background-color: {kolor_tla}; 
                                    border-radius: 10px; 
                                    padding: 1px;
                                    margin-left: -10px;
                                    margin-right: -14px;
                                    margin-bottom: -590px; /* Trik, żeby 'podłożyć' tło pod wykres */
                                    height: 450px;
                                ">
                                </div>
                            """, unsafe_allow_html=True)
                                
                                show_h_3d0 = st.checkbox("Show hydrogens (H)", value=True, key=f"h_3d0_{current_id0}")
                                # Pobieramy obiekt MOL
                                mol_start0 = selected_row0.get('Starting_Structure_MOL')
                                if mol_start0:
                                # Tworzymy kopię, aby nie modyfikować oryginału w DataFrame
                                        view_mol0 = Chem.Mol(mol_start0)
                                
                                # Konwersja obiektu RDKit na blok tekstowy MOL
                                        mol_block0 = Chem.MolToMolBlock(view_mol0)
                                        view = py3Dmol.view(width=500, height=350)
                                        view.addModel(mol_block0, 'mol')
                                        view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.1}, 
                                                   'sphere': {'colorscheme': 'Jmol', 'radius': 0.3}})
                                        if not show_h_3d0:
                                                view.setStyle({'elem': 'H'}, {}) 
                                        view.setBackgroundColor("#363636")
                                        view.zoomTo()

                        
                                # Render
                                        obj0 = view._make_html()
                                # Zwiększyłem wysokość komponentu, by pasowała do widoku
                                        components.html(obj0, height=400, width=610)
                                else:
                                        st.error("Brak obiektu MOL (Starting_Structure_MOL) dla tej cząsteczki.")   
                        with col_d:
                                pass

#%%----------------------------------------------------------------------------------------------------PO S0 OPTYMALIZACJI------------------------------------------------------------------------------
with st.expander("S0-optimization DataFrame (after DFT)", expanded=False):
        cola,colb=st.columns([3,8])
        with cola:
                st.markdown('<span style="color: #ff9300; font-weight: bold;">DataFrame with structures after S0-geometry optimization</span>', unsafe_allow_html=True)
                st.dataframe(df4)
                @st.cache_data
                def convert_df4(df4):
                    return df4.to_csv(index=False, sep=';').encode('utf-8-sig') # utf-8-sig pomaga na polskie znaki

                csv4 = convert_df4(df4)
                buffer4 = io.BytesIO()
                pickle.dump(df4, buffer4)
                pkl_data4 = buffer4.getvalue()

                col1,col2,col3=st.columns([1.5,2,0.5])
                with col1:
                        st.download_button(
                    label="Download as *csv",
                    data=csv4,
                    file_name='S0_Optimized_Structures_FULL.csv',
                    mime='text/csv',
                    key="btn_download_csv",  # UNIKALNY KLUCZ
                )
                with col2:
                        st.download_button(
                    label="Download as *pkl (Pickle)",
                    data=pkl_data4,
                    file_name="DataFrame_S0_Optimized_Structures_FULL.pkl",
                    mime="application/octet-stream",  # Standardowy typ dla plików binarnych
                    key="btn_download_pickle",  # UNIKALNY KLUCZ
                )
                with col3:
                        print()


        



        
        with colb:
                st.text("""
                This dataset contains S0-optimized structures. All data points are aggregated into the `DataFrame_S0_Optimized_Structures_FULL.pkl` dataframe, which you can download as csv or pkl. 
""")
                stats = df4.describe()
                stats=stats.drop(['top','freq'], axis=0)

                stats_ = stats.style.set_table_styles([
                    {'selector': 'th',
                     'props': [
                     ('background-color', '#ff9300'), # Ciemne tło nagłówka
                     ('color', 'black'),             # POMARAŃCZOWY KOLOR CZCIONKI
                     ('font-family', 'cambria'),
                     ('font-weight', 'bold'),
                     ('border', '1px solid #363636'),
                     ('font-size', '14px')
                     ]
                    }
                ])
                st.table(stats_)

                st.markdown("""<hr style="height:5px; border:none; color:#444444; background-color:#444444;" />""", unsafe_allow_html=True)           
                with st.expander("Preview", expanded=False):
                        main_col_left, main_margin1,main_col_mid,main_margin2, main_col_right = st.columns([4,0.05, 1.7,0.05, 0.9])
                        with main_col_left:
                                if 'Linker' in df.columns:
                                        available_ls = df['Linker'].unique()
                                        available_ls = natsorted(available_ls)
                                        selected_linker = st.selectbox("Wybierz typ modyfikacji linkera", available_ls)
                                        df_filtered = df[df['Linker'] == selected_linker].copy()
                                        df_filtered['sort_key'] = df_filtered['ID'].apply(natural_sort_key)
                                        df_filtered = df_filtered.sort_values(by='sort_key').drop(columns=['sort_key'])
                                else:
                                        df_filtered = df.head(16)
                        
                                current_id = st.session_state.get('selected_id', df['ID'].iloc[0])
                                selected_row = df[df['ID'] == current_id].iloc[0]
                        
                                st.markdown("""
                                <style>
                                /* 1. Styl dla przycisku wybranego (Zielony) */
                                div[data-testid="stButton"] button[kind="primary"] {
                                    background-color:  #ff9300 !important;
                                    color: white !important;
                                    border-color: #ff9300 !important;
                                }
                        
                                /* 2. Zmniejszenie wysokości wszystkich przycisków w galerii */
                                div[data-testid="stButton"] button {
                                    padding-top: 0px !important;
                                    padding-bottom: 0px !important;
                                    height: 24px !important; /* Tutaj kontrolujesz wysokość w osi Y */
                                    min-height: 24px !important;
                                    line-height: 24px !important;
                                }
                                
                                /* 3. Wycentrowanie ikony ptaszka w mniejszym przycisku */
                                div[data-testid="stButton"] button p {
                                    font-size: 14px !important;
                                    margin-top: -2px !important;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                                n_cols_gal = 8  # 4 kolumny wewnątrz lewego panelu
                                gallery_rows = [df_filtered[i:i + n_cols_gal] for i in range(0, len(df_filtered), n_cols_gal)]
                                for row_data in gallery_rows:
                                        cols = st.columns(n_cols_gal)
                                        for i, (idx, row) in enumerate(row_data.iterrows()):
                                            with cols[i]:
                                                m = row['Starting_Structure_MOL']
                                                if m:
                                                    # 1. Przygotowanie obrazka
                                                    m_2d = Chem.Mol(m)
                                                    m_2d = Chem.RemoveHs(m_2d)
                                                    AllChem.Compute2DCoords(m_2d)
                                                    img = Draw.MolToImage(m_2d, size=(200, 200))
                                                    img = img.rotate(90, expand=True)
                                                    st.image(img, use_container_width=True)
                                            
                                                    # 2. Podpis ID
                                                    st.markdown(f'<div style="text-align:center; font-size:11px; color:#ffffff;">{row["ID"]}</div>', unsafe_allow_html=True)
                                            
                                                    # --- KLUCZOWA ZMIANA TUTAJ ---
                                                    # Sprawdzamy, czy ten wiersz jest tym wybranym
                                                    is_active = (row['ID'] == current_id)
                                            
                                                    # Jeśli aktywny, dajemy type="primary" (zadziała Twój CSS), jeśli nie - "secondary"
                                                    btn_type = "primary" if is_active else "secondary"
                                                    
                                                    if st.button("✔", key=f"gal_{row['ID']}", use_container_width=True, type=btn_type):
                                                        st.session_state['selected_id'] = row['ID']
                                                        st.rerun()
                        
                        with main_margin1:
                            pass
                            
                            
                        with main_col_right:
                            # Sekcja ustawień pozostaje bez zmian (używamy Twoich domyślnych wartości)
                            st.markdown(f"""
                                <div style="font-size: 16px; font-weight: bold; margin-bottom: 15px;">Struktura 2D: <span style="color: #ff9300;">{current_id}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                            mol_start = selected_row.get('Starting_Structure_MOL')
                            mol_start_2d = Chem.Mol(mol_start)
                            mol_start_2d = Chem.RemoveHs(mol_start_2d)
                            AllChem.Compute2DCoords(mol_start_2d)
                            img = Draw.MolToImage(mol_start_2d, size=(400, 400))
                            st.image(img, use_container_width=True)
                            
                            
                            with st.expander("Wygląd 3D", expanded=True):
                                thickness = st.slider("Grubość wiązań:", 0.05, 0.6, 0.15, 0.05, key=f"thick_{current_id}")
                                show_h_3d = st.checkbox("Pokaż wodory (H)", value=True, key=f"h_3d_{current_id}")
                                bg_color = st.select_slider("Tło:", options=["white", "#363636", "black"], value="#363636", key=f"bg_{current_id}")
                        
                        with main_margin2:
                            pass
                            
                        # --- 3. ŚRODKOWA KOLUMNA: WIZUALIZACJA 3D ---
                        with main_col_mid:
                            
                            st.markdown(f"""
                                <div style="font-size: 16px; font-weight: bold; margin-bottom: 15px;">Struktura 3D: <span style="color: #ff9300;">{current_id}</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div style="
                                    background-color: {kolor_tla}; 
                                    border-radius: 10px; 
                                    padding: 1px;
                                    margin-left: -10px;
                                    margin-right: -14px;
                                    margin-bottom: -590px; /* Trik, żeby 'podłożyć' tło pod wykres */
                                    height: 450px;
                                ">
                                </div>
                            """, unsafe_allow_html=True)
                            tab1, tab2, tab3 = st.tabs(["Starting_Structure", "S0-Optimized_Structure", "Overlay"])
                            with tab1:
                                # W sekcji wizualizacji (main_col_mid):
                                # Pobieramy obiekt MOL
                                mol_start = selected_row.get('Starting_Structure_MOL')
                            
                                if mol_start:
                                # Tworzymy kopię, aby nie modyfikować oryginału w DataFrame
                                    view_mol = Chem.Mol(mol_start)
                                
                                # Konwersja obiektu RDKit na blok tekstowy MOL
                                    mol_block = Chem.MolToMolBlock(view_mol)
                                
                                    view = py3Dmol.view(width=500, height=350)
                                # Ważne: zmieniamy format na 'mol'
                                    view.addModel(mol_block, 'mol')
                        
                                # Ustawiamy styl stick
                                    view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': thickness}, 
                                                   'sphere': {'colorscheme': 'Jmol', 'radius': 0.3}})
                                
                                # Obsługa ukrywania wodorów
                                    if not show_h_3d:
                                        view.setStyle({'elem': 'H'}, {}) 
                                    view.zoomTo()
                                    view.setBackgroundColor(bg_color)
                                # Render
                                    obj = view._make_html()
                                # Zwiększyłem wysokość komponentu, by pasowała do widoku
                                    components.html(obj, height=400, width=610)
                                else:
                                        st.error("Brak obiektu MOL (Starting_Structure_MOL) dla tej cząsteczki.")   
                            with tab2:
                            # Pobieramy obiekt MOL
                                mol_start = selected_row.get('S0_MOL_Opt')
                                if mol_start:
                                # Tworzymy kopię, aby nie modyfikować oryginału w DataFrame
                                    view_mol = Chem.Mol(mol_start)
                                
                                # Konwersja obiektu RDKit na blok tekstowy MOL
                                    mol_block = Chem.MolToMolBlock(view_mol)
                                
                                    view = py3Dmol.view(width=500, height=350)
                                # Ważne: zmieniamy format na 'mol'
                                    view.addModel(mol_block, 'mol')
                                
                                # Ustawiamy styl stick
                                    view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': thickness}, 
                                                   'sphere': {'colorscheme': 'Jmol', 'radius': 0.3}})
                                
                                # Obsługa ukrywania wodorów
                                    if not show_h_3d:
                                        view.setStyle({'elem': 'H'}, {}) 
                                    
                                    view.zoomTo()
                                    view.setBackgroundColor(bg_color)
                                
                                # Render
                                    obj = view._make_html()
                                # Zwiększyłem wysokość komponentu, by pasowała do widoku
                                    components.html(obj, height=400, width=610)
                                else:
                                        st.error("Brak obiektu MOL (S0_MOL_Opt) dla tej cząsteczki.")
                                

                                st.markdown("""<hr style="height:5px; border:none; color:#444444; background-color:#444444;" />""", unsafe_allow_html=True)
                                
                        with tab3:
                                start = selected_row.get('Starting_Structure_MOL')
                                mol_opt = selected_row.get('S0_MOL_Opt') # Zakładam, że tu jest obiekt RDKit Mol
                                
                                if mol_start and mol_opt:
                            # Tworzymy kopie, żeby nie psuć oryginałów
                                        m1 = Chem.Mol(mol_start)
                                        m2 = Chem.Mol(mol_opt)
                            # Usunięcie wodorów do obliczeń, jeśli użytkownik tak wybrał (opcjonalnie)
                                        if not show_h_3d:
                                            m1 = Chem.RemoveHs(m1)
                                            m2 = Chem.RemoveHs(m2)
                            # --- NAKŁADANIE (ALIGNMENT) ---
                            # Dopasowujemy m2 (zoptymalizowaną) do m1 (startowej)
                                        rmsd = rdMolAlign.AlignMol(m2, m1)
                            # Przygotowanie bloków tekstowych
                                        block1 = Chem.MolToMolBlock(m1)
                                        block2 = Chem.MolToMolBlock(m2)
                        
                            # --- WIZUALIZACJA py3Dmol ---
                                        view = py3Dmol.view(width=600, height=350)
                            
                            # Dodajemy pierwszą strukturę (np. szara)
                                        view.addModel(block1, 'mol')
                                        view.setStyle({'model': 0}, {'stick': {'color': '#919191', 'radius': thickness}})
                            
                            # Dodajemy drugą strukturę (np. Twoje kolory Jmol lub konkretny kolor)
                                        view.addModel(block2, 'mol')
                                        view.setStyle({'model': 1}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': thickness}})                                
                                    # Obsługa wodorów
                                        if not show_h_3d:
                                            view.setStyle({'elem': 'H'}, {})
                                        view.zoomTo()
                                        view.setBackgroundColor(bg_color)
                                        obj = view._make_html()
                                        components.html(obj, height=400, width=610)
                                else:
                                        st.error("Brak jednej ze struktur (Startowej lub Zoptymalizowanej) do nałożenia.")
                        










# Zakładamy, że selected_id to ID wybrane przez użytkownika (np. ze st.selectbox)
row = df[df['ID'] == current_id].iloc[0]

# 1. Pobranie danych z nowej kolumny
vibrations = row.get('Vibrational_Vectors', None)

       

# --- FUNKCJA POMOCNICZA DLA ANIMACJI ---
def create_vibration_xyz(mol, vectors, amplitude=2, num_frames=45):
    if not vectors or mol is None:
        return None
    xyz_frames = ""
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    for i in range(num_frames):
        scale = amplitude * np.sin(2 * np.pi * i / num_frames)
        xyz_frames += f"{mol.GetNumAtoms()}\nFrame {i}\n"
        for a_idx, sym in enumerate(symbols):
            pos = conf.GetAtomPosition(a_idx)
            new_x = pos.x + vectors[a_idx][0] * scale
            new_y = pos.y + vectors[a_idx][1] * scale
            new_z = pos.z + vectors[a_idx][2] * scale
            xyz_frames += f"{sym} {new_x:10.6f} {new_y:10.6f} {new_z:10.6f}\n"
    return xyz_frames


with st.expander("Frequency Analysis",expanded=False):
# --- WIDOK W STREAMLIT ---
    row = df[df['ID'] == current_id].iloc[0]
    vibrations = row.get('Vibrational_Vectors', [])

    if isinstance(vibrations, list) and len(vibrations) > 0:
        col_list, col_view, col_param = st.columns([1, 2,1])
    
        with col_list:
            st.write("### 📋 Lista Modów")
            # Wybór modu
            mode_options = [v['mode_index'] for v in vibrations]
            selected_mode_idx = st.radio(
                "Wybierz drganie do animacji:",
                options=mode_options,
                format_func=lambda x: f"Mod {x}: {next(v['frequency'] for v in vibrations if v['mode_index'] == x):.2f} cm⁻¹"
            )
            
            # Pobranie danych wybranego modu

    
        with col_param:
            current_mode = next(v for v in vibrations if v['mode_index'] == selected_mode_idx)
            freq_val = current_mode['frequency']
            
            st.metric("Częstotliwość", f"{freq_val:.2f} cm⁻¹")
            amp = st.slider("Amplituda drgań", 0.1, 5.0, 4.0, key="amp_slider")
        
        with col_view:
            st.write("### 🎬 Podgląd 3D")
            
            # Generowanie klatek animacji dla wybranego modu
            # Upewnij się, że w S0_MOL_Opt masz obiekt RDKit
            vibr_xyz = create_vibration_xyz(row['S0_MOL_Opt'], current_mode['vectors'], amplitude=amp)
            
            if vibr_xyz:
                view = py3Dmol.view(width=500, height=400)
                view.addModelsAsFrames(vibr_xyz, 'xyz')
                view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.1}, 
                               'sphere': {'colorscheme': 'Jmol', 'radius': 0.3}})
                view.setBackgroundColor(kolor_tla)
                view.animate({
        'loop': 'forward', # lub 'backAndForth' dla naturalnego ruchu
        'step': 3,       # mniejsza wartość = znacznie większa szybkość i płynność
        'reps': 0          # 0 oznacza nieskończoną liczbę powtórzeń (continuous)
    })
                view.zoomTo()
                
                tjs = view._make_html()
                components.html(tjs, height=420)
            else:
                st.error("Błąd generowania klatek XYZ.")
        
                
                
                
    else:
        st.warning("Brak danych wibracyjnych dla tego emitera.")

#%%------------------------------------------------------------------------------------ENERGIE-------------------------------------------------------------------------------------------------------------------

#1. Panel wyboru kolumn po prawej stronie
with st.expander("Energies", expanded=False):
    cola,colb=st.columns([1,4])
    with cola:   # Sprawdzamy czy df2 nie jest pusty
        if not df2.empty:
            # Automatycznie wykrywamy kolumny numeryczne do kolorowania
            numeric_cols = df2.select_dtypes(include=['number']).columns
            
            # Tworzymy ostylowany widok dla całego dataframe
            styled_df = df2.style.background_gradient(
                cmap='coolwarm', 
                subset=numeric_cols
            ).format(precision=3) # Zaokrąglenie wszystkich liczb do 4 miejsc
            
            # Wyświetlamy tabelę na pełną szerokość
            st.dataframe(
                styled_df, 
                height=600, 
                use_container_width=True)
            
    with colb:
        tab1, tab2, tab3 = st.tabs(["Energy: S1", "Energy: T1", "Energies: S1-T1"])
        with tab1:
            coll,colr=st.columns([2,2])
            with coll:
                def natural_key(string_):
                    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
                heatmap_dataS1 = df2.pivot_table(index="Linker", 
                                                    columns="Substituent", 
                                                    values="S1", 
                                                    aggfunc='mean')
                # 3. Sortowanie osi
                sorted_linkers = sorted(heatmap_dataS1.index, key=natural_key)
                sorted_substituents = sorted(heatmap_dataS1.columns, key=natural_key)
                heatmap_dataS1 = heatmap_dataS1.reindex(index=sorted_linkers, columns=sorted_substituents)
                
                fig1 = px.imshow(heatmap_dataS1,
                    labels=dict(x="Podstawnik", y="Linker", color="S1 [eV]"),
                    x=sorted_substituents,
                    y=sorted_linkers,
                    color_continuous_scale="jet", # Twoja ulubiona paleta
                    range_color=[float(heatmap_dataS1.min().min()), float(heatmap_dataS1.max().max())],             # Twoje skalowanie
                    text_auto=".2f",                  # Wyświetlanie wartości w kratkach
                    aspect="auto"                     # Automatyczne dopasowanie proporcji
                )
                
                # 5. Estetyka wykresu
                fig1.update_layout(
                    xaxis_nticks=len(sorted_substituents),
                    yaxis_nticks=len(sorted_linkers),
                    width=900, 
                    height=400)
                
                # 6. Wyświetlenie w Streamlit
                st.plotly_chart(fig1, use_container_width=True)
            with colr:
                     # 1. Funkcja pomocnicza
                def get_number(text):
                    if not isinstance(text, str): return 0
                    match = re.search(r'\d+', text)
                    return int(match.group()) if match else 0
            
            # --- INTERFEJS WYBORU ---
                st.markdown(f"""
        <div style="
            background-color: {kolor_tla}; 
            border-radius: 10px; 
            padding: 1px;
            margin-left: -10px;
            margin-right: 0px;
            margin-bottom: -590px; /* Trik, żeby 'podłożyć' tło pod wykres */
            height: 450px;
        ">
        </div>
    """, unsafe_allow_html=True)
                tabl, tabr= st.tabs(["Group by Linker", "Group by Substutuent"])
                with tabl:

                    sort_option = st.radio(
                        "Sorting by",
                        ["Energy S1 (descending)","Substituent order (R1 -> R16)"],
                        horizontal=True) 


                    # 2. Przygotowanie danych
                    # Zawsze potrzebujemy numeru R do głównego grupowania
                    df2['R_num'] = df2['Substituent'].apply(get_number)
                    df2['L_num'] = df2['Linker'].apply(get_number)
                    
                    if sort_option == "Substituent order (R1 -> R16)":
                        # Sortujemy: R rosnąco, potem Linker rosnąco
                        df_plot = df2.sort_values(by=['R_num', 'L_num'], ascending=[True, True]).copy()  
                    else:
                        # Sortujemy: R rosnąco, potem Kąt malejąco
                        df_plot = df2.sort_values(by=['S1', 'L_num'], ascending=[True, True]).copy()
                    
                    # Stała kolejność w legendzie (L2, L3...)
                    sorted_linkers = sorted(df_plot['Linker'].unique(), key=get_number)
                    
                    # 3. Tworzenie wykresu Plotly
                    fig2 = px.scatter(
                        df_plot,
                        x='ID',
                        y='S1',
                        color='Linker',
                        category_orders={"Linker": sorted_linkers}, 
                        labels={
                            'ID': 'ID Związku',
                            'S1': 'S1 [eV]',
                            'Linker': 'Linker'
                        },
                        hover_data=['Linker', 'Substituent', 'S1']
                    )
                    
                    # 4. Stylizacja
                    fig2.update_traces(marker=dict(size=11, line=dict(width=1, color='white')))
                    fig2.update_layout(
                                xaxis=dict(
                                    showticklabels=False, # TO UKRYWA PODPISY (R1-L2-cośtam)
                                    showgrid=False,       # Opcjonalnie: ukrywa pionowe linie siatki
                                    title=None            # Ukrywa napis "ID Związku"
                                ),
                        yaxis=dict(
                            showgrid=True,           # Włącza linie siatki
                            gridcolor='white',      # Kolor linii siatki
                            gridwidth=0.5,           # Grubość linii
                            zerolinecolor='red',  # Linia zerowa też na pomarańczowo
                            dtick=0.5                # Co ile ma być linia (np. co 0.5 eV)
                        ),
                                template='plotly_dark' if bg_color in ["#111111", "#000000", "#2D2D2D"] else 'plotly_white',
                                plot_bgcolor=bg_color,
                                paper_bgcolor=bg_color,
                                margin=dict(l=60, r=10, t=30, b=10),
                                height=300 # Możesz teraz zmniejszyć wysokość, bo nie ma napisów na dole
                            )
                
                    
                    # 5. Wyświetlenie
                    st.plotly_chart(fig2, use_container_width=True)
                with tabr:
                    sort_option = st.radio(
                        "Sorting by",
                        ["Energy S1 (descending)","Linker order (L2 -> L10)"],
                        horizontal=True) 
                    
                    # 2. Przygotowanie danych
                    # Zawsze potrzebujemy numeru R do głównego grupowania
                    df2['R_num'] = df2['Substituent'].apply(get_number)
                    df2['L_num'] = df2['Linker'].apply(get_number)
                    
                    if sort_option == "Linker order (L2 -> L10)":
                        # Sortujemy: R rosnąco, potem Linker rosnąco
                        df_plot = df2.sort_values(by=['R_num', 'L_num'], ascending=[True, True]).copy()
                    else:
                        # Sortujemy: R rosnąco, potem Kąt malejąco
                        df_plot = df2.sort_values(by=['S1', 'R_num'], ascending=[True, True]).copy()
                    
                    # Stała kolejność w legendzie (L2, L3...)
                    sorted_subs = sorted(df_plot['Substituent'].unique(), key=get_number)
                    
                    # 3. Tworzenie wykresu Plotly
                    fig3 = px.scatter(
                        df_plot,
                        x='ID',
                        y='S1',
                        color='Substituent',
                        category_orders={"Substituent": sorted_subs}, 
                        labels={
                            'ID': 'ID Związku',
                            'S1': 'S1 [eV]',
                            'Substituent': 'Substituent'
                        },
                        hover_data=['Substituent', 'Linker', 'S1']
                    )
                    
                    # 4. Stylizacja
                    fig3.update_traces(marker=dict(size=11, line=dict(width=1, color='white')))
                    fig3.update_layout(
                                xaxis=dict(
                                    showticklabels=False, # TO UKRYWA PODPISY (R1-L2-cośtam)
                                    showgrid=False,       # Opcjonalnie: ukrywa pionowe linie siatki
                                    title=None            # Ukrywa napis "ID Związku"
                                ),
                        yaxis=dict(
                            showgrid=True,           # Włącza linie siatki
                            gridcolor='white',      # Kolor linii siatki
                            gridwidth=0.5,           # Grubość linii
                            zerolinecolor='red',  # Linia zerowa też na pomarańczowo
                            dtick=0.5                # Co ile ma być linia (np. co 0.5 eV)
                        ),
                                template='plotly_dark' if bg_color in ["#111111", "#000000", "#2D2D2D"] else 'plotly_white',
                                plot_bgcolor=bg_color,
                                paper_bgcolor=bg_color,
                                margin=dict(l=60, r=10, t=30, b=10),
                                height=300 # Możesz teraz zmniejszyć wysokość, bo nie ma napisów na dole
                            )
                
                    
                    # 5. Wyświetlenie
                    st.plotly_chart(fig3, use_container_width=True,key="heatm")

        
                            

        with tab2:
            coll,colr=st.columns([2,2])
            with coll:
                def natural_key(string_):
                    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
                heatmap_dataT1 = df2.pivot_table(index="Linker", 
                                                    columns="Substituent", 
                                                    values="T1", 
                                                    aggfunc='mean')
                # 3. Sortowanie osi
                sorted_linkers = sorted(heatmap_dataT1.index, key=natural_key)
                sorted_substituents = sorted(heatmap_dataT1.columns, key=natural_key)
                heatmap_dataT1 = heatmap_dataT1.reindex(index=sorted_linkers, columns=sorted_substituents)
                
                fig1 = px.imshow(heatmap_dataT1,
                    labels=dict(x="Podstawnik", y="Linker", color="T1 [eV]"),
                    x=sorted_substituents,
                    y=sorted_linkers,
                    color_continuous_scale="jet", # Twoja ulubiona paleta
                    range_color=[float(heatmap_dataT1.min().min()), float(heatmap_dataT1.max().max())],             # Twoje skalowanie
                    text_auto=".2f",                  # Wyświetlanie wartości w kratkach
                    aspect="auto"                     # Automatyczne dopasowanie proporcji
                )
                
                # 5. Estetyka wykresu
                fig1.update_layout(
                    xaxis_nticks=len(sorted_substituents),
                    yaxis_nticks=len(sorted_linkers),
                    width=900, 
                    height=400)
                
                # 6. Wyświetlenie w Streamlit
                st.plotly_chart(fig1, use_container_width=True)
            with colr:
                     # 1. Funkcja pomocnicza
                def get_number(text):
                    if not isinstance(text, str): return 0
                    match = re.search(r'\d+', text)
                    return int(match.group()) if match else 0
            
            # --- INTERFEJS WYBORU ---
                st.markdown(f"""
        <div style="
            background-color: {kolor_tla}; 
            border-radius: 10px; 
            padding: 1px;
            margin-left: -10px;
            margin-right: 0px;
            margin-bottom: -590px; /* Trik, żeby 'podłożyć' tło pod wykres */
            height: 450px;
        ">
        </div>
    """, unsafe_allow_html=True)
                tabl, tabr= st.tabs(["Group by Linker", "Group by Substutuent"])
                with tabl:

                    sort_option = st.radio(
                        "Sorting by",
                        ["Energy T1 (descending)","Substituent order (R1 -> R16)"],
                        horizontal=True) 


                    # 2. Przygotowanie danych
                    # Zawsze potrzebujemy numeru R do głównego grupowania
                    df2['R_num'] = df2['Substituent'].apply(get_number)
                    df2['L_num'] = df2['Linker'].apply(get_number)
                    
                    if sort_option == "Substituent order (R1 -> R16)":
                        # Sortujemy: R rosnąco, potem Linker rosnąco
                        df_plot = df2.sort_values(by=['R_num', 'L_num'], ascending=[True, True]).copy()  
                    else:
                        # Sortujemy: R rosnąco, potem Kąt malejąco
                        df_plot = df2.sort_values(by=['T1', 'L_num'], ascending=[True, True]).copy()
                    
                    # Stała kolejność w legendzie (L2, L3...)
                    sorted_linkers = sorted(df_plot['Linker'].unique(), key=get_number)
                    
                    # 3. Tworzenie wykresu Plotly
                    fig2 = px.scatter(
                        df_plot,
                        x='ID',
                        y='T1',
                        color='Linker',
                        category_orders={"Linker": sorted_linkers}, 
                        labels={
                            'ID': 'ID Związku',
                            'T1': 'T1 [eV]',
                            'Linker': 'Linker'
                        },
                        hover_data=['Linker', 'Substituent', 'T1']
                    )
                    
                    # 4. Stylizacja
                    fig2.update_traces(marker=dict(size=11, line=dict(width=1, color='white')))
                    fig2.update_layout(
                                xaxis=dict(
                                    showticklabels=False, # TO UKRYWA PODPISY (R1-L2-cośtam)
                                    showgrid=False,       # Opcjonalnie: ukrywa pionowe linie siatki
                                    title=None            # Ukrywa napis "ID Związku"
                                ),
                        yaxis=dict(
                            showgrid=True,           # Włącza linie siatki
                            gridcolor='white',      # Kolor linii siatki
                            gridwidth=0.5,           # Grubość linii
                            zerolinecolor='red',  # Linia zerowa też na pomarańczowo
                            dtick=0.5                # Co ile ma być linia (np. co 0.5 eV)
                        ),
                                template='plotly_dark' if bg_color in ["#111111", "#000000", "#2D2D2D"] else 'plotly_white',
                                plot_bgcolor=bg_color,
                                paper_bgcolor=bg_color,
                                margin=dict(l=60, r=10, t=30, b=10),
                                height=300 # Możesz teraz zmniejszyć wysokość, bo nie ma napisów na dole
                            )
                
                    
                    # 5. Wyświetlenie
                    st.plotly_chart(fig2, use_container_width=True)
                with tabr:
                    sort_option = st.radio(
                        "Sorting by",
                        ["Energy T1 (descending)","Linker order (L2 -> L10)"],
                        horizontal=True) 
                    
                    # 2. Przygotowanie danych
                    # Zawsze potrzebujemy numeru R do głównego grupowania
                    df2['R_num'] = df2['Substituent'].apply(get_number)
                    df2['L_num'] = df2['Linker'].apply(get_number)
                    
                    if sort_option == "Linker order (L2 -> L10)":
                        # Sortujemy: R rosnąco, potem Linker rosnąco
                        df_plot = df2.sort_values(by=['R_num', 'L_num'], ascending=[True, True]).copy()
                    else:
                        # Sortujemy: R rosnąco, potem Kąt malejąco
                        df_plot = df2.sort_values(by=['T1', 'R_num'], ascending=[True, True]).copy()
                    
                    # Stała kolejność w legendzie (L2, L3...)
                    sorted_subs = sorted(df_plot['Substituent'].unique(), key=get_number)
                    
                    # 3. Tworzenie wykresu Plotly
                    fig3 = px.scatter(
                        df_plot,
                        x='ID',
                        y='T1',
                        color='Substituent',
                        category_orders={"Substituent": sorted_subs}, 
                        labels={
                            'ID': 'ID Związku',
                            'T1': 'T1 [eV]',
                            'Substituent': 'Substituent'
                        },
                        hover_data=['Substituent', 'Linker', 'T1']
                    )
                    
                    # 4. Stylizacja
                    fig3.update_traces(marker=dict(size=11, line=dict(width=1, color='white')))
                    fig3.update_layout(
                                xaxis=dict(
                                    showticklabels=False, # TO UKRYWA PODPISY (R1-L2-cośtam)
                                    showgrid=False,       # Opcjonalnie: ukrywa pionowe linie siatki
                                    title=None            # Ukrywa napis "ID Związku"
                                ),
                        yaxis=dict(
                            showgrid=True,           # Włącza linie siatki
                            gridcolor='white',      # Kolor linii siatki
                            gridwidth=0.5,           # Grubość linii
                            zerolinecolor='red',  # Linia zerowa też na pomarańczowo
                            dtick=0.5                # Co ile ma być linia (np. co 0.5 eV)
                        ),
                                template='plotly_dark' if bg_color in ["#111111", "#000000", "#2D2D2D"] else 'plotly_white',
                                plot_bgcolor=bg_color,
                                paper_bgcolor=bg_color,
                                margin=dict(l=60, r=10, t=30, b=10),
                                height=300 # Możesz teraz zmniejszyć wysokość, bo nie ma napisów na dole
                            )
                
                    
                    # 5. Wyświetlenie
                    st.plotly_chart(fig3, use_container_width=True,key="heatmT")
                




            
        with tab3:
            coll,colr=st.columns([2,2])
            with coll:
                st.plotly_chart(fig1, use_container_width=True, key="heatmap_S1_pierwszy")
            with colr:
                st.plotly_chart(fig3, use_container_width=True, key="heatmap_t1_pierwszy")
            heatmap_diff = heatmap_dataS1 - heatmap_dataT1

# Rysujemy mapę różnicy (Delta E_ST)
            fig_delta = px.imshow(
    heatmap_diff,
    labels=dict(x="Podstawnik", y="Linker", color="Delta E_ST [eV]"),
    color_continuous_scale="Viridis", # Inna skala, żeby odróżnić
    text_auto=".2f",
    title="Różnica Energii S1 - T1 (Delta E_ST)"
)
            st.plotly_chart(fig_delta, use_container_width=True)

#%%------------------------------------------------------------------------------------DIHEDRALS-------------------------------------------------------------------------------------------------------------------
with st.expander("Dihedrals", expanded=False):
    cola, colb, colc = st.columns([1,2,2])
    
    with cola:
        st.dataframe(df3, height=600, use_container_width=True)
        
    with colb:
        # 1. Funkcja do sortowania naturalnego
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
        
        # 2. Przygotowanie danych
        heatmap_data = df3.pivot_table(index="Linker", 
                                            columns="Substituent", 
                                            values="Torsion_DL2", 
                                            aggfunc='mean')
        
        # 3. Sortowanie osi
        sorted_linkers = sorted(heatmap_data.index, key=natural_key)
        sorted_substituents = sorted(heatmap_data.columns, key=natural_key)
        heatmap_data = heatmap_data.reindex(index=sorted_linkers, columns=sorted_substituents)
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Podstawnik", y="Linker", color="Kąt [°]"),
            x=sorted_substituents,
            y=sorted_linkers,
            color_continuous_scale="jet", # Twoja ulubiona paleta
            range_color=[60, 90],             # Twoje skalowanie
            text_auto=".3f",                  # Wyświetlanie wartości w kratkach
            aspect="auto"                     # Automatyczne dopasowanie proporcji
        )
        
        # 5. Estetyka wykresu
        fig.update_layout(
            title='Dihedrals Donor - Linker (D-L)',
            xaxis_nticks=len(sorted_substituents),
            yaxis_nticks=len(sorted_linkers),
            width=900, 
            height=600
        )
        
        # 6. Wyświetlenie w Streamlit
        st.plotly_chart(fig, use_container_width=True)
    with colc:
        tab1, tab2, tab3 = st.tabs(["Scatter: Group - Substituents 1", "Scatter: Group - Substituents 2", "Scatter: Group - Linker"])

        with tab1:
    
            df_plot = df3.sort_values('Torsion_DL2').copy()
            
            # 2. Tworzenie interaktywnego wykresu Plotly
            fig = px.scatter(
                df_plot,
                x='ID',              # Oś X: nazwa związku
                y='Torsion_DL2',     # Oś Y: kąt
                color='Substituent', # Kolor punktu zależny od podstawnika
                title='Scatter plot  of dihedrals D-L: increasing D-L in order of Linker',
                labels={
                    'ID': 'ID',
                    'Torsion_DL2': 'Dihedral D-L [°]',
                    'Substituent': 'Substituent'
                },
                hover_name='ID',     # Wytłuszczona nazwa w dymku po najechaniu
                # Dodatkowe dane widoczne w dymku:
                hover_data={'Linker': True, 'Substituent': True, 'Torsion_DL2': ':.3f'}
            )
            
            # 3. Personalizacja wykresu (estetyka i zakresy)
            fig.update_traces(
                marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')) # Większe punkty z obwódką
            )
            
            fig.update_layout(
            
                yaxis=dict(range=[-2, 95]),   # Sztywny zakres osi Y od 0 do 95 (z małym zapasem)
                xaxis=dict(
                            showticklabels=False, # Ukrywa podpisy (ID)
                            title=None,           # Ukrywa napis "Związek"
                            showgrid=False        # Opcjonalnie: usuwa pionowe linie siatki
                        ),          # Obrócenie etykiet na osi X
                legend_title_text='Substituent',
                template='plotly_white',       # Jasny, czysty styl (odpowiednik whitegrid)
                height=500                     # Wysokość wykresu
            )
            
            # 4. Wyświetlenie w Streamlit
            st.plotly_chart(fig, use_container_width=True)


        with tab2:
            # 1. Funkcja wyciągająca tylko numer (zwraca int)
            def get_r_number(text):
                if not isinstance(text, str): return 0
                match = re.search(r'\d+', text)
                return int(match.group()) if match else 0
            
            # 2. Przygotowanie danych do wykresu
            # Tworzymy tymczasową kolumnę z samym numerem (np. "R12" -> 12)
            df3['R_num_internal'] = df3['Substituent'].apply(get_r_number)
            
            # Sortujemy po numerze podstawnika, a potem po kącie
            df_plot = df3.sort_values(by=['R_num_internal', 'Torsion_DL2']).copy()
            
            # Usuwamy pomocniczą kolumnę, żeby nie śmieciła w dymkach (hover)
            df_plot = df_plot.drop(columns=['R_num_internal'])
            
            # 3. Wykres Plotly
            fig = px.scatter(
                df_plot,
                x='ID',
                y='Torsion_DL2',
                color='Substituent',
                title='Kąty pogrupowane według podstawników (rosnąco)',
                labels={'Torsion_DL2': 'Kąt [°]', 'ID': 'Związek'},
                template='plotly_dark'
            )
            
            # Wymuszenie kolejności kategorii na osi X
    
            fig.update_layout(
                        height=500,
                        xaxis=dict(
                            showticklabels=False, # Ukrywa podpisy (ID)
                            title=None,           # Ukrywa napis "Związek"
                            showgrid=False        # Opcjonalnie: usuwa pionowe linie siatki
                        )
                    )
            
            
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # 1. Funkcja pomocnicza
            def get_number(text):
                if not isinstance(text, str): return 0
                match = re.search(r'\d+', text)
                return int(match.group()) if match else 0
        
        # --- INTERFEJS WYBORU ---
        
            sort_option = st.radio(
                "Sorting by",
                ["Dihedral order (descending)","Linker order (L2 -> L10)"],
                horizontal=True
            ) 
            
            # 2. Przygotowanie danych
            # Zawsze potrzebujemy numeru R do głównego grupowania
            df3['R_num'] = df3['Substituent'].apply(get_number)
            df3['L_num'] = df3['Linker'].apply(get_number)
            
            if sort_option == "Linker order (L2 -> L10)":
                # Sortujemy: R rosnąco, potem Linker rosnąco
             
                df_plot = df3.sort_values(by=['R_num', 'L_num'], ascending=[True, True]).copy()
                current_title = 'Linker order (L2 -> L10)'
            else:
                # Sortujemy: R rosnąco, potem Kąt malejąco
                df_plot = df3.sort_values(by=['Torsion_DL2', 'L_num'], ascending=[True, True]).copy()
                current_title = 'Dihedral order (descending)'
            
            # Stała kolejność w legendzie (L2, L3...)
            sorted_linkers = sorted(df_plot['Linker'].unique(), key=get_number)
            
            # 3. Tworzenie wykresu Plotly
            fig = px.scatter(
                df_plot,
                x='ID',
                y='Torsion_DL2',
                color='Linker',
                category_orders={"Linker": sorted_linkers}, 
                title=current_title,
                labels={
                    'ID': 'ID Związku',
                    'Torsion_DL2': 'Dihedral D-L [°]',
                    'Linker': 'Linker'
                },
                hover_data=['Linker', 'Substituent', 'Torsion_DL2']
            )
            
            # 4. Stylizacja
            fig.update_traces(marker=dict(size=11, line=dict(width=1, color='white')))
            fig.update_layout(
                        yaxis=dict(range=[-2, 95], title='Dihedral D-L [°]'),
                        xaxis=dict(
                            showticklabels=False, # TO UKRYWA PODPISY (R1-L2-cośtam)
                            showgrid=False,       # Opcjonalnie: ukrywa pionowe linie siatki
                            title=None            # Ukrywa napis "ID Związku"
                        ),
                        template='plotly_dark',
                        height=500 # Możesz teraz zmniejszyć wysokość, bo nie ma napisów na dole
                    )
    
            
            # 5. Wyświetlenie
            st.plotly_chart(fig, use_container_width=True)
