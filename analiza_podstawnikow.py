# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors, rdDetermineBonds, Draw, rdDepictor
import pandas as pd
import re

def wykonaj_analize_L2(df, path_xyz):
    # --- 1. PRZYGOTOWANIE RDZENIA ---
    if not os.path.exists(path_xyz):
        print(f"BŁĄD: Plik {path_xyz} nie istnieje.")
        return []

    raw_core = Chem.MolFromXYZFile(path_xyz)
    core_mol = Chem.Mol(raw_core)
    rdDetermineBonds.DetermineConnectivity(core_mol)
    core_no_h = Chem.RemoveHs(core_mol)
    core_smiles = Chem.MolToSmiles(core_no_h)
    core_final = Chem.MolFromSmiles(core_smiles)

    # --- 2. FILTROWANIE I SORTOWANIE (R2, R3... R16) ---
    df_l2 = df[df['ID'].str.contains('L2', na=False)].copy()
    
    # Wyciągamy numer po 'R', aby posortować matematycznie (2, 3, 10...)
    df_l2['sort_num'] = df_l2['ID'].str.extract(r'R(\d+)').astype(float)
    df_l2 = df_l2.sort_values('sort_num').drop(columns=['sort_num'])

    final_results = []

    # --- 3. PĘTLA ANALIZY ---
    for idx, row in df_l2.iterrows():
        mol_raw = row['S0_MOL_Opt']
        mol_id = row['ID']
        
        if mol_raw is None:
            continue

        # Przygotowanie do MCS
        m = Chem.RemoveHs(mol_raw)
        c = Chem.RemoveHs(core_final)

        # Znajdowanie części wspólnej
        res = rdFMCS.FindMCS([m, c], completeRingsOnly=True, ringMatchesRingOnly=True)
        
        if res.smartsString:
            common_mol = Chem.MolFromSmarts(res.smartsString)
            diff = Chem.ReplaceCore(m, common_mol)
            
            wzor = "H"
            smi_clean = "H"
            img_single = None
            
            if diff:
                # Czyszczenie SMILES
                smi_raw = Chem.MolToSmiles(diff).replace('[*]', '').replace('*', '')
                smi_clean = re.sub(r'\[\d+\]', '', smi_raw)
                
                if smi_clean and smi_clean != "":
                    sub_mol = Chem.MolFromSmiles(smi_clean)
                    if sub_mol:
                        # Wzór sumaryczny
                        sub_mol_hs = Chem.AddHs(sub_mol)
                        wzor = rdMolDescriptors.CalcMolFormula(sub_mol_hs)
                        
                        # Generowanie obrazka 2D
                        rdDepictor.Compute2DCoords(sub_mol)
                        img_single = Draw.MolToImage(sub_mol, size=(200, 200))
            
            # Dodajemy do listy wynikowej
            final_results.append({
                "ID": mol_id,
                "Wzor": wzor,
                "SMILES": smi_clean,
                "Obrazek": img_single
            })

    return final_results
