# -*- coding: utf-8 -*-
import os
import re
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors, rdDetermineBonds, Draw, rdDepictor

def wykonaj_analize_L2(df, path_xyz):
    """
    Funkcja analizująca podstawniki dla serii L2 w trybie Dark Mode.
    Zwraca listę słowników z ID, wzorem, SMILES i obiektem obrazka PIL (czarne tło).
    """
    # --- 1. PRZYGOTOWANIE RDZENIA ---
    if not os.path.exists(path_xyz):
        return []

    try:
        raw_core = Chem.MolFromXYZFile(path_xyz)
        if raw_core is None: return []
            
        core_mol = Chem.Mol(raw_core)
        rdDetermineBonds.DetermineConnectivity(core_mol)
        core_no_h = Chem.RemoveHs(core_mol)
        core_smiles = Chem.MolToSmiles(core_no_h)
        core_final = Chem.MolFromSmiles(core_smiles)
    except:
        return []

    # --- 2. FILTROWANIE I SORTOWANIE ---
    df_l2 = df[df['ID'].astype(str).str.contains('L1', na=False)].copy()
    if df_l2.empty: return []

    df_l2['sort_num'] = df_l2['ID'].str.extract(r'R(\d+)').astype(float)
    df_l2 = df_l2.sort_values('sort_num').drop(columns=['sort_num'])

    final_results = []

    # --- 3. PĘTLA ANALIZY ---
    for idx, row in df_l2.iterrows():
        mol_raw = row['S0_MOL_Opt']
        mol_id = row['ID']
        if mol_raw is None: continue

        m = Chem.RemoveHs(mol_raw)
        c = Chem.RemoveHs(core_final)

        res = rdFMCS.FindMCS([m, c], completeRingsOnly=True, ringMatchesRingOnly=True)
        
        if res.smartsString:
            common_mol = Chem.MolFromSmarts(res.smartsString)
            diff = Chem.ReplaceCore(m, common_mol)
            
            wzor = "H"
            smi_clean = "H"
            img_single = None
            
            if diff:
                smi_raw = Chem.MolToSmiles(diff).replace('[*]', '').replace('*', '')
                smi_clean = re.sub(r'\[\d+\]', '', smi_raw)
                
                if smi_clean and smi_clean != "":
                    sub_mol = Chem.MolFromSmiles(smi_clean)
                    if sub_mol:
                        # Wzór
                        sub_mol_hs = Chem.AddHs(sub_mol)
                        wzor = rdMolDescriptors.CalcMolFormula(sub_mol_hs)
                        
                        # --- GENEROWANIE OBRAZKA DARK MODE (Wersja stabilna) ---
                        rdDepictor.Compute2DCoords(sub_mol)
                        
                        # Inicjalizacja rysownika Cairo
                        d2d = Draw.MolDraw2DCairo(300, 300)
                        dopts = d2d.drawOptions()
                        
                        # Ustawienia kolorystyczne
                        dopts.backgroundColour = (0.21, 0.21, 0.21, 1.0) # Czarne tło
                        
                        # Zamiast defaultColor, używamy bezpieczniejszych ustawień:
                        # RDKit automatycznie dobierze jasne wiązania, jeśli tło jest ciemne,
                        # ale możemy to wymusić ustawiając kolor symboli.
                        dopts.symbolColour = (1, 1, 1, 1)   # Białe symbole (N, O, S...)
                        dopts.bondLineWidth = 2             # Grubość linii
                        
                        # Wyłączenie kolorowania atomów wg układu okresowego (opcjonalnie)
                        # dzięki temu wszystkie atomy będą białe
                        dopts.updateAtomPalette({i: (1, 1, 1, 1) for i in range(100)})
                        
                        d2d.DrawMolecule(sub_mol)
                        d2d.FinishDrawing()
                        
                        # Konwersja na obrazek PIL
                        img_data = d2d.GetDrawingText()
                        img_single = Image.open(io.BytesIO(img_data))
            
            final_results.append({
                "ID": mol_id,
                "Wzor": wzor,
                "SMILES": smi_clean,
                "Obrazek": img_single
            })

    return final_results
