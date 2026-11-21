import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import os

# ---------------- CONFIG ----------------
# Set the MIMIC-IV root to the directory containing hosp/ and icu/.
RAW_DATA_PATH = r'data/mimic-iv/mimic-iv-3.1'
OUTPUT_PATH = r'data/mimic-iv/mimic_iv.pkl'
# ----------------------------------------

def preprocess_mimic_iv():
    if os.path.exists("data/mimic-iv/chart_lab.pkl"):
        #print(f"Preprocessed file already exists at {OUTPUT_PATH}. Skipping preprocessing.")
        with open("data/mimic-iv/chart_lab.pkl", "rb") as f:
            events = pickle.load(f)
        path_hosp = os.path.join(RAW_DATA_PATH, 'hosp')
        path_icu = os.path.join(RAW_DATA_PATH, 'icu')
        icu = pd.read_csv(os.path.join(path_icu, 'icustays.csv.gz'),
                          usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])

        # Rename to match MIMIC-III script logic
        icu.rename(columns={
            'subject_id': 'SUBJECT_ID', 'hadm_id': 'HADM_ID', 'stay_id': 'ICUSTAY_ID',
            'intime': 'INTIME', 'outtime': 'OUTTIME'
        }, inplace=True)

        icu = icu.loc[icu.INTIME.notna()]
        icu = icu.loc[icu.OUTTIME.notna()]

        # 2. Filter Patients (Adults)
        # MIMIC-IV: patients.csv.gz (in hosp folder)
        # No DOB, use anchor_age + anchor_year
        pat = pd.read_csv(os.path.join(path_hosp, 'patients.csv.gz'),
                          usecols=['subject_id', 'anchor_age', 'anchor_year', 'gender'])
        pat.rename(columns={
            'subject_id': 'SUBJECT_ID', 'gender': 'GENDER',
            'anchor_age': 'ANCHOR_AGE', 'anchor_year': 'ANCHOR_YEAR'
        }, inplace=True)

        icu = icu.merge(pat, on='SUBJECT_ID', how='left')
        icu['INTIME'] = pd.to_datetime(icu.INTIME)

        # Calculate Age: anchor_age + (admission_year - anchor_year)
        icu['admit_year'] = icu['INTIME'].dt.year
        icu['AGE'] = icu['ANCHOR_AGE'] + (icu['admit_year'] - icu['ANCHOR_YEAR'])

        icu = icu.loc[icu.AGE >= 18]  # Filter adults
        print(f"Cohort size after filtering: {len(icu)} ICU stays")
    else:
        path_hosp = os.path.join(RAW_DATA_PATH, 'hosp')
        path_icu = os.path.join(RAW_DATA_PATH, 'icu')

        print("Step 1: Processing Cohort (Patients & Admissions)...")
        # 1. Get ICU Stays
        # MIMIC-IV: icustays.csv.gz (in icu folder)
        # Mapping: stay_id -> ICUSTAY_ID, hadm_id -> HADM_ID, subject_id -> SUBJECT_ID
        icu = pd.read_csv(os.path.join(path_icu, 'icustays.csv.gz'),
                          usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])

        # Rename to match MIMIC-III script logic
        icu.rename(columns={
            'subject_id': 'SUBJECT_ID', 'hadm_id': 'HADM_ID', 'stay_id': 'ICUSTAY_ID',
            'intime': 'INTIME', 'outtime': 'OUTTIME'
        }, inplace=True)

        icu = icu.loc[icu.INTIME.notna()]
        icu = icu.loc[icu.OUTTIME.notna()]

        # 2. Filter Patients (Adults)
        # MIMIC-IV: patients.csv.gz (in hosp folder)
        # No DOB, use anchor_age + anchor_year
        pat = pd.read_csv(os.path.join(path_hosp, 'patients.csv.gz'),
                          usecols=['subject_id', 'anchor_age', 'anchor_year', 'gender'])
        pat.rename(columns={
            'subject_id': 'SUBJECT_ID', 'gender': 'GENDER',
            'anchor_age': 'ANCHOR_AGE', 'anchor_year': 'ANCHOR_YEAR'
        }, inplace=True)

        icu = icu.merge(pat, on='SUBJECT_ID', how='left')
        icu['INTIME'] = pd.to_datetime(icu.INTIME)

        # Calculate Age: anchor_age + (admission_year - anchor_year)
        icu['admit_year'] = icu['INTIME'].dt.year
        icu['AGE'] = icu['ANCHOR_AGE'] + (icu['admit_year'] - icu['ANCHOR_YEAR'])

        icu = icu.loc[icu.AGE >= 18]  # Filter adults
        print(f"Cohort size after filtering: {len(icu)} ICU stays")

        # --------------------------------------------------------
        print("Step 2: Processing Chart Events...")
        print(f"Cohort size after filtering: {len(icu)} ICU stays")

        icu['OUTTIME'] = pd.to_datetime(icu['OUTTIME'])

        # --------------------------------------------------------
        print("Step 2: Processing Chart Events...")

        ch = []
        # Using chunksize because chartevents is huge
        chunk_size = 5000000

        for chunk in tqdm(pd.read_csv(os.path.join(path_icu, 'chartevents.csv.gz'),
                                      chunksize=chunk_size,
                                      usecols=['stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'])):
            # Rename columns immediately
            chunk.rename(columns={
                'stay_id': 'ICUSTAY_ID', 'itemid': 'ITEMID', 'charttime': 'CHARTTIME',
                'value': 'VALUE', 'valuenum': 'VALUENUM', 'valueuom': 'VALUEUOM'
            }, inplace=True)

            chunk = chunk.loc[chunk.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]

            chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'])

            chunk = chunk.merge(icu[['ICUSTAY_ID', 'INTIME', 'OUTTIME']], on='ICUSTAY_ID', how='left')

            # Keep events recorded within the ICU stay.
            chunk = chunk.loc[(chunk['CHARTTIME'] >= chunk['INTIME']) &
                              (chunk['CHARTTIME'] <= chunk['OUTTIME'])]

            chunk.drop(columns=['INTIME', 'OUTTIME'], inplace=True)

            chunk = chunk.loc[chunk.CHARTTIME.notna()]
            chunk = chunk.dropna(subset=['VALUE', 'VALUENUM'], how='all')
            ch.append(chunk)

        ch = pd.concat(ch)
        # ch = ch.loc[~(ch.VALUE.isna() & ch.VALUENUM.isna())]
        ch['TABLE'] = 'chart'
        print('Read chartevents done.')

        # --------------------------------------------------------
        print("Step 3: Processing Lab Events...")
        # MIMIC-IV: labevents.csv.gz (in hosp folder)
        la = pd.read_csv(os.path.join(path_hosp, 'labevents.csv.gz'),
                         usecols=['hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'])

        la.rename(columns={
            'hadm_id': 'HADM_ID', 'itemid': 'ITEMID', 'charttime': 'CHARTTIME',
            'value': 'VALUE', 'valuenum': 'VALUENUM', 'valueuom': 'VALUEUOM'
        }, inplace=True)

        la = la.loc[la.HADM_ID.isin(icu.HADM_ID)]
        la = la.loc[la.CHARTTIME.notna()]
        la = la.loc[~(la.VALUE.isna() & la.VALUENUM.isna())]
        la['ICUSTAY_ID'] = np.nan
        la['TABLE'] = 'lab'
        print('Read labevents done.')

        # --------------------------------------------------------
        print("Step 4: Extracting Features (Filtering IDs)...")

        events = pd.DataFrame()

        # --- Vital Signs (Chart Events) ---
        # UPDATED IDs: Only keeping MetaVision IDs (>= 220000)

        # BP
        dbp = [220051, 225310, 220180]  # Removed CareVue IDs
        sbp = [220050, 225309, 220179, 224167, 227243]
        mbp = [220052, 225312, 224322, 220181]

        ch_bp = ch.loc[ch.ITEMID.isin(dbp + sbp + mbp)]
        ch_bp = ch_bp.loc[(ch_bp.VALUENUM >= 0) & (ch_bp.VALUENUM <= 375)]
        ch_bp.loc[ch_bp.ITEMID.isin(dbp), 'NAME'] = 'DBP'
        ch_bp.loc[ch_bp.ITEMID.isin(sbp), 'NAME'] = 'SBP'
        ch_bp.loc[ch_bp.ITEMID.isin(mbp), 'NAME'] = 'MBP'
        ch_bp['VALUEUOM'] = 'mmHg'
        ch_bp['VALUE'] = None
        events = pd.concat([events, ch_bp])
        del ch_bp

        # GCS
        gcs_eye = [220739]
        gcs_motor = [223901]
        gcs_verbal = [223900]
        ch_gcs = ch.loc[ch.ITEMID.isin(gcs_eye + gcs_motor + gcs_verbal)]
        ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_eye), 'NAME'] = 'GCS_eye'
        ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_motor), 'NAME'] = 'GCS_motor'
        ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_verbal), 'NAME'] = 'GCS_verbal'
        ch_gcs['VALUEUOM'] = None
        ch_gcs['VALUE'] = None
        events = pd.concat([events, ch_gcs])
        del ch_gcs

        # Heart Rate
        hr = [220045]
        ch_hr = ch.loc[ch.ITEMID.isin(hr)]
        ch_hr = ch_hr.loc[(ch_hr.VALUENUM >= 0) & (ch_hr.VALUENUM <= 390)]
        ch_hr['NAME'] = 'HR'
        ch_hr['VALUEUOM'] = 'bpm'
        ch_hr['VALUE'] = None
        events = pd.concat([events, ch_hr])
        del ch_hr

        # Respiratory Rate
        rr = [220210, 224689, 224422, 224690, 227860, 227918]
        ch_rr = ch.loc[ch.ITEMID.isin(rr)]
        ch_rr = ch_rr.loc[(ch_rr.VALUENUM >= 0) & (ch_rr.VALUENUM <= 330)]
        ch_rr['NAME'] = 'RR'
        ch_rr['VALUEUOM'] = 'brpm'
        ch_rr['VALUE'] = None
        events = pd.concat([events, ch_rr])
        del ch_rr

        # Temperature
        temp_c = [223762]
        temp_f = [223761]
        ch_temp_c = ch.loc[ch.ITEMID.isin(temp_c)]
        ch_temp_f = ch.loc[ch.ITEMID.isin(temp_f)]
        ch_temp_f['VALUENUM'] = (ch_temp_f['VALUENUM'] - 32) * 5 / 9
        ch_temp = pd.concat([ch_temp_c, ch_temp_f])
        ch_temp = ch_temp.loc[(ch_temp.VALUENUM >= 14.2) & (ch_temp.VALUENUM <= 47)]
        ch_temp['NAME'] = 'Temperature'
        ch_temp['VALUEUOM'] = 'C'
        ch_temp['VALUE'] = None
        events = pd.concat([events, ch_temp])
        del ch_temp, ch_temp_c, ch_temp_f

        # Weight
        we_kg = [224639, 226512, 226846]
        we_lb = [226531]
        ch_we_kg = ch.loc[ch.ITEMID.isin(we_kg)]
        ch_we_lb = ch.loc[ch.ITEMID.isin(we_lb)]
        ch_we_lb['VALUENUM'] = ch_we_lb['VALUENUM'] * 0.453592
        ch_we = pd.concat([ch_we_kg, ch_we_lb])
        ch_we = ch_we.loc[(ch_we.VALUENUM >= 0) & (ch_we.VALUENUM <= 300)]
        ch_we['NAME'] = 'Weight'
        ch_we['VALUEUOM'] = 'kg'
        ch_we['VALUE'] = None
        events = pd.concat([events, ch_we])
        del ch_we, ch_we_kg, ch_we_lb

        # Height
        he_in = [226707]
        he_cm = [226730]
        ch_he_in = ch.loc[ch.ITEMID.isin(he_in)]
        ch_he_cm = ch.loc[ch.ITEMID.isin(he_cm)]
        ch_he_in['VALUENUM'] = ch_he_in['VALUENUM'] * 2.54
        ch_he = pd.concat([ch_he_in, ch_he_cm])
        ch_he = ch_he.loc[(ch_he.VALUENUM >= 0) & (ch_he.VALUENUM <= 275)]
        ch_he['NAME'] = 'Height'
        ch_he['VALUEUOM'] = 'cm'
        ch_he['VALUE'] = None
        events = pd.concat([events, ch_he])
        del ch_he, ch_he_in, ch_he_cm

        # FiO2
        fio2 = [223835]
        ch_fio2 = ch.loc[ch.ITEMID.isin(fio2)]
        idx = ch_fio2.VALUENUM > 1.0
        ch_fio2.loc[idx, 'VALUENUM'] = ch_fio2.loc[idx, 'VALUENUM'] / 100
        ch_fio2 = ch_fio2.loc[(ch_fio2.VALUENUM >= 0.2) & (ch_fio2.VALUENUM <= 1)]
        ch_fio2['NAME'] = 'FiO2'
        ch_fio2['VALUEUOM'] = None
        ch_fio2['VALUE'] = None
        events = pd.concat([events, ch_fio2])
        del ch_fio2

        # Capillary Refill
        cr = [224308, 223951]
        ch_cr = ch.loc[ch.ITEMID.isin(cr)]
        idx = (ch_cr.VALUE == 'Normal <3 Seconds') | (ch_cr.VALUE == 'Normal <3 secs')
        ch_cr.loc[idx, 'VALUENUM'] = 0
        idx = (ch_cr.VALUE == 'Abnormal >3 Seconds') | (ch_cr.VALUE == 'Abnormal >3 secs')
        ch_cr.loc[idx, 'VALUENUM'] = 1
        ch_cr['VALUEUOM'] = None
        ch_cr['NAME'] = 'CRR'
        events = pd.concat([events, ch_cr])
        del ch_cr

        # --- Lab & Chart Combined (Glucose, Bilirubin) ---
        # Glucose
        gl_bl = [225664, 50809, 52024]
        gl_wb = [226537]
        gl_se = [220621, 50931]

        def safe_concat_lab_chart_auto(mixed_ids, name):
            # MIMIC-IV chart item identifiers start at 220000.
            chart_ids = [x for x in mixed_ids if x >= 220000]
            lab_ids = [x for x in mixed_ids if x < 220000]

            df_c = ch.loc[ch.ITEMID.isin(chart_ids)] if chart_ids else pd.DataFrame()
            df_l = la.loc[la.ITEMID.isin(lab_ids)] if lab_ids else pd.DataFrame()

            res = pd.concat([df_c, df_l])
            res['NAME'] = name
            return res

        ev_blgl = safe_concat_lab_chart_auto(gl_bl, 'Glucose (Blood)')
        ev_blgl = ev_blgl.loc[(ev_blgl.VALUENUM >= 0) & (ev_blgl.VALUENUM <= 2200)]

        ev_wbgl = safe_concat_lab_chart_auto(gl_wb, 'Glucose (Whole Blood)')
        ev_wbgl = ev_wbgl.loc[(ev_wbgl.VALUENUM >= 0) & (ev_wbgl.VALUENUM <= 2200)]

        ev_segl = safe_concat_lab_chart_auto(gl_se, 'Glucose (Serum)')
        ev_segl = ev_segl.loc[(ev_segl.VALUENUM >= 0) & (ev_segl.VALUENUM <= 2200)]

        ev_gl = pd.concat([ev_blgl, ev_wbgl, ev_segl])
        ev_gl['VALUEUOM'] = 'mg/dL'
        ev_gl['VALUE'] = None
        events = pd.concat([events, ev_gl])
        del ev_gl, ev_blgl, ev_wbgl, ev_segl

        # Bilirubin (Keep original logic, mostly Lab items which are stable 5xxxx)
        br_to = [50885]
        br_di = [50883]
        br_in = [50884]
        ev_br = pd.concat([la.loc[la.ITEMID.isin(br_to + br_di + br_in)]])  # Chart IDs removed as they were CV
        ev_br = ev_br.loc[(ev_br.VALUENUM >= 0) & (ev_br.VALUENUM <= 66)]
        ev_br.loc[ev_br.ITEMID.isin(br_to), 'NAME'] = 'Bilirubin (Total)'
        ev_br.loc[ev_br.ITEMID.isin(br_di), 'NAME'] = 'Bilirubin (Direct)'
        ev_br.loc[ev_br.ITEMID.isin(br_in), 'NAME'] = 'Bilirubin (Indirect)'
        ev_br['VALUEUOM'] = 'mg/dL'
        ev_br['VALUE'] = None
        events = pd.concat([events, ev_br])
        del ev_br

        # Intubated (Lab item)
        itb = [50812]
        la_itb = la.loc[la.ITEMID.isin(itb)]
        idx = (la_itb.VALUE == 'INTUBATED.')
        la_itb.loc[idx, 'VALUENUM'] = 1
        idx = (la_itb.VALUE == 'NOT INTUBATED.')
        la_itb.loc[idx, 'VALUENUM'] = 0
        la_itb['VALUEUOM'] = None
        la_itb['NAME'] = 'Intubated'
        la_itb = la_itb.dropna(subset=['VALUENUM'])
        events = pd.concat([events, la_itb])
        del la_itb
        # --- Massive Lab Features Map ---
        # Note: Lab IDs (5xxxx) are generally consistent between III and IV.
        # Chart IDs in this list must be filtered for MetaVision (22xxxx).

        # Original lists filtered for IV:
        features_map = {
            'O2 Saturation': [[220227, 220277], [50817], [0, 100], '%'],  # Removed 834, etc.
            'Sodium': [[], [50983, 50824], [0, 250], 'mEq/L'],
            'Potassium': [[], [50971, 50822], [0, 15], 'mEq/L'],
            'Magnesium': [[], [50960], [0, 22], 'mg/dL'],
            'Phosphate': [[], [50970], [0, 22], 'mg/dL'],
            'Calcium Total': [[], [50893], [0, 40], 'mg/dL'],
            'Calcium Free': [[], [50808], [0, 10], 'mmol/L'],
            'WBC': [[], [51301, 51300], [0, 1100], 'K/uL'],
            'Hct': [[], [50810, 51221], [0, 100], '%'],
            'Hgb': [[], [51222, 50811], [0, 30], 'g/dL'],
            'Chloride': [[], [50902, 50806], [0, 200], 'mEq/L'],
            'Bicarbonate': [[], [50882, 50803], [0, 66], 'mEq/L'],
            'ALT': [[], [50861], [0, 11000], 'IU/L'],
            'ALP': [[], [50863], [0, 4000], 'IU/L'],
            'AST': [[], [50878], [0, 22000], 'IU/L'],
            'Albumin': [[], [50862], [0, 10], 'g/dL'],
            'Lactate': [[], [50813], [0, 33], 'mmol/L'],
            'LDH': [[], [50954], [0, 35000], 'IU/L'],
            'SG Urine': [[], [51498], [0, 2], ''],
            'pH Urine': [[220734], [51491, 51094], [0, 14], ''],  # Removed CV IDs
            'pH Blood': [[], [50820], [0, 14], ''],
            'PO2': [[], [50821], [0, 770], 'mmHg'],
            'PCO2': [[], [50818], [0, 220], 'mmHg'],
            'Total CO2': [[], [50804], [0, 65], 'mEq/L'],
            'Base Excess': [[], [50802], [-31, 28], 'mEq/L'],
            'Monocytes': [[], [51254], [0, 100], '%'],
            'Basophils': [[], [51146], [0, 100], '%'],
            'Eoisinophils': [[], [51200], [0, 100], '%'],
            'Neutrophils': [[], [51256], [0, 100], '%'],
            'Lymphocytes': [[], [51244, 51245], [0, 100], '%'],
            'Lymphocytes (Absolute)': [[], [51133], [0, 25000], '#/uL'],
            'PT': [[], [51274], [0, 150], 'sec'],
            'PTT': [[], [51275], [0, 150], 'sec'],
            'INR': [[], [51237], [0, 150], ''],
            'Anion Gap': [[], [50868], [0, 55], 'mg/dL'],
            'BUN': [[], [51006], [0, 275], 'mEq/L'],
            'Creatinine Blood': [[], [50912], [0, 66], 'mg/dL'],
            'Creatinine Urine': [[], [51082], [0, 650], 'mg/dL'],
            'MCH': [[], [51248], [0, 50], 'pg'],
            'MCHC': [[], [51249], [0, 50], '%'],
            'MCV': [[], [51250], [0, 150], 'fL'],
            'RDW': [[], [51277], [0, 37], '%'],
            'Platelet Count': [[], [51265], [0, 2200], 'K/uL'],
            'RBC': [[], [51279], [0, 14], 'm/uL']
        }

        for k, v in features_map.items():
            chart_ids, lab_ids, bounds, uom = v
            # Filter chart IDs to ensure only MV (>=220000) if any exist
            chart_ids = [x for x in chart_ids if x >= 220000]

            ev_k = pd.concat((ch.loc[ch.ITEMID.isin(chart_ids)], la.loc[la.ITEMID.isin(lab_ids)]))
            ev_k = ev_k.loc[(ev_k.VALUENUM >= bounds[0]) & (ev_k.VALUENUM <= bounds[1])]
            ev_k['NAME'] = k
            ev_k['VALUEUOM'] = uom
            ev_k['VALUE'] = None
            events = pd.concat([events, ev_k])

        # Free memory
        del ch, la
        print("Features extracted.")
        with open("data/mimic-iv/chart_lab.pkl", "wb") as f:
            pickle.dump(events, f)
    # --------------------------------------------------------
    if os.path.exists("data/mimic-iv/oe.pkl"):
        with open("data/mimic-iv/oe.pkl", "rb") as f:
            oe = pickle.load(f)

    else:
        print("Step 5: Processing Output Events...")
        # MIMIC-IV: outputevents.csv.gz (in icu folder)
        oe = pd.read_csv(os.path.join(path_icu, 'outputevents.csv.gz'),
                         usecols=['stay_id', 'itemid', 'charttime', 'value', 'valueuom'])
        oe.rename(columns={
            'stay_id': 'ICUSTAY_ID', 'itemid': 'ITEMID', 'charttime': 'CHARTTIME',
            'value': 'VALUE', 'valueuom': 'VALUEUOM'
        }, inplace=True)

        oe['VALUENUM'] = oe.VALUE
        oe.VALUE = None
        oe = oe.loc[oe.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
        oe = oe.merge(icu[['ICUSTAY_ID', 'INTIME', 'OUTTIME']], on='ICUSTAY_ID', how='left')
        oe['CHARTTIME'] = pd.to_datetime(oe['CHARTTIME'])
        oe = oe.loc[(oe.CHARTTIME >= oe.INTIME) & (oe.CHARTTIME <= oe.OUTTIME)]
        oe.drop(columns=['INTIME', 'OUTTIME'], inplace=True)
        oe['TABLE'] = 'output'
        with open("data/mimic-iv/oe.pkl", "wb") as f:
            pickle.dump(oe, f)
    # Output Items (MetaVision IDs)
    # Note: MIMIC-IV IDs for output are different from CareVue but similiar to MetaVision.
    # We will use the explicit list from the original script, filtering only MV IDs.

    # Pre-filtering dictionary for Outputs
    out_features = {
        'Ultrafiltrate': [[227510, 227511], [0, 7000], 'mL'],  # Removed CV
        'Urine': [[226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558], [0, 2500], 'mL'],
        # Added generic MV urine IDs as strict string matching is hard without D_ITEMS loaded
        'Stool': [[226579, 226579], [0, 4000], 'mL'],
        'Chest Tube': [[226593, 226590, 226591, 226595, 226592], [0, 2500], 'mL'],
        'Gastric': [[226576, 226575, 226573, 226630], [0, 4000], 'mL'],
        'EBL': [[226626, 226629], [0, 10000], 'mL'],
        'Emesis': [[226571], [0, 2000], 'mL'],
        'Jackson-Pratt': [[226600]],  # Placeholder for JP, MV doesn't use "Jackson" label directly often
        'Residual': [[227510, 227511], [0, 1050], 'mL'],
        'Pre-admission Output': [[226633], [0, 13000], 'ml']
    }
    # Note: For strict adherence to the original script which used string matching on D_ITEMS,
    # we would need to load d_items.csv.gz. For brevity in this fix, I am assuming
    # the IDs used in the "InputEvents" section of the original script (which were MV IDs) cover most cases.
    # *Important*: The original script relied on `d_items` string matching for Urine/Stool/Jackson.
    # Below is a simplified logic using known MV IDs.

    # Re-implementing specific Output Lists (filtered for MV only)
    ur = [226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584]  # Urine MV
    st = [226579, 226627]  # Stool MV
    jp = [226589, 226599]  # Jackson Pratt (Approx)

    # Overwrite dict with clean lists
    out_features['Urine'][0] = ur
    out_features['Stool'][0] = st
    out_features['Jackson-Pratt'] = [jp, [0, 2000], 'ml']

    for k, v in out_features.items():
        ev_k = oe.loc[oe.ITEMID.isin(v[0])]
        if len(ev_k) > 0:
            ind = (ev_k.VALUENUM >= v[1][0]) & (ev_k.VALUENUM <= v[1][1])
            med = ev_k.VALUENUM.loc[ind].median()
            ev_k.loc[~ind, 'VALUENUM'] = med
            ev_k['NAME'] = k
            ev_k['VALUEUOM'] = v[2]
            events = pd.concat([events, ev_k])
    del oe

    if os.path.exists("data/mimic-iv/ie.pkl"):
        print(f"Saved preprocessed events to data/mimic-iv/events.pkl")
        with open("data/mimic-iv/ie.pkl", "rb") as f:
            ie=pickle.load(f)
    else:
        # --------------------------------------------------------
        print("Step 6: Processing Input Events (Medications)...")
        # MIMIC-IV: inputevents.csv.gz (in icu folder)
        # This roughly corresponds to INPUTEVENTS_MV in MIMIC-III.
        # Structure: stay_id, itemid, starttime, endtime, amount, amountuom

        ie = pd.read_csv(os.path.join(path_icu, 'inputevents.csv.gz'),
                         usecols=['stay_id', 'itemid', 'starttime', 'endtime', 'amount', 'amountuom', 'patientweight'])

        ie.rename(columns={
            'stay_id': 'ICUSTAY_ID', 'itemid': 'ITEMID', 'starttime': 'STARTTIME',
            'endtime': 'ENDTIME', 'amount': 'AMOUNT', 'amountuom': 'AMOUNTUOM',
            'patientweight': 'PATIENTWEIGHT'
        }, inplace=True)

        ie = ie.loc[ie.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
        ie = ie.merge(icu[['ICUSTAY_ID', 'INTIME', 'OUTTIME']], on='ICUSTAY_ID', how='left')
        ie['STARTTIME'] = pd.to_datetime(ie['STARTTIME'])
        ie = ie.loc[(ie.STARTTIME >= ie.INTIME) & (ie.STARTTIME <= ie.OUTTIME)]
        ie.drop(columns=['INTIME', 'OUTTIME'], inplace=True)

        ie.STARTTIME = pd.to_datetime(ie.STARTTIME)
        ie.ENDTIME = pd.to_datetime(ie.ENDTIME)

        # Hourly Splitting Logic (Adapted from original script for MV)
        ie['TD'] = ie.ENDTIME - ie.STARTTIME
        new_ie = ie.loc[ie.TD <= pd.Timedelta(1, 'h')].drop(columns=['STARTTIME', 'TD'])
        ie_long = ie.loc[ie.TD > pd.Timedelta(1, 'h')]

        new_rows = []
        print("Splitting long infusions...")
        for _, row in tqdm(ie_long.iterrows(), total=len(ie_long)):
            icuid, iid, amo, uom, stm, td = row.ICUSTAY_ID, row.ITEMID, row.AMOUNT, row.AMOUNTUOM, row.STARTTIME, row.TD
            pw = row.PATIENTWEIGHT
            td_secs = td.total_seconds()
            num_hours = int(td_secs // 3600)
            hour_amount = (3600 * amo) / td_secs

            for i in range(1, num_hours + 1):
                new_rows.append([icuid, iid, stm + pd.Timedelta(i, 'h'), hour_amount, uom, pw])

            rem_mins = (td_secs % 3600) / 60
            if rem_mins > 0:
                new_rows.append([icuid, iid, row['ENDTIME'], (rem_mins * 60 * amo) / td_secs, uom, pw])

        if len(new_rows) > 0:
            new_rows_df = pd.DataFrame(new_rows,
                                       columns=['ICUSTAY_ID', 'ITEMID', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'PATIENTWEIGHT'])
            ie = pd.concat((new_ie, new_rows_df))
        else:
            ie = new_ie

        del new_ie, ie_long
        ie['TABLE'] = 'input_mv'
        ie.rename(columns={'ENDTIME': 'CHARTTIME', 'AMOUNT': 'VALUENUM', 'AMOUNTUOM': 'VALUEUOM'}, inplace=True)

        # Unit Conversions
        ind = (ie.VALUEUOM == 'mcg')
        ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM'] * 0.001
        ie.loc[ind, 'VALUEUOM'] = 'mg'
        ind = (ie.VALUEUOM == 'L')
        ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM'] * 1000
        ie.loc[ind, 'VALUEUOM'] = 'ml'
        with open("data/mimic-iv/ie.pkl", "wb") as f:
            pickle.dump(ie,f)
    # Input Features Map (Filtered for MV IDs only)
    input_features = {
        'Vasopressin': [[222315], [0, 400], 'units'],
        'Vacomycin': [[225798], [0, 8], 'g'],  # Unit conv handling needed? Original script handled 'mg' -> 'g'
        'Calcium Gluconate': [[221456, 227525], [0, 200], 'g'],
        'Furosemide': [[221794, 228340], [0, 250], 'mg'],
        'Famotidine': [[225907], [0, 1], 'dose'],
        'Piperacillin': [[225893, 225892], [0, 1], 'dose'],
        'Cefazolin': [[225850], [0, 2], 'dose'],
        'Fiber': [[225936, 227695, 225928, 226051, 226050, 226048, 227699, 227696, 226049, 226027, 227698], [0, 1600],
                  'ml'],
        'Pantoprazole': [[225910], [0, 1], 'dose'],
        'Magnesium Sulphate': [[222011, 227524], [0, 125], 'g'],
        'KCl': [[225166, 227536], [0, 501], 'mEq'],
        'Midazolam': [[221668], [0, 500], 'mg'],
        'Propofol': [[222168], [0, 12000], 'mg'],
        'Albumin 25%': [[220862], [0, 750], 'ml'],
        'Albumin 5%': [[220864], [0, 1300], 'ml'],
        'Fresh Frozen Plasma': [[220970], [0, 33000], 'ml'],
        'Lorazepam': [[221385], [0, 300], 'mg'],
        'Morphine Sulfate': [[], [0, 4000], 'mg'],
        # CV ID 225154 removed, check if MV exists (often 221744 Fentanyl used instead)
        'Gastric Meds': [[], [0, 7000], 'ml'],
        'Lactated Ringers': [[225828], [0, 17000], 'ml'],
        'Milrinone': [[221986], [0, 50], 'ml'],
        'OR/PACU Crystalloid': [[226364, 226375], [0, 22000], 'ml'],
        'Packed RBC': [[225168, 226368, 227070], [0, 17250], 'ml'],
        'PO intake': [[226452, 226377], [0, 11000], 'ml'],
        'Neosynephrine': [[221749], [0, 1200], 'mg'],
        'Piggyback': [[226089], [0, 1000], 'ml'],
        'Nitroglycerine': [[222056], [0, 350], 'mg'],
        'Nitroprusside': [[222051], [0, 430], 'mg'],
        'Metoprolol': [[225974], [0, 151], 'mg'],
        'Norepinephrine': [[221906], [0, 80], 'mg'],
        'Colloid': [[226365, 226376], [0, 20000], 'ml'],
        'Hydralazine': [[221828], [0, 80], 'mg'],
        'GT Flush': [[226453], [0, 2100], 'ml'],
        'Hydromorphone': [[221833], [0, 125], 'mg'],
        'Fentanyl': [[225942, 221744], [0, 20], 'mg'],
        'Insulin Regular': [[223258], [0, 1500], 'units'],
        'Insulin Humalog': [[223262], [0, 340], 'units'],
        'Insulin largine': [[223260], [0, 150], 'units'],
        'Insulin NPH': [[223259], [0, 100], 'units'],
        'D5W': [[220949], [0, 11000], 'ml'],
        'Dextrose Other': [[225823, 225825, 220950, 225827, 225941, 220952, 228142, 228140, 228141], [0, 4000], 'ml'],
        'Normal Saline': [[225158], [0, 11000], 'ml'],
        'Half Normal Saline': [[225159], [0, 2000], 'ml'],
        'Sterile Water': [[225944], [0, 10000], 'ml'],
        'Solution': [[225943], [0, 1500], 'ml'],
        'Dopamine': [[221662], [0, 1300], 'mg'],
        'Epinephrine': [[221289], [0, 100], 'mg'],
        'Amiodarone': [[221347, 228339], [0, 1200], 'mg'],
        'TPN': [[225916, 225917], [0, 1600], 'ml'],
        'Magnesium Sulfate (Bolus)': [[227523], [0, 250], 'ml'],
        'KCl (Bolus)': [[227522], [0, 500], 'ml'],
        'Heparin': [[225975, 225152], [0, 25300], 'units']
    }

    for k, v in input_features.items():
        if not v[0]: continue  # Skip if no IDs left
        ev_k = ie.loc[ie.ITEMID.isin(v[0])]
        if len(ev_k) == 0: continue

        # Specific unit fix for Vanc/Calcium/Mag as in original
        if k in ['Vacomycin', 'Calcium Gluconate', 'Magnesium Sulphate']:
            ind = ev_k.VALUEUOM.isin(['mg'])
            ev_k.loc[ind, 'VALUENUM'] = ev_k.loc[ind, 'VALUENUM'] * 0.001
            ev_k.loc[ind, 'VALUEUOM'] = 'g'

        ind = (ev_k.VALUENUM >= v[1][0]) & (ev_k.VALUENUM <= v[1][1])
        med = ev_k.VALUENUM.loc[ind].median()
        ev_k.loc[~ind, 'VALUENUM'] = med
        ev_k['NAME'] = k
        ev_k['VALUEUOM'] = v[2]
        events = pd.concat([events, ev_k])

    # Extract Weight from InputEvents (MIMIC-IV has patientweight in inputevents)
    # The original script extracted weight from MV inputevents
    ie_weight = ie[['ICUSTAY_ID', 'CHARTTIME', 'PATIENTWEIGHT']].copy()
    ie_weight.rename(columns={'PATIENTWEIGHT': 'VALUENUM'}, inplace=True)
    ie_weight['VALUEUOM'] = 'kg'
    ie_weight['NAME'] = 'Weight'
    ie_weight = ie_weight.dropna(subset=['VALUENUM'])
    ie_weight = ie_weight.loc[(ie_weight.VALUENUM >= 0) & (ie_weight.VALUENUM <= 300)]
    events = pd.concat([events, ie_weight])

    del ie
    # --------------------------------------------------------
    print("Mapping Lab Events to ICU Stays...")
    # Lab events only have HADM_ID. We need to assign ICUSTAY_ID based on time.

    if os.path.exists("data/mimic-iv/events_map.pkl"):
        print(f"Loading mapped events from data/mimic-iv/events_map.pkl")
        with open("data/mimic-iv/events_map.pkl", "rb") as f:
            events=pickle.load(f)
    else:
        icu.INTIME = pd.to_datetime(icu.INTIME)
        icu.OUTTIME = pd.to_datetime(icu.OUTTIME)
        events.CHARTTIME = pd.to_datetime(events.CHARTTIME)

        mask_lab = events['ICUSTAY_ID'].isna()

        if mask_lab.sum() > 0:
            # Match only the columns needed for the ICU stay assignment.
            lab_lookup = events.loc[mask_lab, ['HADM_ID', 'CHARTTIME']].copy()

            lab_lookup['orig_index'] = lab_lookup.index

            merged = lab_lookup.merge(
                icu[['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']],
                on='HADM_ID',
                how='left'
            )

            valid_mask = (merged['CHARTTIME'] >= merged['INTIME']) & (merged['CHARTTIME'] <= merged['OUTTIME'])
            matched = merged.loc[valid_mask]

            # Retain the first match if ICU stay windows overlap.
            matched = matched.drop_duplicates(subset=['orig_index'])

            events.loc[matched['orig_index'], 'ICUSTAY_ID'] = matched.set_index('orig_index')['ICUSTAY_ID']

            # Drop lab events that could not be assigned to an ICU stay.
            before_len = len(events)
            events.dropna(subset=['ICUSTAY_ID'], inplace=True)
            print(f"Dropped {before_len - len(events)} unmapped Lab events.")

            del lab_lookup, merged, matched
            import gc
            gc.collect()

        print(f"Events after mapping: {len(events)}")
        with open("data/mimic-iv/events_map.pkl", "wb") as f:
            pickle.dump(events, f)
    # --------------------------------------------------------
    print("Step 7: Final Merging & Splitting...")
    # Convert times
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)

    # Filter ICU
    icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

    # Relative Time
    events = events.merge(icu[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID', how='left')
    events['rel_charttime'] = events.CHARTTIME - events.INTIME
    events.drop(columns=['INTIME', 'CHARTTIME'], inplace=True)
    events.rel_charttime = events.rel_charttime.dt.total_seconds() // 60
    initial_len = len(events)
    events = events.loc[(events.rel_charttime > -1440) & (events.rel_charttime < 43200)]
    print(f"Final Sanity Check: Dropped {initial_len - len(events)} rows with extreme timestamps.")
    # Filter Short Stays (>24h)
    icu_full = icu.copy()
    icu['OUTTIME'] = pd.to_datetime(icu['OUTTIME'])
    icu = icu.loc[(icu.OUTTIME - icu.INTIME) >= pd.Timedelta(24, 'h')]

    # Mortality Label
    # MIMIC-IV admissions: hospital_expire_flag (0/1)
    adm = pd.read_csv(os.path.join(path_hosp, 'admissions.csv.gz'),
                      usecols=['hadm_id', 'hospital_expire_flag', 'deathtime'])
    adm.rename(
        columns={'hadm_id': 'HADM_ID', 'hospital_expire_flag': 'in_hospital_mortality', 'deathtime': 'DEATHTIME'},
        inplace=True)

    icu = icu.merge(adm, on='HADM_ID', how='left')
    icu.DEATHTIME = pd.to_datetime(icu.DEATHTIME)

    # Alive for at least 24h
    icu = icu.loc[((icu.DEATHTIME - icu.INTIME) >= pd.Timedelta(24, 'h')) | icu.DEATHTIME.isna()]

    # Event in first 24h check
    icu = icu.loc[icu.ICUSTAY_ID.isin(events.loc[events.rel_charttime < 24 * 60].ICUSTAY_ID)]

    # Final Renaming
    events.rename(columns={'rel_charttime': 'minute', 'NAME': 'variable',
                           'VALUENUM': 'value', 'ICUSTAY_ID': 'ts_id'}, inplace=True)

    # Static Features (Age, Gender)
    icu_full.rename(columns={'ICUSTAY_ID': 'ts_id'}, inplace=True)
    data_age = icu_full[['ts_id', 'AGE']].copy()
    data_age['variable'] = 'Age'
    data_age.rename(columns={'AGE': 'value'}, inplace=True)

    data_gen = icu_full[['ts_id', 'GENDER']].copy()
    data_gen['value'] = 0
    data_gen.loc[data_gen.GENDER == 'F', 'value'] = 1  # F=1, M=0
    data_gen['variable'] = 'Gender'
    data_gen.drop(columns=['GENDER'], inplace=True)

    data = pd.concat((data_age, data_gen), ignore_index=True)
    data['minute'] = 0
    events = pd.concat((data, events), ignore_index=True)

    # Filter columns and aggregate
    events = events[['ts_id', 'minute', 'variable', 'value', 'TABLE']]
    events['value'] = events['value'].astype(float)
    events.loc[events['TABLE'].isna(), 'TABLE'] = 'N/A'

    # --------------------------------------------------------
    print("Aggregating events (Memory Optimized)...")

    events['variable'] = events['variable'].astype('category')


    def unique_table_str(x):
        return ','.join(sorted(list(set(x))))

    all_ts_ids = events['ts_id'].unique()

    n_splits = 500
    chunks = np.array_split(all_ts_ids, n_splits)

    aggregated_results = []

    for chunk_ids in tqdm(chunks, desc="Grouping by chunks"):
        subset = events[events['ts_id'].isin(chunk_ids)]

        if len(subset) == 0: continue

        subset_agg = subset.groupby(['ts_id', 'minute', 'variable'], observed=True).agg({
            'value': 'mean',
            'TABLE': unique_table_str
        }).reset_index()

        aggregated_results.append(subset_agg)

        del subset, subset_agg

    print("Concatenating chunks...")
    events = pd.concat(aggregated_results, ignore_index=True)

    del aggregated_results
    import gc
    gc.collect()

    print(f"Aggregation done. Final shape: {events.shape}")

    # def f(x):
    #     if len(x) == 0:
    #         return x[0]
    #     else:
    #         return ','.join(x)
    #
    # events['TABLE'] = events['TABLE'].apply(f)

    # Outcomes DataFrame
    oc = icu_full[['ts_id', 'HADM_ID', 'SUBJECT_ID']].merge(adm, on='HADM_ID', how='left')
    # Retain patient identity so downstream code can audit patient-level isolation.
    oc = oc[['ts_id', 'SUBJECT_ID', 'in_hospital_mortality']]

    # Split
    all_subjects = icu.SUBJECT_ID.unique()
    np.random.seed(0)
    np.random.shuffle(all_subjects)
    S = len(all_subjects)
    bp1, bp2 = int(0.64 * S), int(0.8 * S)
    train_sub = all_subjects[:bp1]
    valid_sub = all_subjects[bp1:bp2]
    test_sub = all_subjects[bp2:]
    icu.rename(columns={'ICUSTAY_ID': 'ts_id'}, inplace=True)
    icu['ts_id'] = icu['ts_id'].astype(int)
    events['ts_id'] = events['ts_id'].astype(int)
    train_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(train_sub)].ts_id)
    valid_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(valid_sub)].ts_id)
    test_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(test_sub)].ts_id)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pickle.dump([events, oc, train_ids, valid_ids, test_ids], open(OUTPUT_PATH, 'wb'))
    print(f"Done! Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    preprocess_mimic_iv()
