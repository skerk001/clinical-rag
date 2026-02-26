"""
Synthetic Clinical Note Generator for ClinicalRAG

Generates realistic clinical documents including:
- Discharge summaries
- Progress notes  
- Medication reconciliation notes
- Radiology reports
- Lab result interpretations

These synthetic notes are designed to be clinically realistic in structure
and terminology while containing no real patient data. They cover common
conditions seen in inpatient settings: CHF, COPD, diabetes, pneumonia,
acute kidney injury, sepsis, and stroke.

Author: Samir Kerkar
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ============================================================================
# CLINICAL TEMPLATES
# Each template category contains multiple variations to create a diverse
# corpus. The structure mirrors real EHR documentation patterns — this is
# important because RAG retrieval quality depends heavily on how well the
# synthetic data represents the chunking patterns of real clinical notes.
# ============================================================================

# Common patient demographics for generating realistic headers
FIRST_NAMES_M = ["James", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", "Daniel"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica", "Sarah", "Karen"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

# ICD-10 codes mapped to conditions — mirrors real EHR coding practices
CONDITION_ICD10 = {
    "CHF": {"code": "I50.9", "description": "Heart failure, unspecified"},
    "COPD_exacerbation": {"code": "J44.1", "description": "COPD with acute exacerbation"},
    "Type2_diabetes": {"code": "E11.65", "description": "Type 2 diabetes mellitus with hyperglycemia"},
    "Pneumonia": {"code": "J18.9", "description": "Pneumonia, unspecified organism"},
    "AKI": {"code": "N17.9", "description": "Acute kidney failure, unspecified"},
    "Sepsis": {"code": "A41.9", "description": "Sepsis, unspecified organism"},
    "Stroke": {"code": "I63.9", "description": "Cerebral infarction, unspecified"},
    "Atrial_fibrillation": {"code": "I48.91", "description": "Unspecified atrial fibrillation"},
    "GI_bleed": {"code": "K92.2", "description": "Gastrointestinal hemorrhage, unspecified"},
    "DVT_PE": {"code": "I26.99", "description": "Other pulmonary embolism without acute cor pulmonale"},
}

# ============================================================================
# DISCHARGE SUMMARY TEMPLATES
# These are the most information-dense clinical documents and the primary
# target for RAG retrieval. Each template follows the standard H&P format
# that real discharge summaries use.
# ============================================================================

DISCHARGE_SUMMARIES = [
    # --- CHF Templates ---
    {
        "condition": "CHF",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Acute decompensated heart failure (I50.31)
SECONDARY DIAGNOSES: Hypertension (I10), Type 2 diabetes mellitus (E11.9), Chronic kidney disease stage III (N18.3)

CHIEF COMPLAINT: Progressive dyspnea on exertion and bilateral lower extremity edema over 5 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with a history of heart failure with reduced ejection fraction (HFrEF, last EF 30%), hypertension, type 2 diabetes, and CKD stage III who presented to the ED with worsening shortness of breath and leg swelling. The patient reports a 10-pound weight gain over the past week and has been unable to sleep flat for 3 nights, requiring 3 pillows (increased from baseline of 1). The patient admits to dietary indiscretion with high sodium meals and missed two doses of furosemide last week. No chest pain, palpitations, or fever. BNP on admission was 1,847 pg/mL (baseline ~400). Chest X-ray showed bilateral pleural effusions and pulmonary vascular congestion.

HOSPITAL COURSE:
The patient was admitted to the telemetry unit and started on IV furosemide 40mg BID with strict I&O monitoring. Over 72 hours, the patient diuresed approximately 4.5 liters with improvement in symptoms. Oxygen requirements decreased from 3L nasal cannula to room air. Daily weights trended downward by 4.2 kg. Renal function remained stable with creatinine 1.6-1.8 mg/dL (baseline 1.5). Potassium was repleted twice during diuresis. An echocardiogram was performed showing EF 28% (stable from prior), moderate mitral regurgitation, and grade II diastolic dysfunction. Cardiology was consulted and recommended optimization of guideline-directed medical therapy.

MEDICATIONS AT DISCHARGE:
1. Furosemide 60mg PO daily (increased from 40mg)
2. Carvedilol 12.5mg PO BID (unchanged)
3. Lisinopril 20mg PO daily (unchanged)
4. Spironolactone 25mg PO daily (NEW — added per cardiology recommendation)
5. Empagliflozin 10mg PO daily (NEW — SGLT2 inhibitor for dual cardiorenal benefit)
6. Metformin 1000mg PO BID (unchanged)
7. Atorvastatin 40mg PO daily (unchanged)
8. KCl 20mEq PO daily (NEW — potassium supplementation during increased diuresis)

DISCHARGE CONDITION: Stable. Ambulating without supplemental oxygen. Weight at discharge: 92.3 kg.

DISCHARGE INSTRUCTIONS:
- Daily weight monitoring: call clinic if weight increases >2 lbs in one day or >5 lbs in one week
- Sodium restriction: <2000mg/day
- Fluid restriction: <2 liters/day
- Follow-up with cardiology in 1 week
- Follow-up with PCP in 2 weeks
- Labs (BMP) in 3 days to monitor potassium and renal function after spironolactone initiation
- Return to ED if worsening shortness of breath, chest pain, or weight gain >3 lbs overnight

PROGNOSIS: Guarded. Patient has progressive HFrEF with recurrent admissions (3rd in 12 months). Advanced heart failure therapies including ICD evaluation should be discussed at cardiology follow-up.
""",
    },
    {
        "condition": "CHF",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Heart failure with preserved ejection fraction, acute exacerbation (I50.31)
SECONDARY DIAGNOSES: Obesity (E66.01), Obstructive sleep apnea (G47.33), Atrial fibrillation (I48.91), Hypertension (I10)

CHIEF COMPLAINT: Worsening shortness of breath and orthopnea for 3 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with known HFpEF (last EF 58%), obesity (BMI 38.4), obstructive sleep apnea on CPAP, and paroxysmal atrial fibrillation who presented with progressive dyspnea. The patient was in their usual state of health until 3 days prior when they noted difficulty breathing with minimal exertion (walking to the bathroom). Two-pillow orthopnea developed, and the patient noted increased abdominal girth. The patient has been compliant with medications but reports poor CPAP adherence over the past month. BNP was 623 pg/mL. ECG showed atrial fibrillation with rapid ventricular response at 122 bpm.

HOSPITAL COURSE:
The patient was admitted for acute decompensated HFpEF with concomitant atrial fibrillation with RVR. IV diltiazem drip was initiated for rate control with successful reduction to 78 bpm, then transitioned to oral diltiazem ER 180mg daily. IV furosemide 20mg BID was administered with net negative fluid balance of 2.8 liters over 48 hours. The patient's symptoms improved significantly. Sleep medicine was consulted regarding CPAP non-adherence, and the patient was refitted with a new mask interface. Echocardiogram confirmed EF 55%, grade II diastolic dysfunction with elevated E/e' ratio of 14, and moderate left atrial enlargement (LA volume index 42 mL/m²). NT-proBNP trended down from 623 to 289 pg/mL at discharge.

MEDICATIONS AT DISCHARGE:
1. Furosemide 40mg PO daily (increased from 20mg)
2. Diltiazem ER 180mg PO daily (NEW — for rate control)
3. Apixaban 5mg PO BID (unchanged — for atrial fibrillation anticoagulation)
4. Losartan 50mg PO daily (unchanged)
5. Empagliflozin 10mg PO daily (NEW — SGLT2 inhibitor, EMPEROR-Preserved evidence)
6. Amlodipine 5mg PO daily (unchanged)

DISCHARGE INSTRUCTIONS:
- Resume CPAP nightly — compliance is critical for both OSA and heart failure management
- Sodium restriction <2000mg/day
- Daily weights with 2-lb rule
- Cardiology follow-up in 2 weeks
- PCP follow-up in 1 week for blood pressure check and medication review
- Return to ED for worsening dyspnea, palpitations, or weight gain
""",
    },

    # --- COPD Templates ---
    {
        "condition": "COPD_exacerbation",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Acute exacerbation of COPD (J44.1)
SECONDARY DIAGNOSES: Tobacco use disorder (F17.210), Hypertension (I10), Anxiety disorder (F41.9)

CHIEF COMPLAINT: Worsening cough, increased sputum production, and dyspnea for 4 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with COPD (GOLD Stage III, FEV1 38% predicted on last PFTs), current smoker (45 pack-year history), and anxiety who presented with an acute exacerbation. The patient reports increased cough with purulent yellow-green sputum, worsening dyspnea at rest, and wheezing over 4 days. The patient used their albuterol rescue inhaler approximately 12 times in the past 24 hours with minimal relief. No fever, hemoptysis, or chest pain. The patient had 2 exacerbations in the past 12 months, the most recent requiring hospitalization 4 months ago. The patient continues to smoke approximately half a pack per day despite counseling.

HOSPITAL COURSE:
On admission, the patient was tachypneic (RR 28) and hypoxic (SpO2 86% on room air). ABG showed pH 7.33, pCO2 52, pO2 58, HCO3 27 (acute on chronic respiratory acidosis). The patient was started on supplemental oxygen via nasal cannula at 2L (targeting SpO2 88-92%), nebulized albuterol/ipratropium q4h, IV methylprednisolone 125mg daily, and azithromycin 500mg for suspected bacterial exacerbation given purulent sputum. Chest X-ray showed hyperinflated lungs with flattened diaphragms, no consolidation or pneumothorax. Sputum culture grew normal respiratory flora. Over 3 days, the patient improved with decreasing oxygen requirements and improved air movement on exam. Transitioned to oral prednisone 40mg on hospital day 3. Respiratory therapy performed inhaler technique education — the patient was using their MDI incorrectly (not using spacer, poor coordination). Smoking cessation counseling was provided and the patient expressed willingness to attempt quitting. Nicotine replacement therapy was initiated.

MEDICATIONS AT DISCHARGE:
1. Prednisone 40mg PO daily x 2 more days (total 5-day course per REDUCE trial protocol)
2. Azithromycin 250mg PO daily x 2 more days (total 5-day course)
3. Tiotropium 18mcg inhaled daily (unchanged — long-acting anticholinergic)
4. Budesonide/formoterol 160/4.5mcg 2 puffs BID (SWITCHED from fluticasone/salmeterol — step-up to ICS/LABA)
5. Albuterol HFA 90mcg 2 puffs q4-6h PRN (rescue inhaler with spacer — SPACER IS CRITICAL)
6. Nicotine patch 21mg/day (NEW — smoking cessation)
7. Lisinopril 10mg PO daily (unchanged)
8. Sertraline 50mg PO daily (unchanged)

DISCHARGE INSTRUCTIONS:
- MOST IMPORTANT: Use spacer device with all MDIs — demonstrated technique before discharge
- Continue nicotine replacement; consider adding varenicline at PCP follow-up if patch alone insufficient
- Pulmonary rehabilitation referral placed — call to schedule within 1 week
- Influenza and pneumococcal vaccines are up to date
- Follow-up with pulmonology in 2 weeks
- PCP follow-up in 1 week
- Return to ED for worsening dyspnea, fever >101.5°F, hemoptysis, or confusion
- Action plan provided: GREEN zone (stable), YELLOW zone (increased symptoms — double rescue inhaler, start prednisone pack), RED zone (severe symptoms — call 911)

PROGNOSIS: Concerning. Patient has GOLD Stage III COPD with frequent exacerbations (≥2/year) and continues to smoke. Without smoking cessation and medication adherence, progressive decline is expected. FEV1 decline rate estimated at 60-80 mL/year given continued smoking versus 30 mL/year if cessation achieved.
""",
    },

    # --- Pneumonia Templates ---
    {
        "condition": "Pneumonia",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Community-acquired pneumonia (J18.1)
SECONDARY DIAGNOSES: Type 2 diabetes mellitus (E11.65), Hypertension (I10), Chronic kidney disease stage II (N18.2)

CHIEF COMPLAINT: Fever, productive cough, and pleuritic chest pain for 3 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with type 2 diabetes (last HbA1c 8.2%), hypertension, and CKD stage II who presented with 3 days of fever (Tmax 102.4°F at home), productive cough with rust-colored sputum, and right-sided pleuritic chest pain. The patient also reports rigors, malaise, and decreased oral intake. No recent travel, sick contacts, or hospitalization in the past 90 days. The patient is a non-smoker with no history of lung disease. CURB-65 score was 2 (confusion absent, BUN 28 mg/dL, RR 24, BP 108/68, age {age}).

HOSPITAL COURSE:
Admission labs showed WBC 16.8 with left shift (89% neutrophils), procalcitonin 2.4 ng/mL, creatinine 1.4 (baseline 1.1), and glucose 287 mg/dL. Chest X-ray demonstrated right lower lobe consolidation with small associated parapneumonic effusion. Blood cultures were drawn x2 (both eventually no growth). Sputum culture grew Streptococcus pneumoniae sensitive to penicillin and ceftriaxone. The patient was started on ceftriaxone 1g IV daily plus azithromycin 500mg IV daily per ATS/IDSA guidelines for CAP. Insulin sliding scale was initiated for hyperglycemia with basal-bolus correction given steroid-exacerbated diabetes. Over 72 hours, the patient defervesced, WBC normalized to 9.2, and oxygen requirements decreased from 4L NC to room air. Creatinine improved to 1.2 with IV fluid resuscitation. Repeat chest X-ray on day 3 showed improving consolidation. Transitioned to oral antibiotics on hospital day 3.

MEDICATIONS AT DISCHARGE:
1. Amoxicillin 1000mg PO TID x 4 more days (step-down from IV ceftriaxone, total 7-day course)
2. Azithromycin 250mg PO daily x 2 more days (total 5-day course)
3. Metformin 1000mg PO BID (HELD during admission for AKI — RESUME now that creatinine normalized)
4. Glipizide 10mg PO BID (unchanged)
5. Lisinopril 20mg PO daily (unchanged — held briefly during hypotension, resumed at discharge)
6. Amlodipine 5mg PO daily (unchanged)

DISCHARGE INSTRUCTIONS:
- Complete full antibiotic course as prescribed
- Follow-up chest X-ray in 6-8 weeks to confirm radiographic resolution (important for lung cancer screening given age)
- Pneumococcal vaccination status reviewed — PCV20 administered during hospitalization
- PCP follow-up in 1 week with repeat BMP to confirm renal recovery
- Diabetes management: HbA1c 8.2% — needs optimization. Discuss at PCP visit.
- Return to ED for recurrent fever, worsening dyspnea, chest pain, or hemoptysis
""",
    },

    # --- Sepsis Templates ---
    {
        "condition": "Sepsis",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Sepsis secondary to urinary tract infection (A41.9, N39.0)
SECONDARY DIAGNOSES: Acute kidney injury stage 2 (N17.9), Type 2 diabetes mellitus (E11.9), Benign prostatic hyperplasia (N40.0)

CHIEF COMPLAINT: Fever, confusion, and decreased urine output for 2 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with type 2 diabetes, BPH with chronic urinary retention (uses intermittent catheterization), and mild cognitive impairment who was brought to the ED by family for fever, altered mental status, and decreased urine output over 2 days. The family notes the patient became increasingly confused and lethargic, was not eating or drinking, and had foul-smelling urine. In the ED, temperature was 103.1°F, HR 112, BP 88/54, RR 22, SpO2 95% on RA. Lactate was 3.8 mmol/L. The patient met Sepsis-3 criteria (SOFA score increase ≥2 from baseline).

HOSPITAL COURSE:
Sepsis protocol was activated. The patient received 30 mL/kg crystalloid bolus (2.1L) within the first hour and empiric IV piperacillin-tazobactam 3.375g q6h after blood and urine cultures were obtained. A Foley catheter was placed with return of 800mL cloudy, malodorous urine. Urinalysis showed >100 WBC, positive nitrites, large leukocyte esterase. Urine culture grew Escherichia coli >100,000 CFU/mL, susceptible to ceftriaxone, TMP-SMX, and ciprofloxacin (resistant to ampicillin). Blood cultures grew E. coli in 2/2 bottles (concordant with urine source). Antibiotics were narrowed to IV ceftriaxone 2g daily on culture data. 

The patient's hemodynamics improved with fluid resuscitation — MAP >65 maintained without vasopressors after initial 4L resuscitation. Creatinine peaked at 3.2 mg/dL (baseline 1.3) consistent with KDIGO Stage 2 AKI, then trended down to 1.8 by discharge. Mental status cleared by hospital day 3. Blood cultures cleared on day 2. Endocrinology was consulted for glucose management during sepsis — insulin drip was used for the first 24 hours, then transitioned to basal-bolus regimen. Urology was consulted for BPH management and recommended tamsulosin with outpatient follow-up for possible TURP evaluation.

MEDICATIONS AT DISCHARGE:
1. Ciprofloxacin 500mg PO BID x 7 more days (step-down, total 14-day course for bacteremia from urinary source)
2. Tamsulosin 0.4mg PO daily (NEW — for BPH/urinary retention)
3. Metformin 500mg PO BID (REDUCED from 1000mg — restart at lower dose given recent AKI, titrate up at PCP visit)
4. Insulin glargine 18 units SC at bedtime (NEW — for tighter glucose control during recovery)
5. Lisinopril 10mg PO daily (HELD — do not restart until creatinine <1.5 and confirmed at PCP visit)

DISCHARGE INSTRUCTIONS:
- Complete full antibiotic course — this is critical as bacteremia was documented
- Monitor urine output: should be at minimum 1 liter per day; if significantly decreased, return to ED
- Resume intermittent catheterization schedule per urology; clean technique is essential to prevent recurrent UTI
- Labs (BMP, CBC) in 3 days at outpatient lab
- PCP follow-up in 5 days for renal function check and medication reconciliation
- Urology follow-up in 2 weeks for BPH evaluation
- Return to ED for fever >101°F, confusion, decreased urine output, or signs of recurrent infection
""",
    },

    # --- Type 2 Diabetes Templates ---
    {
        "condition": "Type2_diabetes",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Diabetic ketoacidosis in type 2 diabetes (E11.10)
SECONDARY DIAGNOSES: Type 2 diabetes mellitus with hyperglycemia (E11.65), Hypertension (I10), Hyperlipidemia (E78.5)

CHIEF COMPLAINT: Nausea, vomiting, abdominal pain, and polyuria for 2 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with type 2 diabetes (diagnosed 8 years ago, last HbA1c 10.4% three months ago), hypertension, and hyperlipidemia who presented with 2 days of progressive nausea, vomiting (6 episodes), diffuse abdominal pain, polyuria, and polydipsia. The patient admits to running out of metformin and glipizide 2 weeks ago and not refilling prescriptions due to cost. Blood glucose in the ED was 487 mg/dL, venous pH 7.21, bicarbonate 12, anion gap 22, beta-hydroxybutyrate 5.8 mmol/L, consistent with DKA. Serum osmolality 312 mOsm/kg. The patient denied alcohol use, and lipase was within normal limits.

HOSPITAL COURSE:
DKA protocol was initiated: IV insulin drip at 0.1 units/kg/hr, aggressive IV fluid resuscitation with 0.9% NaCl (1L/hr x 2 hours, then 250mL/hr), and electrolyte monitoring q2h. Potassium was 5.1 on admission (pseudohyperkalemia from acidosis) and trended to 3.4 during treatment — KCl 40mEq was added to fluids. Phosphate was repleted. The anion gap closed at 18 hours, and the patient was transitioned to subcutaneous insulin with a 2-hour overlap of IV drip. The precipitating factor was medication non-adherence due to cost. Social work was consulted and connected the patient with a patient assistance program for medications and a community health worker for ongoing support. Diabetes education was provided including sick day management, signs of DKA, and importance of medication adherence. The patient received hands-on insulin injection training with teach-back demonstration.

MEDICATIONS AT DISCHARGE:
1. Insulin glargine 22 units SC at bedtime (NEW — basal insulin for persistent hyperglycemia)
2. Insulin lispro sliding scale with meals (see attached scale — correctional doses)
3. Metformin 500mg PO BID (RESTARTED at lower dose — titrate to 1000mg BID at PCP visit if tolerated)
4. Empagliflozin 10mg PO daily (NEW — SGLT2 inhibitor for additional A1c reduction and cardiorenal benefit; HOLD if feeling sick/dehydrated — risk of euglycemic DKA)
5. Lisinopril 20mg PO daily (unchanged)
6. Atorvastatin 40mg PO daily (unchanged)

DISCHARGE INSTRUCTIONS:
- Insulin injection technique was demonstrated and patient performed teach-back successfully
- Check blood glucose at minimum 4 times daily: fasting, before lunch, before dinner, and at bedtime
- DKA warning signs: nausea/vomiting, abdominal pain, fruity breath, rapid breathing — GO TO ER IMMEDIATELY
- HOLD empagliflozin during any illness, surgery, or dehydration (euglycemic DKA risk)
- Patient assistance program application submitted for insulin and empagliflozin
- Diabetes educator follow-up in 1 week
- Endocrinology referral placed — appointment in 3 weeks
- PCP follow-up in 1 week for medication titration and HbA1c recheck
- Dietary counseling appointment scheduled with registered dietitian
""",
    },

    # --- Acute Kidney Injury ---
    {
        "condition": "AKI",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Acute kidney injury, KDIGO Stage 3 (N17.9)
SECONDARY DIAGNOSES: Chronic kidney disease stage IIIB (N18.32), Heart failure with reduced ejection fraction (I50.22), Type 2 diabetes mellitus (E11.9)

CHIEF COMPLAINT: Decreased urine output, fatigue, and bilateral lower extremity edema for 4 days.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with CKD stage IIIB (baseline creatinine 2.1-2.3), HFrEF (EF 35%), and type 2 diabetes who presented with oliguria, worsening edema, and fatigue. The patient's PCP had recently started ibuprofen 600mg TID for knee osteoarthritis pain 10 days prior to admission. The patient also reports decreased oral intake due to nausea over the past week. On admission, creatinine was 5.8 mg/dL (from baseline of 2.2 one month ago), BUN 78, potassium 5.9, bicarbonate 16. Urine output was approximately 200mL in the past 24 hours per patient estimate.

HOSPITAL COURSE:
The patient was admitted to the ICU for AKI with hyperkalemia. Emergent management of hyperkalemia included IV calcium gluconate, insulin/dextrose, and sodium polystyrene sulfonate. Continuous telemetry monitoring showed no peaked T waves or arrhythmias. Nephrology was urgently consulted. The etiology was determined to be multifactorial: NSAID-induced (afferent arteriolar vasoconstriction in the setting of ACE inhibitor and diuretic use — the classic "triple whammy"), compounded by volume depletion from poor oral intake and cardiorenal syndrome. All nephrotoxic medications were discontinued: ibuprofen (permanently), lisinopril (held), and furosemide (held initially). IV fluids were given cautiously given the patient's heart failure (250mL/hr x 4 hours, then reassessed). Renal ultrasound showed normal-sized kidneys without hydronephrosis, no obstruction. Urine studies showed FeNa 0.4% (pre-renal pattern) and bland sediment.

Over 5 days, creatinine improved from 5.8 to 3.1 mg/dL with conservative management. Urine output recovered to >1L/day by hospital day 3. Dialysis was not required. Furosemide was restarted at a reduced dose once renal function stabilized. ACE inhibitor was held at discharge pending further renal recovery. Pain management was transitioned to acetaminophen and low-dose tramadol.

MEDICATIONS AT DISCHARGE:
1. Furosemide 20mg PO daily (REDUCED from 40mg — will reassess at nephrology follow-up)
2. Carvedilol 6.25mg PO BID (REDUCED from 12.5mg during AKI — uptitrate at cardiology follow-up)
3. Acetaminophen 650mg PO q6h PRN pain (REPLACEMENT for ibuprofen)
4. Tramadol 50mg PO q8h PRN severe pain (short-term only)
5. Insulin glargine 16 units SC at bedtime (REDUCED from 22 units — less renal clearance of insulin during AKI)
6. Sodium bicarbonate 650mg PO TID (NEW — for metabolic acidosis management)
7. Lisinopril — HELD. Do not restart until cleared by nephrology.
8. Ibuprofen — PERMANENTLY DISCONTINUED. No NSAIDs of any kind.

DISCHARGE INSTRUCTIONS:
- CRITICAL: No NSAIDs (ibuprofen, naproxen, meloxicam, aspirin >81mg) — these caused your kidney injury
- Maintain adequate hydration: at least 1.5-2 liters of fluid daily unless otherwise instructed
- Labs (BMP) in 3 days
- Nephrology follow-up in 1 week — they will decide about restarting lisinopril
- Cardiology follow-up in 2 weeks for heart failure medication optimization
- PCP follow-up in 1 week
- Report any decrease in urine output, swelling, nausea, or confusion immediately
""",
    },

    # --- Stroke Templates ---
    {
        "condition": "Stroke",
        "template": """DISCHARGE SUMMARY

PATIENT: {patient_name}
MRN: {mrn}
DOB: {dob}
ADMISSION DATE: {admit_date}
DISCHARGE DATE: {discharge_date}
ATTENDING PHYSICIAN: Dr. {attending}
PRIMARY DIAGNOSIS: Acute ischemic stroke, left MCA territory (I63.411)
SECONDARY DIAGNOSES: Atrial fibrillation (I48.91), Hypertension (I10), Hyperlipidemia (E78.5), Type 2 diabetes mellitus (E11.9)

CHIEF COMPLAINT: Sudden onset right-sided weakness and speech difficulty.

HISTORY OF PRESENT ILLNESS:
{patient_name} is a {age}-year-old {sex} with atrial fibrillation (not on anticoagulation due to prior patient refusal), hypertension, hyperlipidemia, and type 2 diabetes who was brought by EMS with witnessed sudden onset right-sided weakness and slurred speech. Last known well time was 90 minutes prior to ED arrival. NIHSS score on arrival was 12 (right facial droop, right arm/leg weakness 3/5, dysarthria, partial gaze preference). CT head was negative for hemorrhage. CT angiography showed occlusion of the left M1 segment of the MCA.

HOSPITAL COURSE:
The patient was within the 4.5-hour window and received IV alteplase (0.9 mg/kg, 10% bolus, remainder over 60 minutes). Interventional neuroradiology was consulted for mechanical thrombectomy given large vessel occlusion. Successful recanalization was achieved (TICI 2b) via aspiration thrombectomy. Post-procedure, the patient was admitted to the neuro-ICU for 24-hour monitoring with q1h neurological checks. NIHSS improved to 6 at 24 hours (persistent right arm weakness 4/5, mild facial droop, mild dysarthria). MRI brain confirmed left MCA infarct involving the left insular cortex and portions of the frontal operculum with no hemorrhagic transformation. Telemetry confirmed atrial fibrillation with rates 70-90s on rate control.

Swallow evaluation by speech therapy: passed modified barium swallow for regular diet with thin liquids. Physical therapy and occupational therapy evaluations revealed moderate right upper extremity weakness limiting fine motor tasks, mild right lower extremity weakness with antalgic gait. PT recommended acute inpatient rehabilitation. The patient and family agreed to anticoagulation initiation after thorough discussion of stroke recurrence risk without anticoagulation (approximately 5% per year per CHA2DS2-VASc score of 5) versus bleeding risk. Apixaban was chosen given superior safety profile in AF-related stroke prevention.

MEDICATIONS AT DISCHARGE:
1. Apixaban 5mg PO BID (NEW — for atrial fibrillation and stroke prevention)
2. Atorvastatin 80mg PO daily (INCREASED from 40mg — high-intensity statin post-stroke)
3. Metoprolol succinate 50mg PO daily (unchanged — AF rate control)
4. Amlodipine 10mg PO daily (unchanged — blood pressure goal <130/80 post-stroke)
5. Metformin 1000mg PO BID (unchanged)
6. Aspirin — DISCONTINUED given initiation of apixaban for AF

DISCHARGE DISPOSITION: Acute inpatient rehabilitation facility.

DISCHARGE INSTRUCTIONS:
- Blood thinner (apixaban) is critical — do NOT miss doses. Missing even one dose significantly increases stroke recurrence risk.
- Signs of recurrent stroke (BE FAST): Balance loss, Eyes (vision changes), Face drooping, Arm weakness, Speech difficulty, Time to call 911.
- Fall precautions: use assistive device as recommended by physical therapy
- Neurology follow-up in 4 weeks with repeat MRI
- Cardiology follow-up in 2 weeks for AF management
- PCP follow-up within 1 week of rehab discharge
- Continue all rehabilitation exercises as prescribed
""",
    },
]

# ============================================================================
# PROGRESS NOTE TEMPLATES
# Shorter daily notes that provide follow-up context. Important for RAG
# because they contain treatment response data and clinical decision-making.
# ============================================================================

PROGRESS_NOTES = [
    {
        "condition": "CHF",
        "template": """PROGRESS NOTE — HOSPITAL DAY {day_number}

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
PROVIDER: Dr. {provider}

SUBJECTIVE:
Patient reports improved breathing overnight. Able to lay flat with 1 pillow (down from 3 on admission). Denies chest pain or palpitations. Appetite improving. Patient asking about when they can go home.

OBJECTIVE:
Vitals: T 98.2°F, HR 72, BP 118/74, RR 16, SpO2 97% on room air
Weight: {weight} kg (down {weight_change} kg from admission)
I&O: Intake 1200mL / Output 2800mL (net negative 1600mL)
Exam: JVP ~8cm (improved from 14cm on admission). Lungs with bibasilar crackles, improved. Trace bilateral pedal edema (was 2+ on admission). Regular rate and rhythm, S3 no longer appreciated.
Labs: BNP 892 (down from 1847 on admission). Na 138, K 3.8, Cr 1.7 (stable), BUN 32.

ASSESSMENT AND PLAN:
Acute decompensated heart failure — improving with IV diuresis.
1. Continue IV furosemide 40mg BID, reassess for PO conversion tomorrow
2. Daily weight and strict I&Os
3. Will add spironolactone 25mg daily per cardiology recommendation once K trend confirmed stable
4. Echocardiogram scheduled for today
5. Continue telemetry monitoring
6. Pharmacy consulted for discharge medication reconciliation
7. Anticipate discharge in 24-48 hours if diuresis goals met
""",
    },
    {
        "condition": "COPD_exacerbation",
        "template": """PROGRESS NOTE — HOSPITAL DAY {day_number}

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
PROVIDER: Dr. {provider}

SUBJECTIVE:
Patient reports breathing is "much better." Cough is less frequent and sputum is now clear/white (was purulent yellow-green on admission). Used albuterol rescue inhaler twice today (down from 12 times on admission day). Sleeping better — able to rest for 4-hour stretches. Patient expressed interest in smoking cessation program. Anxiety is manageable.

OBJECTIVE:
Vitals: T 98.4°F, HR 82, BP 132/78, RR 18, SpO2 92% on 1L NC (was 86% on RA at admission)
Exam: Decreased breath sounds bilaterally with mild scattered expiratory wheezes (improved from diffuse wheezes with accessory muscle use on admission). No accessory muscle use at rest. Speaking in full sentences.
ABG: pH 7.38, pCO2 46, pO2 68 on 1L NC (improved from pH 7.33, pCO2 52 on admission)
Peak flow: 180 L/min (baseline reported as 220 L/min; admission was unable to perform)

ASSESSMENT AND PLAN:
Acute COPD exacerbation — significantly improved.
1. Wean O2: trial off nasal cannula with SpO2 monitoring. Goal SpO2 88-92%.
2. Transition IV steroids to oral prednisone 40mg PO daily (day 3 of 5-day course)
3. Continue azithromycin day 3 of 5
4. Smoking cessation: nicotine patch 21mg started today. Provided quitline number.
5. Respiratory therapy to reassess inhaler technique before discharge
6. Pulmonary rehab referral placed
7. Likely discharge tomorrow if tolerates oxygen wean
""",
    },
    {
        "condition": "Sepsis",
        "template": """PROGRESS NOTE — HOSPITAL DAY {day_number}

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
PROVIDER: Dr. {provider}

SUBJECTIVE:
Patient is more alert and oriented today. Recognizes family members. Reports mild suprapubic discomfort but denies dysuria or fever. Appetite returning — tolerated breakfast. Family reports patient is "more like themselves."

OBJECTIVE:
Vitals: T 99.1°F (Tmax overnight 99.8°F, trending down from 103.1°F on admission), HR 88, BP 118/72 (off pressors since HD1), RR 16, SpO2 98% on RA
Exam: Alert and oriented x3 (was AO x1 on admission). Abdomen soft, mild suprapubic tenderness. Foley draining clear yellow urine (was cloudy on admission).
Labs: WBC 10.2 (down from 18.4), Cr 2.4 (down from peak 3.2, baseline 1.3), K 4.2, lactate 1.1 (normalized from 3.8 on admission)
Urine output: 1.8L in past 24 hours (recovered from oliguria)
Micro: Blood cultures no growth x48 hours (previously positive E. coli). Repeat blood cultures drawn yesterday — pending.

ASSESSMENT AND PLAN:
Sepsis secondary to E. coli UTI with bacteremia — improving significantly.
1. Continue IV ceftriaxone (narrowed from pip-tazo based on sensitivities) — plan for 14-day total course given bacteremia. Will transition to PO ciprofloxacin at discharge.
2. Renal function recovering — trend BMP daily. Nephrotoxic med avoidance.
3. Foley catheter: plan to remove tomorrow with post-void residual check. Urology following.
4. Glucose management: transition from insulin drip to SC basal-bolus today.
5. Likely step down from ICU today, discharge in 48-72 hours if clinically stable.
""",
    },
]

# ============================================================================
# MEDICATION RECONCILIATION TEMPLATES
# These are critical for RAG because medication questions are among the most
# common clinical queries. They contain drug names, doses, reasons, and
# important safety notes.
# ============================================================================

MEDICATION_RECONCILIATION_NOTES = [
    {
        "condition": "CHF",
        "template": """MEDICATION RECONCILIATION NOTE

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
PHARMACIST: {pharmacist}

INDICATION: Discharge medication reconciliation for acute decompensated heart failure

MEDICATIONS REVIEWED AND RECONCILED:

CONTINUED (no changes):
- Carvedilol 12.5mg PO BID — beta-blocker for HFrEF. Target HR 60-70. Do not discontinue abruptly.
- Lisinopril 20mg PO daily — ACE inhibitor for HFrEF and hypertension. Monitor K+ and Cr.
- Atorvastatin 40mg PO daily — statin for cardiovascular risk reduction.
- Metformin 1000mg PO BID — for type 2 diabetes. Hold if Cr >1.5 or during acute illness.

CHANGED:
- Furosemide: INCREASED from 40mg PO daily → 60mg PO daily. Reason: Suboptimal volume status on prior dose; required IV diuresis. Monitor daily weights, BMP in 3 days.

NEW MEDICATIONS:
- Spironolactone 25mg PO daily — mineralocorticoid receptor antagonist for HFrEF (mortality benefit per RALES trial). CRITICAL: Monitor potassium in 3-5 days — risk of hyperkalemia, especially with ACE inhibitor. Contraindicated if K+ >5.0.
- Empagliflozin 10mg PO daily — SGLT2 inhibitor for HFrEF (EMPEROR-Reduced evidence). Dual cardiorenal benefit. Hold during acute illness/dehydration. May cause genital mycotic infections — counsel patient. May cause euglycemic DKA in patients with diabetes — counsel on sick-day rules.
- Potassium chloride 20mEq PO daily — supplementation during increased loop diuretic dosing. Discontinue if K+ >5.0.

DISCONTINUED:
- None

DRUG INTERACTIONS IDENTIFIED:
- Spironolactone + Lisinopril + KCl: HIGH RISK for hyperkalemia. Close monitoring essential. BMP in 3 days, then weekly x4 weeks.
- Empagliflozin + Furosemide: Additive diuretic effect — monitor for hypotension and volume depletion.

PATIENT COUNSELING PROVIDED:
- Daily weight monitoring technique demonstrated
- Low-sodium diet education (<2000mg/day) with examples of high-sodium foods to avoid
- Medication adherence importance — missed diuretic doses directly precipitated this admission
- Signs/symptoms requiring immediate medical attention reviewed
""",
    },
]

# ============================================================================
# RADIOLOGY REPORT TEMPLATES
# ============================================================================

RADIOLOGY_REPORTS = [
    {
        "condition": "CHF",
        "template": """RADIOLOGY REPORT

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
EXAM: Chest X-ray (PA and Lateral)
ORDERING PHYSICIAN: Dr. {ordering_md}
RADIOLOGIST: Dr. {radiologist}

CLINICAL INDICATION: Acute decompensated heart failure. Assess for pulmonary edema and pleural effusions.

COMPARISON: Chest X-ray from {comparison_date}.

FINDINGS:
Heart: Cardiomegaly (cardiothoracic ratio 0.62, increased from 0.58 on prior). Stable left ventricular configuration.
Mediastinum: Widened mediastinum consistent with vascular congestion. No discrete mass.
Lungs: Bilateral perihilar haziness with Kerley B lines consistent with pulmonary edema. Bilateral pleural effusions, small to moderate (right greater than left), increased from prior. No focal consolidation or pneumothorax.
Bones: Degenerative changes of the thoracic spine. No acute fracture.

IMPRESSION:
1. Pulmonary edema with bilateral pleural effusions, consistent with decompensated heart failure. Worsened compared to prior study.
2. Cardiomegaly, mildly progressive.
""",
    },
    {
        "condition": "Pneumonia",
        "template": """RADIOLOGY REPORT

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
EXAM: Chest X-ray (PA and Lateral)
ORDERING PHYSICIAN: Dr. {ordering_md}
RADIOLOGIST: Dr. {radiologist}

CLINICAL INDICATION: Fever, cough, pleuritic chest pain. Rule out pneumonia.

COMPARISON: No prior imaging available.

FINDINGS:
Heart: Normal heart size. No pericardial effusion.
Mediastinum: Normal. No lymphadenopathy.
Lungs: Right lower lobe consolidation with air bronchograms, approximately 6 cm in greatest dimension. Small right-sided parapneumonic effusion. No cavitation. Left lung is clear. No pneumothorax.
Bones: Unremarkable.

IMPRESSION:
1. Right lower lobe pneumonia with small parapneumonic effusion. Clinical correlation for community-acquired versus atypical pathogens recommended.
2. Follow-up imaging in 6-8 weeks recommended to document resolution, particularly in patients with risk factors for malignancy.
""",
    },
]

# ============================================================================
# LAB RESULT INTERPRETATION TEMPLATES
# ============================================================================

LAB_INTERPRETATIONS = [
    {
        "condition": "AKI",
        "template": """LABORATORY RESULTS WITH CLINICAL INTERPRETATION

PATIENT: {patient_name} | MRN: {mrn}
DATE: {note_date}
ORDERING PHYSICIAN: Dr. {provider}

COMPREHENSIVE METABOLIC PANEL:
- Sodium: 134 mEq/L (ref: 136-145) — mild hyponatremia, likely dilutional
- Potassium: 5.9 mEq/L (ref: 3.5-5.0) *** CRITICAL *** — hyperkalemia secondary to AKI and decreased renal excretion. REQUIRES URGENT TREATMENT.
- Chloride: 102 mEq/L (ref: 98-106) — normal
- CO2/Bicarbonate: 16 mEq/L (ref: 22-29) *** LOW *** — non-anion gap metabolic acidosis from renal failure
- BUN: 78 mg/dL (ref: 7-20) *** HIGH *** — elevated, consistent with acute kidney injury
- Creatinine: 5.8 mg/dL (ref: 0.7-1.3) *** CRITICAL *** — KDIGO Stage 3 AKI (>3x baseline of 2.1)
- Glucose: 198 mg/dL (ref: 70-100) — hyperglycemia in known diabetic
- Calcium: 8.1 mg/dL (ref: 8.5-10.5) — mildly low, expected in AKI
- Phosphorus: 6.2 mg/dL (ref: 2.5-4.5) *** HIGH *** — hyperphosphatemia from decreased renal clearance
- Magnesium: 2.4 mg/dL (ref: 1.7-2.2) — mildly elevated
- Albumin: 3.2 g/dL (ref: 3.5-5.5) — mild hypoalbuminemia

CALCULATED VALUES:
- Anion gap: 16 (mildly elevated)
- eGFR: 8 mL/min/1.73m² (baseline ~28) — severe decline
- BUN/Cr ratio: 13.4 (pre-renal pattern suggested by FeNa 0.4%)

URINALYSIS:
- Specific gravity: 1.025 (concentrated, pre-renal pattern)
- Protein: 1+ (mild proteinuria)
- Blood: negative
- WBC: 0-2/hpf
- RBC: 0-1/hpf
- Casts: rare hyaline casts (non-specific, seen in pre-renal azotemia)
- FeNa: 0.4% (strongly suggestive of pre-renal etiology)

CLINICAL INTERPRETATION:
This laboratory pattern is consistent with acute kidney injury with a predominantly pre-renal component. The FeNa of 0.4% and concentrated urine strongly suggest inadequate renal perfusion. In the context of this patient's history (recent NSAID initiation while on ACE inhibitor and diuretic — the "triple whammy" of renal hemodynamic insult), the mechanism is afferent arteriolar vasoconstriction (NSAIDs blocking prostaglandin-mediated vasodilation) combined with efferent arteriolar vasodilation (ACE inhibitor), resulting in critically reduced glomerular filtration pressure. Volume depletion from poor oral intake further exacerbates the insult.

URGENT ACTIONS REQUIRED:
1. Treat hyperkalemia emergently (calcium gluconate, insulin/dextrose, kayexalate)
2. Discontinue ALL nephrotoxic medications (ibuprofen permanently, hold ACE inhibitor, hold diuretic)
3. Cautious volume resuscitation (balance with heart failure)
4. Continuous telemetry for hyperkalemia-related arrhythmia risk
5. Nephrology consultation
""",
    },
]


def generate_patient() -> dict:
    """Generate a random patient with realistic demographics."""
    sex = random.choice(["male", "female"])
    first_name = random.choice(FIRST_NAMES_M if sex == "male" else FIRST_NAMES_F)
    last_name = random.choice(LAST_NAMES)
    age = random.randint(45, 88)

    # Generate realistic dates
    admit_date = datetime.now() - timedelta(days=random.randint(1, 365))
    los = random.randint(2, 12)  # length of stay
    discharge_date = admit_date + timedelta(days=los)
    dob = datetime.now() - timedelta(days=age * 365 + random.randint(0, 364))

    return {
        "patient_name": f"{first_name} {last_name}",
        "mrn": f"MRN-{random.randint(100000, 999999)}",
        "age": str(age),
        "sex": sex,
        "dob": dob.strftime("%m/%d/%Y"),
        "admit_date": admit_date.strftime("%m/%d/%Y"),
        "discharge_date": discharge_date.strftime("%m/%d/%Y"),
        "attending": f"{random.choice(LAST_NAMES)}",
        "provider": f"{random.choice(LAST_NAMES)}",
        "pharmacist": f"{random.choice(FIRST_NAMES_M + FIRST_NAMES_F)} {random.choice(LAST_NAMES)}, PharmD",
        "ordering_md": f"{random.choice(LAST_NAMES)}",
        "radiologist": f"{random.choice(LAST_NAMES)}",
        "note_date": (admit_date + timedelta(days=random.randint(0, los))).strftime("%m/%d/%Y"),
        "comparison_date": (admit_date - timedelta(days=random.randint(30, 180))).strftime("%m/%d/%Y"),
        "day_number": str(random.randint(1, los)),
        "weight": str(round(random.uniform(65, 120), 1)),
        "weight_change": str(round(random.uniform(1.5, 5.0), 1)),
    }


def generate_clinical_notes(
    num_discharge_summaries: int = 50,
    num_progress_notes: int = 80,
    num_med_rec_notes: int = 30,
    num_radiology_reports: int = 40,
    num_lab_interpretations: int = 20,
    output_dir: str = "data/raw",
) -> list[dict]:
    """
    Generate a corpus of synthetic clinical documents.

    This creates a diverse set of clinical documents across multiple conditions
    and document types, each populated with randomized but realistic patient
    demographics and clinical details. The output is both saved as JSON files
    and returned as a list of document dictionaries.

    The volume defaults (50 discharge summaries, 80 progress notes, etc.)
    are designed to create a corpus of ~220 documents — enough to meaningfully
    test chunking strategies and retrieval quality without being unwieldy.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_documents = []

    # Generate discharge summaries — the most information-rich documents
    print(f"Generating {num_discharge_summaries} discharge summaries...")
    for i in range(num_discharge_summaries):
        template_data = random.choice(DISCHARGE_SUMMARIES)
        patient = generate_patient()
        content = template_data["template"].format(**patient)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "discharge_summary",
            "condition": template_data["condition"],
            "patient_name": patient["patient_name"],
            "mrn": patient["mrn"],
            "date": patient["discharge_date"],
            "content": content.strip(),
            "metadata": {
                "document_type": "discharge_summary",
                "primary_condition": template_data["condition"],
                "icd10": CONDITION_ICD10.get(template_data["condition"], {}).get("code", ""),
                "patient_age": patient["age"],
                "patient_sex": patient["sex"],
                "admission_date": patient["admit_date"],
                "discharge_date": patient["discharge_date"],
            },
        }
        all_documents.append(doc)

    # Generate progress notes
    print(f"Generating {num_progress_notes} progress notes...")
    for i in range(num_progress_notes):
        template_data = random.choice(PROGRESS_NOTES)
        patient = generate_patient()
        content = template_data["template"].format(**patient)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "progress_note",
            "condition": template_data["condition"],
            "patient_name": patient["patient_name"],
            "mrn": patient["mrn"],
            "date": patient["note_date"],
            "content": content.strip(),
            "metadata": {
                "document_type": "progress_note",
                "primary_condition": template_data["condition"],
                "hospital_day": patient["day_number"],
                "patient_age": patient["age"],
                "patient_sex": patient["sex"],
            },
        }
        all_documents.append(doc)

    # Generate medication reconciliation notes
    print(f"Generating {num_med_rec_notes} medication reconciliation notes...")
    for i in range(num_med_rec_notes):
        template_data = random.choice(MEDICATION_RECONCILIATION_NOTES)
        patient = generate_patient()
        content = template_data["template"].format(**patient)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "medication_reconciliation",
            "condition": template_data["condition"],
            "patient_name": patient["patient_name"],
            "mrn": patient["mrn"],
            "date": patient["note_date"],
            "content": content.strip(),
            "metadata": {
                "document_type": "medication_reconciliation",
                "primary_condition": template_data["condition"],
                "patient_age": patient["age"],
                "patient_sex": patient["sex"],
            },
        }
        all_documents.append(doc)

    # Generate radiology reports
    print(f"Generating {num_radiology_reports} radiology reports...")
    for i in range(num_radiology_reports):
        template_data = random.choice(RADIOLOGY_REPORTS)
        patient = generate_patient()
        content = template_data["template"].format(**patient)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "radiology_report",
            "condition": template_data["condition"],
            "patient_name": patient["patient_name"],
            "mrn": patient["mrn"],
            "date": patient["note_date"],
            "content": content.strip(),
            "metadata": {
                "document_type": "radiology_report",
                "primary_condition": template_data["condition"],
                "patient_age": patient["age"],
                "patient_sex": patient["sex"],
            },
        }
        all_documents.append(doc)

    # Generate lab interpretations
    print(f"Generating {num_lab_interpretations} lab result interpretations...")
    for i in range(num_lab_interpretations):
        template_data = random.choice(LAB_INTERPRETATIONS)
        patient = generate_patient()
        content = template_data["template"].format(**patient)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "lab_interpretation",
            "condition": template_data["condition"],
            "patient_name": patient["patient_name"],
            "mrn": patient["mrn"],
            "date": patient["note_date"],
            "content": content.strip(),
            "metadata": {
                "document_type": "lab_interpretation",
                "primary_condition": template_data["condition"],
                "patient_age": patient["age"],
                "patient_sex": patient["sex"],
            },
        }
        all_documents.append(doc)

    # Save all documents as a single JSON file
    # NOTE: encoding="utf-8" is critical on Windows, which defaults to cp1252
    # and chokes on characters like ≥, °F, →, and m² that appear in clinical notes
    output_file = output_path / "clinical_notes_corpus.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    # Also save individual documents for easier browsing
    for doc in all_documents:
        doc_file = output_path / f"{doc['type']}_{doc['id'][:8]}.txt"
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write(doc["content"])

    print(f"\n✅ Generated {len(all_documents)} clinical documents")
    print(f"   - {num_discharge_summaries} discharge summaries")
    print(f"   - {num_progress_notes} progress notes")
    print(f"   - {num_med_rec_notes} medication reconciliation notes")
    print(f"   - {num_radiology_reports} radiology reports")
    print(f"   - {num_lab_interpretations} lab result interpretations")
    print(f"\n📁 Saved to: {output_file}")

    return all_documents


if __name__ == "__main__":
    documents = generate_clinical_notes()
