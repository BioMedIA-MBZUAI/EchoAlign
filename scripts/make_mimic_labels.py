#!/usr/bin/env python3
"""Rebuild the MIMIC-IV-Echo split and label files from your own MIMIC-IV-Echo download.

    python scripts/make_mimic_labels.py --mimic-echo <MIMIC-IV-Echo folder>
"""
import argparse
import csv
import gzip
import hashlib
import os
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = "splits/mimic_iv_echo.csv"
PROBE_FILE = "labels/mimic_iv_echo_probe.csv"
ZEROSHOT_FILE = "labels/mimic_iv_echo.csv"
SPLITS = ["TRAIN", "VAL", "TEST"]

TEST_FRAC, VAL_FRAC = 0.20, 0.10


def stable_bucket(subject_id: str) -> float:
    h = hashlib.sha256(subject_id.encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


CLASSIFICATION = [
    "aortic_regurgitation", "aortic_root_dilation", "aortic_stenosis", "aortic_valve_replacement",
    "ascending_aorta_dilation", "av_leaflet_degeneration", "cardiac_tamponade", "ivc_dilation",
    "la_enlargement", "lv_diastolic_dysfunction", "lv_enlargement", "lv_filling_pressure",
    "lv_systolic_dysfunction", "lvh", "lvot_obstruction", "mitral_regurgitation", "mitral_stenosis",
    "mitral_valve_device", "mitral_valve_prolapse", "mitral_valve_thickening", "pericardial_effusion",
    "pulmonary_hypertension", "pulmonic_regurgitation", "ra_enlargement", "rv_enlargement",
    "rv_overload_septal_motion", "rv_systolic_dysfunction", "technically_difficult_study",
    "tricuspid_regurgitation",
]
REGRESSION = [
    "av_mean_grad", "av_pk_vel", "la_vol", "lvedd", "lvef", "lvesd", "mv_peak_e_a", "rest_lvot",
    "sinus_diam", "tapse", "tr_velocity",
]
TASKS = CLASSIFICATION + REGRESSION

AV_PROSTHESES = {
    "AVR homograft", "Ball and Cage", "Bileaflet mechanical", "Bioprosthesis",
    "CoreValve", "Edwards-Sapien", "Evolut", "Lotus", "Mechanical", "Sapien 3",
    "Other prosthesis", "Bileaflet, mild thickened",
}
AV_NATIVE_THICK = {
    "Mild thickened (3)", "Mild thickened (?#)", "Mod thickened", "Mod thickened (3)",
    "Severe thickened", "Bicuspid, mod thickened", "Quadricuspid thickened",
    "Unicuspid/thickened",
}
AV_NATIVE_THIN = {
    "Nl (3 leaflets)", "Nl (?# leaflets)", "three AV leaflets", "Bicuspid thin, mobile",
    "Bicuspid", "Unicuspid",
}
AV_UNREADABLE = {"Not well seen", "can't determine # AV leaflets"}

MV_PROSTHESES = {
    "Annular ring", "Ball and cage", "Bileaflet", "Bioprosthesis", "CardiaAQ-Edwards",
    "Edwards-Sapien", "Mechanical", "MitraClip", "Sapien 3 TMVR",
}
MV_NATIVE_THICK = {
    "Mild thick", "Mild thick, elongated", "Mild thick/Elongated",
    "Mild thick/Myxomatous", "Mod thick", "Mod thick/Elongated", "Myxomatous",
    "Severe thick",
}
MV_NATIVE_THIN = {"Normal", "Elongated"}

CLS_TASKS = {
    "mitral_regurgitation": dict(
        field="mitral_regurg",
        pos={"Mod [2+]", "Moderate-severe (3+)", "Severe (4+)"},
        neg={"Trivial", "Physiologic", "Mild (1+)", "Mild-moderate (1-2+)"},
        null={"Present, can't qualify"}),
    "tricuspid_regurgitation": dict(
        field="tricuspid_regurg",
        pos={"Mod [2+]", "Mod-severe [3+]", "Severe [4+]"},
        neg={"Physiologic", "Trivial", "Mild [1+]", "Mild-mod [1-2+]"},
        null={"Present, can't qualify"}),
    "aortic_regurgitation": dict(
        field="aortic_regurg",
        pos={"Mod [2+]", "Mod-severe [3+]", "Severe [4+]"},
        neg={"Trace", "Trace (NL for prosthesis)", "Mild [1+]", "Mild-mod [1-2+]"},
        null={"Present"}),
    "pulmonic_regurgitation": dict(
        field="pulm_regurg",
        pos={"Mild", "Mod/severe", "Significant",
             "Increased PI end-diastolic velocity (>1.3m/s)"},
        neg={"Physiologic"}, null=set()),
    "aortic_stenosis": dict(
        field="aortic_stenosis",
        pos={"Mod (1.0-1.5cm2)", "Severe (<1.0cm2)",
             "Severe (<0.6cm2/m2)", "Very severe (>5m/s;>60mmHg)",
             "Low flow/low gradient severe"},
        neg={"Minimal", "None, incr grad from AR/stroke vol", "None, incr vel high output",
             "Mild (>1.5cm2)"},
        null={"Cannot exclude", "Present - not quantified"}),
    "mitral_stenosis": dict(
        field="mitral_stenosis",
        pos={"Mild (1.5-2.0cm2) Valvular", "Mild MS from MAC",
             "Moderate (1.0-1.5cm2) Valvular", "Moderate MS from MAC",
             "Severe (<1.0 cm2) Valvular", "Severe MS (<1.5 cm2) from MAC",
             "Valvular - not quantified"},
        neg={"No valvular", "Trivial"}, null=set()),
    "pulmonary_hypertension": dict(
        field="phtn_severity",
        pos={"Mod (TR 37-60mmHg)", "Mod-severe (50-60mmHg)", "Severe (TR >60mmHg)"},
        neg={"Normal (TR<26mmHg)", "HIgh normal (TR 21-25 mmHg)", "PA HTN unlikely",
             "Mild (TR 26-36mmHg)", "Mild-mod (TR 35-40mmHg)", "Borderline (TR 26-30mmHg)"},
        null={"Indeterminate"}),
    "la_enlargement": dict(
        field="la_size",
        pos={"Mod LAE (PLA 5.0-5.9cm)", "Mod LAVI (41-48ml/m2)",
             "Sev LAE (PLA >=6.0cm)", "Severe LAVI(>48ml/m2)",
             "LA Elongated (4Ch>5.2cm)", "Dilated - no qualifier"},
        neg={"Nl (PLA <4.1cm)", "Normal LAVI (<34ml/m2)", "Mild LAE (PLA 4.1-4.9cm)",
             "Mild LAVI (34-41ml/m2)"}, null={"Not well seen"}),
    "ra_enlargement": dict(
        field="ra_size",
        pos={"Mod RAE (4Ch 6.0-6.9cm)", "Sev RAE (4Ch >=7.0cm)", "RAE - no qualifier"},
        neg={"Nl (4Ch <5.1cm)", "Mild RAE (4Ch 5.1-5.9cm)"}, null={"RA not well seen"}),
    "rv_enlargement": dict(
        field="rv_chamber_size",
        pos={"Mod RVE (basal diam 5.0-5.9cm)",
             "Severe RVE (basal diam >=6.0cm)", "RVE (nonspecific)",
             "RV dilated outflow tract"},
        neg={"Normal cavity (basal diam <4.2cm)", "NL for BSA", "RV cavity small",
             "Mild RVE (basal diam 4.2-4.9cm)"},
        null={"Not well seen", "RV size undetermined"}),
    "lv_enlargement": dict(
        field="lv_chamber_size",
        pos={"Mod LVE (LVEDD 6.0-6.9cm)", "Sev LVE (LVEDD>6.9cm)", "Dilated"},
        neg={"Nl (LVEDD<5.7cm)", "Nl LV for BSA", "Small", "Mild LVE (LVEDD 5.7-5.9cm)"},
        null={"Borderline LVE (LVEDD 5.5-5.7cm)"}),
    "lvh": dict(
        field="lv_wall_thickness",
        pos={"Mod symmetric LVH (1.5-1.7cm)", "Severe symmetric (>1.7cm)"},
        neg={"Normal (<1.2cm)", "Mild symmetric LVH (1.1-1.4cm)"},
        null={"Asymmetric (Sept>post 1.5:1)", "Apical hypertrophy", "Focal hypertrophy of XXX",
              "Focal non-obstructive hypertrophy basal septum", "Symmetric (no qualifiers)"}),
    "rv_systolic_dysfunction": dict(
        field="rv_function",
        pos={"Mild global RV hypo", "Moderate global RV hypo", "Severe global hypo", "RV function depressed",
             "Basal RV hypo (McConnell's sign)"},
        neg={"Nl RV function", "Low normal function", "Hyperdynamic"},
        null={"Cannot assess RV function", "RV not well seen", "Apical free wall hypo"}),
    "lv_systolic_dysfunction": dict(
        field="lv_function1",
        pos={"Mild focal dysfunction", "Mild global hypo (LVEF 40-55%)",
             "Mild-mod focal dysfunction", "Mild-mod global hypo (LVEF35-45%)",
             "Mod focal LV dysfunction (LVEF30-40%)", "Mod global hypo (LVEF 30-39%)",
             "Mod-severe focal LV dysfunction (LVEF25-35%)",
             "Mod-severe global hypo (LVEF 25-35%)",
             "Severe focal dysfunction; rest nl (LVEF<30%)",
             "Severe global hypo (LVEF<30%)"},
        neg={"Nl regional LV WMA", "Nl regional/global systolic function."},
        null={"Suboptimal quality, cannot exclude WMA",
              "Current focused - not full regional assess"}),
    "lvot_obstruction": dict(
        field="lv_obstruction",
        pos={"Mild (1.5-2.4m/s) LVOT", "Rest mod (2.5-3.9m/s) LVOT",
             "Rest SEVERE (>=60mmHg) LVOT", "Mid cavitary gradient",
             "Mid cavity & LVOT", "Apical gradient", "Subaortic membrane"},
        neg={"No resting LVOT", "No rest or Valsalva LVOT", "No mid or LVOT",
             "No apical gradient", "abnl resting LVOT flow w/o obstruction"},
        null=set()),
    "pericardial_effusion": dict(
        field="pericardial_effusion",
        pos={"Moderate (1.0-2.0cm)", "Moderate-large", "Large (>2.0cm)"},
        neg={"Trivial", "Very small (<0.5cm)", "Anterior fat pad", "Small (<1.0cm)",
             "Small-moderate"}, null=set()),
    "mitral_valve_prolapse": dict(
        field="mv_prolapse",
        pos={"Mild MVP", "Mild anterior MVP", "Mild bileaflet MVP", "Mild posterior MVP",
             "Mod MVP", "Mod/severe MVP", "Moderate bileaflet MVP",
             "Moderate posterior MVP", "Severe bileaflet MVP", "Severe posterior MVP",
             "Partial anterior flail", "Partial flail", "Partial post flail-scallop",
             "Partial posterior flail", "Failure to coapt",
             "Immobile posterior leaflet"},
        neg={"No MVP", "Straight, no frank MVP"}, null={"Rheumatic deformity"}),
    "av_leaflet_degeneration": dict(
        field="av_leaflets",
        pos=AV_NATIVE_THICK, neg=AV_NATIVE_THIN,
        null=AV_UNREADABLE | AV_PROSTHESES),
    "aortic_valve_replacement": dict(
        field="av_leaflets",
        pos=AV_PROSTHESES, neg=AV_NATIVE_THICK | AV_NATIVE_THIN, null=AV_UNREADABLE),
    "mitral_valve_thickening": dict(
        field="mv_leaflets",
        pos=MV_NATIVE_THICK, neg=MV_NATIVE_THIN,
        null=MV_PROSTHESES | {"Not well seen", "partial flail mitral leaflet"}),
    "mitral_valve_device": dict(
        field="mv_leaflets",
        pos=MV_PROSTHESES, neg=MV_NATIVE_THICK | MV_NATIVE_THIN,
        null={"Not well seen", "partial flail mitral leaflet"}),
    "aortic_root_dilation": dict(
        field="aortic_sinus",
        pos={"Mild (3.7-4.4cm)", "Mod (4.5-5.4cm)"},
        neg={"Nl (<3.7cm)", "Nl index"}, null=set()),
    "ascending_aorta_dilation": dict(
        field="ascending",
        pos={"Mild (3.5-4.4cm)", "Mod (4.5-5.4cm)", "Severe (>5.4cm)"},
        neg={"Nl (nl M<3.9;F<3.6)", "Nl index"}, null={"Not seen"}),
    "ivc_dilation": dict(
        field="ivc",
        pos={">2.1cm, <50% (>15mmHg)", ">2.1cm, >50% (10-15mmHg)",
             "IVC dilated (>2.5cm)"},
        neg={"<2.2cm, >50% (0-5mmHg)", "IVC<=2.1cm, <50% collapse (5-10mmHg)",
             "NL IVC <2.2 cm", "Intubated, IVC<1.5cm, RA <10mmHg"},
        null={"IVC not visualized", "Intubated - cannot assess RA pressure"}),
    "lv_filling_pressure": dict(
        field="diastolic_fcn",
        pos={"E/e' >14; PCWP>18mmHg"},
        neg={"E/e'<8, E/A; PCPP <12mmHg", "E/e'<8, E/A; PCWP <12mmHg",
             "E/e'<8; PCWP <12mmHg"},
        null={"Indeterminate", "Unable to assess diastolic F( )"}),
    "lv_diastolic_dysfunction": dict(
        field="diastolic_grade",
        pos={"I", "II", "III", "III/IV", "IV", "DD Present"}, neg={"0 (nl)"},
        null={"Indeterminate", "Unable to assess"}),
    "technically_difficult_study": dict(
        field="technical_quality",
        pos={"Suboptimal", "Poor"}, neg={"Adequate", "Good"}, null=set()),
    "cardiac_tamponade": dict(
        field="tamponade",
        pos={"Yes", "Early", "RA & RV collapse & respir variation",
             "RA & RV diastolic collapse/tamponade",
             "RA diastolic collapse/early tamponade",
             "RV diastolic collapse/tamponade", "Resp variation MV/TV-tamponade"},
        neg={"No", "No 2D/Doppler signs", "No RA systolic or RV diastolic collapse",
             "No RV diastolic collapse", "No respiratory variation MV/TV",
             "Resp variation MV/TV - no RA/RV collapse"},
        null=set()),
    "rv_overload_septal_motion": dict(
        field="septal_motion",
        pos={"RV pressure overload", "RV pressure/volume overload",
             "RV volume overload"},
        neg={"Normal", "Post-op septal motion", "c/w conduction abnl/pacing",
             "c/w pericardial constriction", "abnormal - nonspecific"}, null=set()),
}
assert sorted(CLS_TASKS) == sorted(CLASSIFICATION)

NUMERIC_RANGES = {
    "lvef": (5.0, 90.0),             # percent
    "lvedd": (2.0, 10.0),            # cm
    "lvesd": (0.8, 9.0),             # cm
    "la_vol": (5.0, 400.0),          # mL, absolute (not indexed)
    "tapse": (0.3, 4.0),             # cm
    "tr_velocity": (0.8, 7.0),       # m/s
    "av_pk_vel": (0.4, 7.0),         # m/s
    "av_mean_grad": (0.5, 150.0),    # mmHg
    "rest_lvot": (0.0, 200.0),       # mmHg
    "mv_peak_e_a": (0.05, 10.0),     # ratio
    "sinus_diam": (1.5, 8.0),        # cm
}
assert sorted(NUMERIC_RANGES) == sorted(REGRESSION)

GRADE_TASKS = {}


def grade_task(name, field, order, mapping):
    GRADE_TASKS[name] = dict(field=field, order=order, map=mapping)


REG_T = {'Physiologic': 'trace', 'Trivial': 'trace', 'Mild [1+]': 'mild', 'Mild-mod [1-2+]': 'compound', 'Mod [2+]': 'moderate',
         'Mod-severe [3+]': 'compound', 'Severe [4+]': 'severe', "Present, can't qualify": 'ungraded'}
REG_M = {'Physiologic': 'trace', 'Trivial': 'trace', 'Mild (1+)': 'mild', 'Mild-moderate (1-2+)': 'compound', 'Mod [2+]': 'moderate',
         'Moderate-severe (3+)': 'compound', 'Severe (4+)': 'severe', "Present, can't qualify": 'ungraded'}
REG_A = {'Trace': 'trace', 'Trace (NL for prosthesis)': 'trace', 'Mild [1+]': 'mild', 'Mild-mod [1-2+]': 'compound', 'Mod [2+]': 'moderate',
         'Mod-severe [3+]': 'compound', 'Severe [4+]': 'severe', 'Present': 'ungraded'}
REG = ['normal', 'trace', 'mild', 'moderate', 'severe']
grade_task('tricuspid_regurgitation', 'tricuspid_regurg', REG, REG_T)
grade_task('mitral_regurgitation', 'mitral_regurg', REG, REG_M)
grade_task('aortic_regurgitation', 'aortic_regurg', REG, REG_A)
grade_task('pulmonic_regurgitation', 'pulm_regurg', REG,
           {'Physiologic': 'trace', 'Mild': 'mild', 'Mod/severe': 'compound', 'Significant': 'ungraded',
            'Increased PI end-diastolic velocity (>1.3m/s)': 'ungraded'})
grade_task('pulmonary_hypertension', 'phtn_severity', ['normal', 'borderline', 'mild', 'moderate', 'severe'],
           {'Normal (TR<26mmHg)': 'normal', 'HIgh normal (TR 21-25 mmHg)': 'normal', 'PA HTN unlikely': 'ungraded_negative',
            'Borderline (TR 26-30mmHg)': 'borderline', 'Mild (TR 26-36mmHg)': 'mild', 'Mild-mod (TR 35-40mmHg)': 'compound',
            'Mod (TR 37-60mmHg)': 'moderate', 'Mod-severe (50-60mmHg)': 'compound', 'Severe (TR >60mmHg)': 'severe',
            'Indeterminate': 'not_assessable'})
grade_task('la_enlargement', 'la_size', ['normal', 'mild', 'moderate', 'severe'],
           {'Normal LAVI (<34ml/m2)': 'normal', 'Mild LAVI (34-41ml/m2)': 'mild', 'Mod LAVI (41-48ml/m2)': 'moderate',
            'Severe LAVI(>48ml/m2)': 'severe', 'Nl (PLA <4.1cm)': 'other_scale', 'Mild LAE (PLA 4.1-4.9cm)': 'other_scale',
            'Mod LAE (PLA 5.0-5.9cm)': 'other_scale', 'Sev LAE (PLA >=6.0cm)': 'other_scale', 'LA Elongated (4Ch>5.2cm)': 'other_scale',
            'Dilated - no qualifier': 'ungraded', 'Not well seen': 'not_assessable'})
grade_task('ra_enlargement', 'ra_size', ['normal', 'mild', 'moderate', 'severe'],
           {'Nl (4Ch <5.1cm)': 'normal', 'Mild RAE (4Ch 5.1-5.9cm)': 'mild', 'Mod RAE (4Ch 6.0-6.9cm)': 'moderate',
            'Sev RAE (4Ch >=7.0cm)': 'severe', 'RAE - no qualifier': 'ungraded', 'RA not well seen': 'not_assessable'})
grade_task('rv_enlargement', 'rv_chamber_size', ['normal', 'mild', 'moderate', 'severe'],
           {'Normal cavity (basal diam <4.2cm)': 'normal', 'NL for BSA': 'normal', 'Mild RVE (basal diam 4.2-4.9cm)': 'mild',
            'Mod RVE (basal diam 5.0-5.9cm)': 'moderate', 'Severe RVE (basal diam >=6.0cm)': 'severe', 'RVE (nonspecific)': 'ungraded',
            'RV dilated outflow tract': 'ungraded', 'RV cavity small': 'other', 'Not well seen': 'not_assessable',
            'RV size undetermined': 'not_assessable'})
grade_task('lv_enlargement', 'lv_chamber_size', ['normal', 'borderline', 'mild', 'moderate', 'severe'],
           {'Nl (LVEDD<5.7cm)': 'normal', 'Nl LV for BSA': 'normal', 'Borderline LVE (LVEDD 5.5-5.7cm)': 'borderline',
            'Mild LVE (LVEDD 5.7-5.9cm)': 'mild', 'Mod LVE (LVEDD 6.0-6.9cm)': 'moderate', 'Sev LVE (LVEDD>6.9cm)': 'severe',
            'Dilated': 'ungraded', 'Small': 'other'})
grade_task('lvh', 'lv_wall_thickness', ['normal', 'mild', 'moderate', 'severe'],
           {'Normal (<1.2cm)': 'normal', 'Mild symmetric LVH (1.1-1.4cm)': 'mild', 'Mod symmetric LVH (1.5-1.7cm)': 'moderate',
            'Severe symmetric (>1.7cm)': 'severe', 'Symmetric (no qualifiers)': 'ungraded', 'Asymmetric (Sept>post 1.5:1)': 'ungraded',
            'Apical hypertrophy': 'ungraded', 'Focal hypertrophy of XXX': 'ungraded',
            'Focal non-obstructive hypertrophy basal septum': 'ungraded'})
grade_task('rv_systolic_dysfunction', 'rv_function', ['normal', 'low_normal', 'mild', 'moderate', 'severe'],
           {'Nl RV function': 'normal', 'Low normal function': 'low_normal', 'Hyperdynamic': 'other', 'Mild global RV hypo': 'mild',
            'Moderate global RV hypo': 'moderate', 'Severe global hypo': 'severe', 'RV function depressed': 'ungraded',
            'Apical free wall hypo': 'ungraded', "Basal RV hypo (McConnell's sign)": 'ungraded',
            'Cannot assess RV function': 'not_assessable', 'RV not well seen': 'not_assessable'})
AV_OTHER = ['Bicuspid', 'Bicuspid thin, mobile', 'Bicuspid, mod thickened', 'Unicuspid', 'Unicuspid/thickened', 'Quadricuspid thickened',
            'AVR homograft', 'Ball and Cage', 'Bileaflet mechanical', 'Bioprosthesis', 'CoreValve', 'Edwards-Sapien', 'Evolut', 'Lotus',
            'Mechanical', 'Sapien 3', 'Other prosthesis', 'Bileaflet, mild thickened']
grade_task('av_leaflet_thickening', 'av_leaflets', ['normal', 'mild', 'moderate', 'severe'],
           {'Nl (3 leaflets)': 'normal', 'Nl (?# leaflets)': 'normal', 'three AV leaflets': 'normal', 'Mild thickened (3)': 'mild',
            'Mild thickened (?#)': 'mild', 'Mod thickened': 'moderate', 'Mod thickened (3)': 'moderate', 'Severe thickened': 'severe',
            'Not well seen': 'not_assessable', "can't determine # AV leaflets": 'not_assessable', **{v: 'other' for v in AV_OTHER}})
grade_task('lv_diastolic_dysfunction', 'diastolic_grade', ['normal', 'grade_i', 'grade_ii', 'grade_iii', 'grade_iv'],
           {'0 (nl)': 'normal', 'I': 'grade_i', 'II': 'grade_ii', 'III': 'grade_iii', 'III/IV': 'compound', 'IV': 'grade_iv',
            'DD Present': 'ungraded', 'Indeterminate': 'not_assessable', 'Unable to assess': 'not_assessable'})
grade_task('aortic_stenosis', 'aortic_stenosis', ['normal', 'minimal', 'mild', 'moderate', 'severe', 'very_severe'],
           {'None, incr grad from AR/stroke vol': 'normal', 'None, incr vel high output': 'normal', 'Minimal': 'minimal',
            'Mild (>1.5cm2)': 'mild', 'Mod (1.0-1.5cm2)': 'moderate', 'Severe (<1.0cm2)': 'severe', 'Severe (<0.6cm2/m2)': 'severe',
            'Low flow/low gradient severe': 'severe', 'Very severe (>5m/s;>60mmHg)': 'very_severe', 'Cannot exclude': 'ungraded',
            'Present - not quantified': 'ungraded'})
grade_task('pericardial_effusion', 'pericardial_effusion', ['normal', 'trace', 'small', 'moderate', 'large'],
           {'Trivial': 'trace', 'Very small (<0.5cm)': 'small', 'Small (<1.0cm)': 'small', 'Small-moderate': 'compound',
            'Moderate (1.0-2.0cm)': 'moderate', 'Moderate-large': 'compound', 'Large (>2.0cm)': 'large', 'Anterior fat pad': 'other'})
LVF_FOCAL = ['Mild focal dysfunction', 'Mild-mod focal dysfunction', 'Mod focal LV dysfunction (LVEF30-40%)',
             'Mod-severe focal LV dysfunction (LVEF25-35%)', 'Severe focal dysfunction; rest nl (LVEF<30%)']
grade_task('lv_systolic_dysfunction', 'lv_function1', ['normal', 'mild', 'moderate', 'severe'],
           {'Nl regional/global systolic function.': 'normal', 'Nl regional LV WMA': 'other', 'Mild global hypo (LVEF 40-55%)': 'mild',
            'Mild-mod global hypo (LVEF35-45%)': 'compound', 'Mod global hypo (LVEF 30-39%)': 'moderate',
            'Mod-severe global hypo (LVEF 25-35%)': 'compound', 'Severe global hypo (LVEF<30%)': 'severe',
            'Suboptimal quality, cannot exclude WMA': 'not_assessable', 'Current focused - not full regional assess': 'not_assessable',
            **{v: 'other' for v in LVF_FOCAL}})
grade_task('lv_ef_band', 'lv_function2', ['normal', 'borderline', 'mild', 'moderate', 'severe'],
           {'Nl (M>52; F>54%)': 'normal', 'Borderline (EF50-55%)': 'borderline', 'Mild (EF 40-55%)': 'mild', 'Mild-mod (EF 35-45%)': 'compound',
            'Mod (EF 30-39%)': 'moderate', 'Mod-sev (EF 25-35%)': 'compound', 'Sev (<30%)': 'severe', 'Hyper (EF>75%)': 'other',
            'Depressed': 'ungraded', "Can't assess": 'not_assessable'})
grade_task('aortic_root_dilation', 'aortic_sinus', ['normal', 'mild', 'moderate'],
           {'Nl (<3.7cm)': 'normal', 'Nl index': 'normal', 'Mild (3.7-4.4cm)': 'mild', 'Mod (4.5-5.4cm)': 'moderate'})
grade_task('ascending_aorta_dilation', 'ascending', ['normal', 'mild', 'moderate', 'severe'],
           {'Nl (nl M<3.9;F<3.6)': 'normal', 'Nl index': 'normal', 'Mild (3.5-4.4cm)': 'mild', 'Mod (4.5-5.4cm)': 'moderate',
            'Severe (>5.4cm)': 'severe', 'Not seen': 'not_assessable'})
grade_task('aortic_arch_dilation', 'arch', ['normal', 'mild', 'moderate', 'severe'],
           {'Normal (<3.1cm)': 'normal', 'Mild (3.1-3.9cm)': 'mild', 'Mod (4.0-4.9cm)': 'moderate', 'Severe (>4.9cm)': 'severe',
            'Not well seen': 'not_assessable'})
grade_task('descending_aorta_dilation', 'descending', ['normal', 'mild', 'moderate'],
           {'Normal (<2.5cm)': 'normal', 'Mild (2.5-3.4cm)': 'mild', 'Mod (3.5-4.9cm)': 'moderate', 'Not well seen': 'not_assessable'})
grade_task('mitral_stenosis', 'mitral_stenosis', ['normal', 'trace', 'mild', 'moderate', 'severe'],
           {'No valvular': 'normal', 'Trivial': 'trace', 'Mild (1.5-2.0cm2) Valvular': 'mild', 'Mild MS from MAC': 'mild',
            'Moderate (1.0-1.5cm2) Valvular': 'moderate', 'Moderate MS from MAC': 'moderate', 'Severe (<1.0 cm2) Valvular': 'severe',
            'Severe MS (<1.5 cm2) from MAC': 'severe', 'Valvular - not quantified': 'ungraded'})
grade_task('pulmonic_stenosis', 'pulm_stenosis', ['normal', 'trace', 'mild', 'moderate'],
           {'None (<1.5m/s)': 'normal', 'No PS, high vel due to high output': 'normal', 'Trivial': 'trace', 'Trivial (1.5-1.9m/s)': 'trace',
            'Mild': 'mild', 'Mild (1.5-2.9m/s)': 'mild', 'Mod (2.5-3.0m/s)': 'moderate', 'Moderate': 'moderate', 'Undetermined': 'not_assessable'})
MV_DEV = ['Annular ring', 'Ball and cage', 'Bileaflet', 'Bioprosthesis', 'CardiaAQ-Edwards', 'Edwards-Sapien', 'Mechanical', 'MitraClip', 'Sapien 3 TMVR']
grade_task('mitral_valve_thickening', 'mv_leaflets', ['normal', 'mild', 'moderate', 'severe'],
           {'Normal': 'normal', 'Mild thick': 'mild', 'Mild thick, elongated': 'mild', 'Mild thick/Elongated': 'mild', 'Mild thick/Myxomatous': 'mild',
            'Mod thick': 'moderate', 'Mod thick/Elongated': 'moderate', 'Severe thick': 'severe', 'Elongated': 'other', 'Myxomatous': 'other',
            'partial flail mitral leaflet': 'other', 'Not well seen': 'not_assessable', **{v: 'other' for v in MV_DEV}})
grade_task('tv_leaflet_thickening', 'tv_leaflets', ['normal', 'mild', 'moderate'],
           {'Normal': 'normal', 'Mild thick': 'mild', 'Mod thick': 'moderate', 'Myxomatous': 'other', 'Elongated': 'other', 'TV Prolapse': 'other',
            'Ebsteins': 'other', 'Annuloplasty ring': 'other', 'Bioprosthetic': 'other', 'MitraClip': 'other', 'Other prosthesis': 'other',
            'Not well seen': 'not_assessable'})
grade_task('pv_leaflet_thickening', 'pv_leaflets', ['normal', 'mild', 'moderate'],
           {'Normal': 'normal', 'Mildly thickened': 'mild', 'Mod/sev thickened': 'compound', 'Abnormal': 'ungraded', 'Not well seen': 'not_assessable'})
MVP_OTHER = ['Partial anterior flail', 'Partial flail', 'Partial post flail-scallop', 'Partial posterior flail', 'Failure to coapt',
             'Immobile posterior leaflet', 'Rheumatic deformity']
grade_task('mitral_valve_prolapse', 'mv_prolapse', ['normal', 'mild', 'moderate', 'severe'],
           {'No MVP': 'normal', 'Straight, no frank MVP': 'normal', 'Mild MVP': 'mild', 'Mild anterior MVP': 'mild', 'Mild bileaflet MVP': 'mild',
            'Mild posterior MVP': 'mild', 'Mod MVP': 'moderate', 'Moderate bileaflet MVP': 'moderate', 'Moderate posterior MVP': 'moderate',
            'Mod/severe MVP': 'compound', 'Severe bileaflet MVP': 'severe', 'Severe posterior MVP': 'severe', **{v: 'other' for v in MVP_OTHER}})
grade_task('mitral_annular_calcification', 'mac_severity', ['normal', 'mild', 'moderate', 'severe'],
           {'Mild': 'mild', 'Mod MAC': 'moderate', 'Severe': 'severe'})
grade_task('chordal_thickening', 'chordal_thickening', ['normal', 'mild', 'moderate', 'severe'],
           {'Mild chordal thickening': 'mild', 'Mod chordal thickening': 'moderate', 'Severe thickening': 'severe', 'Torn': 'other'})
UNGRADED_POS = {'ungraded'}

def grade_study(values, spec):
    order, mp = spec['order'], spec['map']
    cats = [mp[v] for v in values]
    if 'compound' in cats:
        return 'compound'
    on = [c for c in cats if c in order]
    if on:
        top = max(on, key=order.index)
        if top == 'normal' and any(c in UNGRADED_POS for c in cats):
            return 'conflict'
        return top
    for c in ['ungraded', 'ungraded_negative', 'not_assessable', 'other_scale', 'other']:
        if c in cats:
            return c
    raise AssertionError(cats)


def meta_dir(root):
    for d in (root / "meta", root):
        if (d / "echo-record-list.csv").exists():
            return d
    raise SystemExit(f"no echo-record-list.csv in {root}/meta or {root}")


def build_split(root, meta):
    studies = defaultdict(lambda: {"subject_id": None, "n_dicom": 0})
    seen_paths = set()
    with open(meta / "echo-record-list.csv") as f:
        for r in csv.DictReader(f):
            p = r["dicom_filepath"]
            if p in seen_paths:
                continue
            seen_paths.add(p)
            if not os.path.exists(f"{root}/{p}"):
                continue
            s = studies[r["study_id"]]
            s["subject_id"] = r["subject_id"]
            s["n_dicom"] += 1
    patients = {v["subject_id"] for v in studies.values()}
    assign = {}
    for p in patients:
        b = stable_bucket(p)
        assign[p] = "TEST" if b < TEST_FRAC else ("VAL" if b < TEST_FRAC + VAL_FRAC else "TRAIN")
    rows = [dict(study_id=sid, cluster_id=s["subject_id"], split=assign[s["subject_id"]], n_clips=s["n_dicom"])
            for sid, s in sorted(studies.items())]
    split = pd.DataFrame(rows, columns=["study_id", "cluster_id", "split", "n_clips"])
    assert split.groupby("cluster_id").split.nunique().max() == 1, "a patient spans two splits"
    print(f"studies with >=1 DICOM on disk: {len(split):,}; patients: {len(patients):,}; "
          f"DICOM files: {split.n_clips.sum():,}")
    print("studies per split: " + ", ".join(f"{k}={v:,}" for k, v in split.split.value_counts().reindex(SPLITS).items()))
    return split


def read_measurements(meta, split_of):
    mid_to_study = {}
    with open(meta / "echo-study-list.csv", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("measurement_id") or "").strip()
            if row["study_id"] in split_of and mid:
                mid_to_study[mid] = row["study_id"]
    join_rate = len(set(mid_to_study.values())) / len(split_of)
    print(f"studies with a structured report: {len(set(mid_to_study.values())):,} ({join_rate:.4f})")
    assert join_rate > 0.95, f"join rate collapsed to {join_rate:.4f}"

    label_fields = {t["field"] for t in CLS_TASKS.values()} | set(NUMERIC_RANGES)
    grade_fields = {t["field"] for t in GRADE_TASKS.values()}
    vals = defaultdict(lambda: defaultdict(list))
    grade_vals = defaultdict(lambda: defaultdict(list))
    with gzip.open(meta / "structured-measurement.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            field = (row.get("measurement") or "").strip()
            study = mid_to_study.get((row.get("measurement_id") or "").strip())
            result = (row.get("result") or "").strip()
            if study is None or not result:
                continue
            if field in label_fields:
                vals[field][study].append(result)
            if field in grade_fields:
                grade_vals[field][study].append(result)
    return vals, grade_vals


def binary(rule, vs):
    if any(v in rule["pos"] for v in vs):
        return 1.0
    if any(v in rule["neg"] for v in vs):
        return 0.0
    return None


def classification_labels(vals):
    out, unmapped = {}, {}
    for task in CLASSIFICATION:
        t = CLS_TASKS[task]
        sm = vals.get(t["field"], {})
        extra = sorted({v for vs in sm.values() for v in vs} - (t["pos"] | t["neg"] | t["null"]))
        if extra:
            unmapped[task] = extra
        lab = {s: binary(t, vs) for s, vs in sm.items()}
        out[task] = {s: y for s, y in lab.items() if y is not None}
    if unmapped:
        raise SystemExit(f"result strings in no value set of their task (the release differs from v1.0): {unmapped}")
    return out


def regression_labels(vals):
    out = {}
    for field in REGRESSION:
        lo, hi = NUMERIC_RANGES[field]
        lab = {}
        for study, vs in vals.get(field, {}).items():
            numeric = []
            for v in vs:
                try:
                    x = float(v)
                except ValueError:
                    continue
                if lo <= x <= hi:
                    numeric.append(x)
            if numeric:
                lab[study] = statistics.median(numeric)
        out[field] = lab
    return out


def study_grades(grade_vals, name):
    spec = GRADE_TASKS[name]
    sm = grade_vals.get(spec["field"], {})
    miss = sorted({v for vs in sm.values() for v in vs} - set(spec["map"]))
    if miss:
        raise SystemExit(f"{name}: result strings with no grade (the release differs from v1.0): {miss}")
    return {s: grade_study(vs, spec) for s, vs in sorted(sm.items())}


def grade_variant(name, grades):
    order = GRADE_TASKS[name]["order"]
    if any(g == "normal" for g in grades.values()):
        return name, order
    return f"{name}_nonone", order[1:]


MIN_MERGED_TEST = 5


def grade_missing(kept, rank, split_of):
    return any({k for s, k in kept.items() if split_of[s] == sp} != set(rank.values()) for sp in SPLITS)


def grade_rows(grade_vals, split_of, subject):
    rows = []
    for name in GRADE_TASKS:
        grades = study_grades(grade_vals, name)
        task, order = grade_variant(name, grades)
        present = {g for g in grades.values() if g in order}
        levels = [g for g in order if g in present]
        if len(levels) < 3:
            continue
        rank = {g: k for k, g in enumerate(levels)}
        kept = {s: rank[g] for s, g in grades.items() if g in rank}
        if grade_missing(kept, rank, split_of):
            # Merge the top two grades once. The task needs three levels, every level in every split, and
            # MIN_MERGED_TEST test studies in the merged grade.
            rank[levels[-1]] = rank[levels[-2]]
            kept = {s: rank[g] for s, g in grades.items() if g in rank}
            if len(set(rank.values())) < 3 or grade_missing(kept, rank, split_of):
                print(f"{task}_grade: a grade is missing from a split after merging the top two, skipped")
                continue
            top = rank[levels[-1]]
            if sum(k == top and split_of[s] == "TEST" for s, k in kept.items()) < MIN_MERGED_TEST:
                print(f"{task}_grade: fewer than {MIN_MERGED_TEST} test studies in the merged grade, skipped")
                continue
            print(f"{task}_grade: a grade is missing from a split, merged {levels[-2]} and {levels[-1]}")
        for sp in SPLITS:
            for s in sorted(s for s in kept if split_of[s] == sp):
                rows.append((s, f"{task}_grade", sp, kept[s], "ordinal", subject[s]))
    return rows


def build_labels(split, vals, grade_vals):
    split_of = dict(zip(split.study_id, split.split))
    subject = dict(zip(split.study_id, split.cluster_id))
    labels = classification_labels(vals)
    labels.update(regression_labels(vals))

    rows = []
    for task in TASKS:
        mode = "classification" if task in CLS_TASKS else "regression"
        lab = labels[task]
        for sp in SPLITS:
            for s in sorted(s for s in lab if split_of[s] == sp):
                y = int(lab[s]) if mode == "classification" else lab[s]
                rows.append((s, task, sp, y, mode, subject[s]))
    rows += grade_rows(grade_vals, split_of, subject)

    probe = pd.DataFrame(rows, columns=["study_id", "task", "split", "label", "mode", "cluster_id"])
    probe["label"] = probe["label"].astype("float64")
    zs = probe[(probe["mode"] == "classification") & (probe.split == "TEST")]
    zs = zs[["study_id", "task", "label", "cluster_id"]].assign(label=lambda d: d.label.astype(int))
    return probe, zs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-echo", required=True, help="root of the MIMIC-IV-Echo v1.0 download (holds meta/ and files/)")
    ap.add_argument("--out", default=str(ROOT / "data"), help="output folder; files go to <out>/splits and <out>/labels")
    a = ap.parse_args()

    root, out = Path(a.mimic_echo), Path(a.out)
    meta = meta_dir(root)
    split = build_split(root, meta)
    split_of = dict(zip(split.study_id, split.split))
    vals, grade_vals = read_measurements(meta, split_of)
    probe, zs = build_labels(split, vals, grade_vals)

    for rel, df in ((SPLIT_FILE, split), (PROBE_FILE, probe), (ZEROSHOT_FILE, zs)):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote {path}: {len(df):,} rows")
    print(probe.groupby(["mode", "split"]).task.nunique().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
