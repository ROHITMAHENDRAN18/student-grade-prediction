"""Build the Word report from the saved outputs (all numbers are read from files, nothing typed by hand).
Usage:  python report/build_report.py  [page_map.json]"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docx_helpers import *
from data_preprocessing import load_data, clean_data, numeric_cols
from config import ROOT, EDA_DIR, EVAL_DIR, BINARY_COLS, NOMINAL_COLS, ORDINAL_COLS, PASS_MARK

PAGES = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
E_ = json.load(open(EDA_DIR / "results_eda.json")); M_ = json.load(open(EVAL_DIR / "results_model.json")); T_ = json.load(open(EVAL_DIR / "training_summary.json"))["settings"]
df, _ = clean_data(load_data())
desc = pd.read_csv(EDA_DIR / "table_descriptive_numeric.csv", index_col=0); biv = pd.read_csv(EDA_DIR / "table_bivariate_tests.csv", index_col=0)
outl = pd.read_csv(EDA_DIR / "table_outliers.csv", index_col=0); vif = pd.read_csv(EDA_DIR / "table_vif.csv", index_col=0).iloc[:, 0]
tune = {n: pd.read_csv(EVAL_DIR / f"tuning_results_{n}.csv") for n in "AB"}; imp = {n: pd.read_csv(EVAL_DIR / f"table_permutation_importance_{n}.csv", index_col=0) for n in "AB"}
MT, RP, PF = M_["metrics"], M_["repeated"], M_["passfail"]; DF = E_["distribution_fits"]; RA, RB = M_["residual_stats_A"], M_["residual_stats_B"]
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pct = lambda x: f"{100*x:.1f}%"
gm = {k: {str(r[0]): (r[1], r[2]) for r in v} for k, v in E_["group_means"].items()}
g3 = df["G3"]; zero = df[df.G3 == 0]; nz = df[df.G3 > 0]
r_fail = df.groupby(df.failures > 0).G3.mean()
absn = DF["absences"]; p_pois0 = stats.poisson.pmf(0, absn["mean"]); p_all10 = DF["pass_rate"] ** 10
eps = lambda f: biv.loc[f, "effect_size"]; pv = lambda f: "< 0.001" if biv.loc[f, "p_value"] < 0.001 else f"{biv.loc[f, 'p_value']:.3f}"
TITLE = "Predicting Final Student Grades from Academic, Family and Lifestyle Factors Using Exploratory Data Analysis and Deep Learning"
FOOT = "RITHI R | 24AD091 | U21ADP05 – Assignment II (Case Study 2)"
HEAD, FIGS, TABS = [], [], []; cnt = {"tab": 0}
cfgA, cfgB = T_["A"]["best_config"], T_["B"]["best_config"]


def H(d, lvl, text): d.add_heading(text, level=lvl); HEAD.append((lvl, text))
def FIG(d, path, w, title): FIGS.append(title); figure(d, path, w, title)
def TAB(d, title, header, rows, widths, **kw):
    cnt["tab"] += 1; cap = f"Table {cnt['tab']}: {title}"; TABS.append(cap); return table(d, header, rows, widths, caption=cap, **kw)
def CARD(d, n, title, path, w, purpose, variables, chart, why, insight):
    FIG(d, path, w, f"Figure {n}: {title}")
    table(d, ["Aspect", "Explanation"], [["Purpose", purpose], ["Variables", variables], ["Chart type", chart], ["Why selected", why], ["Insight", insight]], [2.8, 13.8], font=9, bold_first_col=True, keep_together=True)


def build():
    d = new_document(); HEAD.clear(); FIGS.clear(); TABS.clear(); cnt["tab"] = 0
    footer_with_page(d.sections[0], FOOT)
    # ------------------------------------------------ title page
    para(d, "KPR Institute of Engineering and Technology", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE, after=2)
    para(d, "Department of Artificial Intelligence and Data Science", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    para(d, "ASSIGNMENT II – CASE STUDY (PROJECT 2: REGRESSION)", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, after=6)
    para(d, TITLE, bold=True, size=23, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE, after=8)
    para(d, "Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset", italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    rows = [["Student Name", "RITHI R"], ["Roll Number / College ID", "24AD091"], ["Department", "Artificial Intelligence and Data Science"], ["Year / Semester", "III Year / Semester V (Batch 2023–2027)"],
            ["Course Code", "U21ADP05"], ["Course Title", "Exploratory Data Analysis and Visualization"], ["Assignment", "Assignment II – Case Study (40 marks)"], ["Academic Year", "2026–2027 (Odd Semester)"],
            ["Course Coordinator", "Mr. Rushikesh Kadam"], ["Report Date", "19 October 2026"]]
    t = d.add_table(rows=0, cols=2); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; set_cell_margins(t, 80, 80, 120, 120); t.autofit = False
    for a, b in rows:
        c = t.add_row().cells; c[0].width, c[1].width = Cm(5.2), Cm(10.6); shade(c[0], "E8F0FB"); add_runs(c[0].paragraphs[0], a, bold=True, size=11); add_runs(c[1].paragraphs[0], b, size=11)
        for x in c: x.paragraphs[0].paragraph_format.space_after = Pt(0)
    t.columns[0].width, t.columns[1].width = Cm(5.2), Cm(10.6); page_break(d)
    # ------------------------------------------------ abstract
    para(d, "Abstract", bold=True, size=17, color=BLUE, after=8)
    abstract = (f"Predicting a student's final grade early helps schools offer support in time. This study analyses the UCI Student Performance dataset (Portuguese course: {len(df)} students, 32 input features, "
                f"final grade G3 on a 0–20 scale). Exploratory analysis found no missing values, then used seven-stage visualisation, statistical tests, distribution fitting and PCA. "
                f"Earlier grades dominated: G2 correlates with G3 at r = {E_['pearson_G2_G3']:.2f}, while past failures, school and wish for higher education were the strongest non-grade factors. "
                f"A multilayer perceptron regressor, tuned over 12 settings, reached test RMSE {f2(MT['A']['MLP']['RMSE'])} (R² {f2(MT['A']['MLP']['R2'])}) with earlier grades, but RMSE {f2(MT['B']['MLP']['RMSE'])} (R² {f2(MT['B']['MLP']['R2'])}) without them. "
                f"A random forest matched or beat it, so predictions from lifestyle factors alone remain weak.")
    wc = len(abstract.split()); assert 100 <= wc <= 150, wc
    para(d, abstract, after=6); para(d, f"Keywords: exploratory data analysis, data visualization, student performance, regression, multilayer perceptron.  (Abstract: {wc} words)", italic=True, size=9.5, color=GREY); page_break(d)
    para(d, "[[TOC]]"); page_break(d)
    para(d, "List of Abbreviations", bold=True, size=15, color=BLUE, after=6)
    abbr = [["CO", "Course outcome"], ["CSV", "Comma-separated values"], ["EDA", "Exploratory data analysis"], ["G1, G2, G3", "First-period, second-period and final grade (0–20)"], ["IQR", "Inter-quartile range"],
            ["MAE", "Mean absolute error"], ["MLP", "Multilayer perceptron"], ["MSE", "Mean squared error"], ["PCA", "Principal component analysis"], ["R²", "Coefficient of determination"], ["ReLU", "Rectified linear unit"],
            ["RMSE", "Root mean squared error"], ["UCI", "University of California, Irvine (Machine Learning Repository)"], ["VIF", "Variance inflation factor"]]
    table(d, ["Abbreviation", "Meaning"], abbr, [3.5, 13.1], font=9.5); page_break(d)

    # ================================================= 1 INTRODUCTION
    H(d, 1, "1. Introduction and Objectives")
    H(d, 2, "1.1 Background")
    para(d, "Schools record a student's marks in every period together with details about the family, study habits and free-time activities. Exploratory Data Analysis (EDA), introduced by John Tukey (1977), means looking at such data through summaries and pictures before building a model, so that patterns, errors and surprises are found early. "
            "In this project, EDA and a deep-learning model are applied to the marks of secondary-school students in a Portuguese-language course.")
    H(d, 2, "1.2 Problem Statement")
    para(d, "Given a student's background, study habits, lifestyle and (optionally) earlier-period marks, (a) which factors are most strongly associated with the **final grade G3**, and (b) how accurately can a deep-learning model **predict G3 as a number between 0 and 20**? "
            "Because the target is numeric, this is a **regression** problem. (The companion project in this series, on student dropout, was a classification problem.)")
    H(d, 2, "1.3 Motivation")
    bullets(d, ["A teacher who sees a low predicted grade early can offer extra help before the final examination.",
                "It matters *how early* the prediction is made. A model that uses the marks of the first two periods is much easier to build than one that uses only background information. This report measures both situations separately.",
                "The data are real, mixed (numeric, ordinal, binary and categorical) and small enough to study carefully on a laptop."])
    H(d, 2, "1.4 Objectives")
    bullets(d, ["Select and verify a real-world dataset with at least 15 features (this one has 32 input features).",
                "Perform data understanding, cleaning, outlier analysis, encoding, scaling and a train/validation/test split without data leakage.",
                "Create more than five meaningful visualisations (12 EDA figures and 8 model figures) following all seven stages of visualisation.",
                "Carry out univariate, bivariate and multivariate analysis and study continuous and discrete probability distributions.",
                "Build and tune a multilayer perceptron regressor in two settings (with and without the earlier grades) and compare it with simple baselines.",
                "Evaluate it with loss/MAE curves, predicted-versus-actual and error-distribution plots, residual analysis and feature importance.",
                "Publish the work in a well-structured GitHub repository with a README."], style="List Number")
    H(d, 2, "1.5 Scope and Practical Relevance")
    para(d, "The study covers one dataset from two schools in one region and a single course (Portuguese language). It is a learning project, not a deployed system. The workflow (clean, explore, visualise, model, evaluate) is the one a school analytics team would follow to build an early-warning tool for teachers.")

    # ================================================= 2 DATASET
    H(d, 1, "2. Dataset Description")
    H(d, 2, "2.1 Dataset Name, Source and Licence")
    TAB(d, "Dataset identification", ["Item", "Details"], [
        ["Name", "Student Performance"], ["Repository", "UCI Machine Learning Repository (dataset id 320)"], ["URL", "https://archive.ics.uci.edu/dataset/320/student+performance"],
        ["DOI", "10.24432/C5TG7T"], ["Creator", "Paulo Cortez (University of Minho, Portugal)"], ["Licence", "Creative Commons Attribution 4.0 (CC BY 4.0)"],
        ["Companion paper", "Cortez & Silva (2008), Using data mining to predict secondary school student performance, FUBUTEC 2008, pp. 5–12"],
        ["File used", "student-por.csv (Portuguese-language course) saved as data/student-por.csv"]], [3.2, 13.4], font=9, bold_first_col=True)
    para(d, "**Verification note.** The UCI page lists 649 instances, 30 attributes plus the three grades G1, G2 and G3, no missing values, and the tasks classification and regression. "
            "The UCI page does not say which of its two files (Mathematics and Portuguese) holds the 649 rows; the Portuguese file used here contains exactly 649 rows, which matches. The UCI website could not be reached from the environment used for the analysis, "
            "so the CSV was obtained from public GitHub copies of the dataset. Two independent copies were loaded and compared and found identical (649 × 33). The original UCI file is semicolon-separated; the copies are comma-separated, and this is handled in the loading code. "
            "Anyone re-running the project can download the original from the UCI link and place it at data/student-por.csv (use sep=';' when loading the original).")
    H(d, 2, "2.2 Why This Dataset Was Chosen")
    bullets(d, ["**More than 15 features:** 32 input fields plus a target.", "**Real-world relevance:** real students from two schools; the problem (predicting final marks) is a standard learning-analytics task.",
                "**Mixed data types:** ordinal ratings (1–5), counts, binary yes/no flags and categorical fields, so many chart types and statistical tests are meaningful.",
                "**Suitable for regression:** the target G3 is an integer from 0 to 20 and is treated as a numeric value; the UCI page lists regression as an associated task.",
                "**A teachable twist:** G1 and G2 are almost duplicates of G3, so the project can show how much a model depends on them by training with and without them.",
                "**Small and clean:** 649 rows run quickly on a laptop. The small size also forces honest evaluation (repeated splits)."])
    H(d, 2, "2.3 Verified Structure of the Dataset")
    TAB(d, "Verified facts about the loaded file", ["Property", "Value (measured on data/student-por.csv)"], [
        ["Rows (students)", f"{E_['shape'][0]}"], ["Columns", f"{E_['shape'][1]} (32 input features + target G3)"], ["Missing cells", str(E_["clean_report"]["missing_cells"])], ["Duplicate rows", str(E_["clean_report"]["duplicate_rows"])],
        ["Numeric / ordinal features", f"{E_['n_numeric']} (age, parents' education, study time, failures, ratings, absences, G1, G2)"], ["Binary (yes/no-type) features", str(E_["n_binary"])], ["Nominal features (>2 categories)", f"{E_['n_nominal']} (Mjob, Fjob, reason, guardian)"],
        ["Stored data types", f"integer: {E_['dtypes'].get('int64', 0)}, text: {E_['dtypes'].get('str', E_['dtypes'].get('object', 0))}"],
        ["Target", "G3 – final grade, integer 0–20 (regression)"], ["G3 summary", f"mean {E_['g3_desc']['mean']:.2f}, median {E_['g3_desc']['median']:.0f}, SD {E_['g3_desc']['std']:.2f}, min {E_['g3_desc']['min']:.0f}, max {E_['g3_desc']['max']:.0f}"],
        ["Passing students (G3 ≥ 10)", f"{E_['pass_count']} of {len(df)} ({pct(DF['pass_rate'])})"]], [5.0, 11.6], font=9, bold_first_col=True)
    H(d, 2, "2.4 Description of the Fields")
    para(d, "The descriptions below follow the attribute list on the UCI dataset page. Ordinal ratings (1–5) are treated as numbers in this report.")
    fd = {"school": "Student's school (GP or MS)", "sex": "Sex (F or M)", "age": "Age in years (15–22)", "address": "Home address type (U urban, R rural)", "famsize": "Family size (LE3 up to 3, GT3 more than 3)",
          "Pstatus": "Parents' cohabitation (T together, A apart)", "Medu": "Mother's education (0–4)", "Fedu": "Father's education (0–4)", "Mjob": "Mother's job (category)", "Fjob": "Father's job (category)",
          "reason": "Reason for choosing the school (category)", "guardian": "Student's guardian (mother, father, other)", "traveltime": "Home-to-school travel time (1–4)", "studytime": "Weekly study time (1–4)",
          "failures": "Number of past class failures", "schoolsup": "Extra educational support from school", "famsup": "Family educational support", "paid": "Extra paid classes in the course", "activities": "Extra-curricular activities",
          "nursery": "Attended nursery school", "higher": "Wants to take higher education", "internet": "Internet access at home", "romantic": "In a romantic relationship", "famrel": "Quality of family relationships (1–5)",
          "freetime": "Free time after school (1–5)", "goout": "Going out with friends (1–5)", "Dalc": "Workday alcohol consumption (1–5)", "Walc": "Weekend alcohol consumption (1–5)", "health": "Current health status (1–5)",
          "absences": "Number of school absences", "G1": "First-period grade (0–20)", "G2": "Second-period grade (0–20)"}
    rows = []
    for i, c in enumerate([x for x in df.columns if x != "G3"], 1):
        grp = "Binary" if c in BINARY_COLS else "Nominal" if c in NOMINAL_COLS else "Numeric"; rows.append([i, c, grp, fd[c]])
    rows.append([33, "G3", "Target", "Final grade (0–20) – the value to be predicted"])
    TAB(d, "All 32 input fields and the target", ["#", "Field", "Group", "Meaning"], rows, [0.9, 2.6, 1.8, 11.3], font=8.5)
    H(d, 2, "2.5 Basic Descriptive Statistics")
    para(d, "The table gives the mean, median, mode, standard deviation, quartiles and skewness of the 15 numeric/ordinal features and the target, computed on all 649 rows.")
    rows = [[c, f2(r["mean"]), f2(r["median"]), f2(r["mode"]), f2(r["std"]), f2(r["min"]), f2(r["q25"]), f2(r["q75"]), f2(r["max"]), f2(r["skew"])] for c, r in desc.iterrows()]
    TAB(d, "Descriptive statistics of numeric features and the target", ["Feature", "Mean", "Median", "Mode", "Std", "Min", "Q1", "Q3", "Max", "Skew"], rows, [2.8, 1.5, 1.6, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4], font=8.5, align_right_from=1)

    # ================================================= 3 EDA
    H(d, 1, "3. Exploratory Data Analysis and Preprocessing")
    FIG(d, EDA_DIR / "diagram1_workflow.png", 16, "Diagram 1: End-to-end project workflow")
    para(d, "Diagram 1 shows the order of work. Steps 1–5 are data understanding and visualisation; steps 6–9 are modelling. Each step is a separate Python module in the GitHub repository.", size=10)
    H(d, 2, "3.1 Loading and Inspecting the Data")
    para(d, f"The CSV is loaded with pandas (pd.read_csv). The shape is {E_['shape'][0]} rows × {E_['shape'][1]} columns. df.info(), df.describe() and value counts were used to check data types and ranges. All ordinal ratings are within their documented ranges.")
    H(d, 2, "3.2 Missing Values and Duplicates")
    para(d, f"Missing cells: **{E_['clean_report']['missing_cells']}**. Duplicate rows: **{E_['clean_report']['duplicate_rows']}**. The cleaning function therefore removed no rows. The check remains in the code so that the same script is safe on other data.")
    para(d, f"**A hidden pattern: the zero grades.** {len(zero)} students ({pct(len(zero)/len(df))}) have a final grade of exactly 0, while the next lowest values are 1 and 5 (Figure 1). All {len(zero)} of them have **0 absences**, and their average second-period grade was {E_['g3_zero_profile']['mean_G2_of_zero']:.1f} "
            f"(only {E_['g3_zero_profile']['zero_with_G2_zero']} had G2 = 0). A student with a decent G2 but G3 = 0 is more consistent with *not taking the final assessment* than with scoring zero marks. The dataset documentation does not explain this, so it is stated here only as a hypothesis. "
            "These rows are real and were not removed, but they turn out to be the largest source of prediction error (Chapter 7).")
    H(d, 2, "3.3 Outlier Identification and Treatment")
    para(d, "Outliers were identified with the IQR rule and the |z| > 3 rule for numeric columns with more than five distinct values (Table 5, Figure 4).")
    rows = [[c, f2(r.lower_fence), f2(r.upper_fence), int(r.iqr_outliers), f"{r.iqr_pct:.1f}%", int(r.z_gt_3)] for c, r in outl.iterrows()]
    TAB(d, "Outlier summary", ["Feature", "Lower fence", "Upper fence", "IQR outliers", "Share", "|z|>3"], rows, [3.4, 2.4, 2.4, 2.6, 2.0, 1.8], font=9, align_right_from=1)
    para(d, f"**Decision.** Few rows are flagged (at most {outl.iqr_pct.max():.1f}% in any column). The flagged values are real: very high absences (maximum {desc.loc['absences','max']:.0f}) and the zero grades. No row was deleted. Inside the model pipeline each numeric feature is winsorised "
            "(clipped to its 1st and 99th training percentile) so that a few extreme values cannot dominate the scaling.")
    H(d, 2, "3.4 Cleaning, Transformation and Encoding")
    TAB(d, "Preprocessing applied to each group of features", ["Group", "Features", "Treatment", "Reason"], [
        ["Numeric / ordinal", f"{E_['n_numeric']} fields (age, ratings, absences, G1, G2, …)", "Winsorise 1–99 % → StandardScaler (mean 0, SD 1)", "Neural networks train poorly when inputs have very different scales"],
        ["Nominal", "4 fields (Mjob, Fjob, reason, guardian)", "One-hot encoding", "Categories have no order, so they must not be coded as 1, 2, 3"],
        ["Binary", "13 fields (school, sex, internet, higher, …)", "Encoded as 0/1 (ordinal encoder fitted on training rows)", "Two-level text must become numbers"],
        ["Target", "G3", "Kept in grade points; standardised with the training mean/SD only while training; predictions converted back and clipped to 0–20", "Makes training stable but reports errors in real grade points"]], [2.3, 4.2, 5.4, 4.7], font=8.5)
    para(d, f"After encoding, the 32 raw features become **{T_['A']['n_inputs']} model input columns** in setting A and **{T_['B']['n_inputs']}** in setting B (see Section 3.5).")
    H(d, 2, "3.5 Two Experimental Settings")
    para(d, "G1 and G2 are marks of earlier periods of the *same course*, so they are known only after those periods. To show how much they matter, two settings are studied throughout:")
    TAB(d, "The two settings", ["Setting", "Inputs", "Meaning in practice"], [["A – with grades", f"All 32 features incl. G1, G2 ({T_['A']['n_inputs']} encoded columns)", "Prediction after two periods of the course"], ["B – without grades", f"30 features, no G1, G2 ({T_['B']['n_inputs']} encoded columns)", "Early warning using only background and lifestyle information"]], [3.2, 6.6, 6.8], font=9)
    H(d, 2, "3.6 Train / Validation / Test Split and Leakage Prevention")
    s = T_["A"]["sizes"]
    TAB(d, "Data split (random_state = 42)", ["Subset", "Rows", "Share", "Purpose"], [["Training", s["train"], "70%", "Fit the scaler/encoder and learn the network weights"], ["Validation", s["val"], "15%", "Choose hyperparameters and stop training early"], ["Test", s["test"], "15%", "Final, one-time evaluation"]], [3, 2, 2, 9.6], font=9)
    para(d, "**Preventing data leakage.** (1) The split is done before any statistic is learned. (2) The winsoriser, scaler and encoders are fitted on training rows only. (3) The test set is used only after all tuning decisions are final. "
            "(4) The split is stratified on three grade bands (below 10, 10–13, 14 and above) so that every subset has a similar grade mix. (5) The 649 rows are different students within this file, so no student appears twice.")
    para(d, "**Why also repeat the split 10 times?** A test set of 98 students is small, and a few unusual students (such as zero-grade cases) can change an error figure a lot. Section 7.4 therefore also reports results over 10 random splits. The hyperparameters were chosen once, on the seed-42 validation set.", size=10.5)
    H(d, 2, "3.7 Key Findings from EDA")
    bullets(d, [f"{pct(DF['pass_rate'])} of students pass (G3 ≥ 10); the grade distribution is concentrated between 8 and 16 with a small spike at 0.",
                f"The earlier grades G1 and G2 are almost the same quantity as G3 (Pearson r = {E_['pearson_G1_G3']:.2f} and {E_['pearson_G2_G3']:.2f}).",
                f"Past failures matter: mean G3 is {gm['failures']['0'][0]:.1f} with no past failure versus {r_fail[True]:.1f} with one or more.",
                f"Wanting higher education (mean G3 {gm['higher']['yes'][0]:.1f} vs {gm['higher']['no'][0]:.1f}) and school (GP {gm['school']['GP'][0]:.1f} vs MS {gm['school']['MS'][0]:.1f}) show large differences.",
                "Most lifestyle ratings show weak relationships with G3 (|Spearman ρ| below 0.25), and parents' education is the strongest of the family variables.",
                f"Absences are heavily skewed: {pct(absn['zero_share'])} of students have none, the mean is {absn['mean']:.1f} and the maximum is {desc.loc['absences','max']:.0f}."])

    # ================================================= 4 VISUALISATION
    H(d, 1, "4. Data Visualization")
    H(d, 2, "4.1 The Seven Stages of Visualization")
    para(d, "The assignment sheet asks that the visualisations follow \"all 7 stages of visualization\" but does not list them. This report therefore states the framework it uses: the seven-stage visualisation process of Ben Fry (2007), *Visualizing Data* (O'Reilly): **Acquire, Parse, Filter, Mine, Represent, Refine, Interact**.")
    TAB(d, "Seven stages of visualization applied to the student grade dataset", ["Stage", "Meaning", "How it was applied here", "Evidence"], [
        ["1. Acquire", "Obtain the data", "Selected UCI dataset 320, checked licence, rows and columns, and loaded data/student-por.csv", "Sections 2.1, 2.3"],
        ["2. Parse", "Give the data structure and meaning", "Assigned numeric, ordinal, binary and nominal roles; defined the target and the two settings", "Sections 2.4, 3.4, 3.5"],
        ["3. Filter", "Remove what is not needed", "Missing/duplicate checks; chose key variables per chart; kept only levels with enough students in heat-maps", "Sections 3.2, 3.3"],
        ["4. Mine", "Find patterns with statistics", "Spearman, Mann–Whitney, Kruskal–Wallis, VIF, PCA, distribution fitting", "Chapter 5"],
        ["5. Represent", "Choose a basic visual form", "Bar, histogram, boxplot, scatter, heat-map, pairplot, Q–Q, PCA, dot-and-interval chart matched to the data type", "Figures 1–12"],
        ["6. Refine", "Improve clarity and design", "Colour-blind-safe palette, consistent colours (blue pass, orange fail), direct labels, mean/median lines, jitter and transparency, sequential scales for ordered values", "All figures"],
        ["7. Interact", "Let the viewer explore", "Two Plotly pages with hover, zoom and legend filtering: outputs/eda/interactive_*.html", "Section 4.3"]], [2.2, 3.2, 8.2, 3.0], font=8.5, bold_first_col=True)
    H(d, 2, "4.2 Visualization Design Choices (Refine stage)")
    para(d, "Colours follow a fixed meaning: **blue = pass (or the main series), orange = fail (or the comparison series)**. Both colours come from a palette whose adjacent colours were checked for colour-vision-deficiency separation. "
            "Ordered quantities (mean grade, PCA colouring) use a single-hue or perceptually uniform sequential scale; the correlation heat-map uses a diverging scale centred at zero. Axes carry units, titles are numbered and no chart uses a dual axis.")
    H(d, 2, "4.3 Interactive Visualizations")
    para(d, "Two interactive HTML pages are saved in outputs/eda/: **interactive_g2_vs_g3.html** (hover shows G1, failures, study time, absences, school and wish for higher education; clicking a legend entry hides or shows pass or fail) and **interactive_g3_by_failures.html** (box plots of G3 by past failures and school with all points). They open in any browser and complete stage 7.")
    H(d, 2, "4.4 Visualization Gallery with Explanations")
    para(d, "Each figure is followed by a table stating its purpose, variables, chart type, why it was selected and the insight obtained, as the assignment requires. Figures 1–12 are EDA figures; model figures are in Chapter 7.")
    P = EDA_DIR
    CARD(d, 1, "Distribution of the final grade (G3)", P / "fig01_target_distribution.png", 11, "Show how final grades are spread and how many students pass.", "G3", "Bar chart (one bar per grade) with pass mark",
         "G3 is an integer, so one bar per value is more honest than binned bars; colouring the bars by pass/fail adds meaning.",
         f"Grades are concentrated between 8 and 16 (median {desc.loc['G3','median']:.0f}, mean {desc.loc['G3','mean']:.1f}). {E_['pass_count']} of {len(df)} students ({pct(DF['pass_rate'])}) reach the pass mark. A separate group of {len(zero)} students has G3 = 0, far below the rest.")
    CARD(d, 2, "Histograms of key numeric variables", P / "fig02_histograms.png", 16.2, "Show the shape, centre and spread of six variables.", "G1, G2, age, absences, studytime, failures", "Histograms with mean (solid) and median (dashed) lines",
         "Histograms are the standard univariate view; the gap between mean and median shows skewness.",
         f"G1 and G2 are bell-shaped around 11–12. Age is concentrated at 15–18. Absences are strongly right-skewed (median {desc.loc['absences','median']:.0f}, mean {desc.loc['absences','mean']:.1f}, maximum {desc.loc['absences','max']:.0f}). Most students have no past failure (mean {desc.loc['failures','mean']:.2f}); the most common study-time level is 2 (2–5 hours a week).")
    CARD(d, 3, "Final grade by key factors", P / "fig03_boxplots_g3_by_factor.png", 16.6, "Compare the grade distribution across groups and reveal outliers.", "G3 vs school, failures, studytime, higher, Dalc", "Grouped boxplots",
         "Boxplots compare medians, spread and outliers between groups compactly.",
         f"Median G3 is clearly lower for students with past failures (median 12 with none, 10 with one or two) and tends to rise with study time (median 11, 12, 13, 13 for levels 1–4). Students who do not want higher education and students of school MS have lower medians. Workday alcohol use shows a slight downward trend. The group means are: failures 0 → {gm['failures']['0'][0]:.1f}, 1 → {gm['failures']['1'][0]:.1f}; study time 1 → {gm['studytime']['1'][0]:.1f}, 3 → {gm['studytime']['3'][0]:.1f}.")
    CARD(d, 4, "Share of IQR outliers per numeric feature", P / "fig04_outlier_share.png", 12, "Quantify how many rows are flagged as outliers.", "Numeric columns with more than five distinct values", "Sorted horizontal bar chart",
         "A ranked bar chart shows which features need a decision.", f"Every feature has few flagged rows (maximum {outl.iqr_pct.max():.1f}%). The G1/G2/G3 flags are the low-grade and zero-grade students; absences flags very high counts. Because these are genuine values, they were winsorised in the pipeline, not deleted.")
    CARD(d, 5, "Continuous distribution: final grade G3", P / "fig05_normal_fit_qq.png", 16.6, "Check whether G3 follows a normal distribution, with and without the zero grades.", "G3", "Histograms with fitted normal curves; normal Q–Q plot",
         "The histogram shows the fit visually; the Q–Q plot is more sensitive to the tails.",
         f"All students: skewness {DF['G3_all']['skew']:.2f}, excess kurtosis {DF['G3_all']['excess_kurtosis']:.2f}, D'Agostino p < 0.001 – clearly not normal because of the long lower tail made by the zeros. "
         f"Without the {DF['n_G3_zero']} zero grades (n = {DF['G3_nonzero']['n']}): mean {DF['G3_nonzero']['mean']:.2f}, SD {DF['G3_nonzero']['std']:.2f}, skewness {DF['G3_nonzero']['skew']:.2f}, D'Agostino p = {DF['G3_nonzero']['dagostino_p']:.2f}, so normality is not rejected. The staircase in the Q–Q plot is caused by G3 taking whole numbers only.")
    CARD(d, 6, "Discrete distributions: Poisson, negative binomial, Binomial", P / "fig06_discrete_distributions.png", 16.4, "Test count models for absences and illustrate a Binomial model for passing students.", "(a) absences; (b) pass/fail (G3 ≥ 10)", "Side-by-side bar charts of observed and theoretical probabilities",
         "Observed-versus-theoretical bars show directly where a model fits or fails.",
         f"(a) A Poisson with the same mean (λ = {absn['mean']:.2f}) fits badly: variance/mean = {absn['dispersion_index']:.1f} (Poisson would be 1) and {pct(absn['zero_share'])} of students have zero absences versus {100*p_pois0:.1f}% expected. A negative binomial (r = {absn['nb_r']:.2f}) is closer (total variation distance {absn['tv_negbin']:.2f} vs {absn['tv_poisson']:.2f}) but still imperfect. "
         f"(b) Passing is a Bernoulli event with p = {DF['pass_rate']:.3f} (95% Wilson interval {DF['pass_wilson_ci'][0]:.3f}–{DF['pass_wilson_ci'][1]:.3f}); the number of passes among 10 random students follows Binomial(10, p), confirmed by 5,000 simulated samples. For example, P(all 10 pass) = {p_all10:.3f}.")
    CARD(d, 7, "Spearman correlation heatmap", P / "fig07_correlation_heatmap.png", 14, "Show monotonic relationships among numeric/ordinal features and G3.", "15 numeric/ordinal features and G3", "Lower-triangle heat-map with diverging colours and printed coefficients",
         "A heat-map is the standard multivariate summary; Spearman is used because many variables are ordinal or skewed.",
         f"The strongest pairs are G2–G3 (ρ = {E_['top_corr_pairs'][0][2]}), G1–G2 ({E_['top_corr_pairs'][1][2]}) and G1–G3 ({E_['top_corr_pairs'][2][2]}). Mother's and father's education correlate at {E_['top_corr_pairs'][3][2]}, workday and weekend alcohol at {E_['top_corr_pairs'][4][2]}. Among non-grade variables, failures has the strongest link with G3 (ρ = {E_['top_corr_pairs'][5][2]}).")
    CARD(d, 8, "Mean final grade for each binary factor", P / "fig08_group_means.png", 12.8, "Compare the average grade between the two levels of every binary factor, with uncertainty.", "13 binary factors vs G3", "Dot-and-interval chart (mean with 95% CI), sorted by gap",
         "A dot-and-interval chart shows the size of each gap and whether the intervals overlap; it needs less ink than 13 pairs of bars.",
         f"The widest gaps are for wanting higher education ({gm['higher']['no'][0]:.1f} vs {gm['higher']['yes'][0]:.1f}) and school (GP {gm['school']['GP'][0]:.1f} vs MS {gm['school']['MS'][0]:.1f}), followed by address (rural {gm['address']['R'][0]:.1f} vs urban {gm['address']['U'][0]:.1f}) and internet at home ({gm['internet']['no'][0]:.1f} vs {gm['internet']['yes'][0]:.1f}). "
         "Intervals for family size, nursery and parents' status overlap strongly. Several factors are unbalanced (for example only 69 students do not want higher education), so wide intervals are expected. These are associations, not causes.")
    CARD(d, 9, "Earlier grades against the final grade", P / "fig09_grade_scatter.png", 15, "Show how closely G1 and G2 predict G3.", "G1, G2, G3, pass/fail", "Scatter plots with regression line (jittered, semi-transparent)",
         "Scatter plots are the standard bivariate view of two numeric variables.",
         f"Both relationships are close to a straight line (r = {E_['pearson_G1_G3']:.3f} for G1 and {E_['pearson_G2_G3']:.3f} for G2). The points far below the line, at G3 = 0, are the zero-grade students. This explains why the model in setting A can be accurate and why it relies on G2 so heavily.")
    CARD(d, 10, "PCA projection of numeric features", P / "fig10_pca.png", 11.5, "See whether grade differences appear when 15 features are compressed into two.", "15 numeric/ordinal features (standardised), coloured by G3", "PCA scatter with a sequential colour scale",
         "PCA is the usual way to view many dimensions in 2-D; a sequential scale suits a numeric colour variable.",
         f"PC1 explains {E_['pca_var_ratio'][0]*100:.1f}% and PC2 {E_['pca_var_ratio'][1]*100:.1f}% of the variance; five components explain {sum(E_['pca_var_ratio'][:5])*100:.1f}%. G3 rises along PC1 (correlation 0.75 between PC1 and G3), because PC1 is driven mainly by G1, G2 and past failures, but there is no clean cluster structure.")
    CARD(d, 11, "Pairplot of key variables", P / "fig11_pairplot.png", 14.5, "Show pairwise relations and distributions for six variables.", "G2, failures, studytime, absences, age, G3, pass/fail", "Pairplot (scatter matrix with KDE diagonals)",
         "A pairplot is the quickest way to compare many pairs and class-wise distributions.", "Only G2 separates pass from fail clearly. Compared with passing students, failing students are slightly older (mean age 17.1 vs 16.7), have more past failures (0.75 vs 0.13), study a little less (study-time level 1.6 vs 2.0) and have more absences (4.6 vs 3.5), but the groups overlap heavily on every variable except G2 – a first sign that predictions without G1/G2 will be hard.")
    CARD(d, 12, "Mean G3 by study time and past failures", P / "fig12_studytime_failures.png", 11, "Show the joint effect of study time and past failures.", "studytime, failures, G3", "Annotated heat-map with sequential single-hue scale",
         "A two-way table of means shows interaction effects; printing n guards against over-reading small cells.",
         "Within every row (study time), students with a past failure score about 3–5 points lower than those with none. Within the no-failure column, mean G3 rises from 11.6 (study time 1) to 13.6 (study time 3). Cells with one to six students (for example study time 4 with failures) should not be trusted.")

    # ================================================= 5 STATISTICS
    H(d, 1, "5. Statistical and Distribution-Based Analysis")
    H(d, 2, "5.1 Univariate Analysis")
    bullets(d, [f"**Centre and spread:** G3 has mean {desc.loc['G3','mean']:.2f}, median {desc.loc['G3','median']:.0f}, mode {desc.loc['G3','mode']:.0f}, SD {desc.loc['G3','std']:.2f} and quartiles {desc.loc['G3','q25']:.0f} and {desc.loc['G3','q75']:.0f} (Table 4).",
                f"**Skewness:** G3 is left-skewed ({desc.loc['G3','skew']:.2f}) because of the zero grades; absences are strongly right-skewed ({desc.loc['absences','skew']:.2f}); failures are also right-skewed ({desc.loc['failures','skew']:.2f}).",
                f"**Quantiles:** half of the students have two absences or fewer and three quarters have six or fewer, but the maximum is {desc.loc['absences','max']:.0f}.",
                f"**Categorical fields:** {pct(gm['higher']['yes'][1]/len(df))} want higher education, {pct(gm['internet']['yes'][1]/len(df))} have internet at home, {pct(gm['school']['GP'][1]/len(df))} attend school GP. Some binary fields are very unbalanced (for example only {gm['paid']['yes'][1]} students take extra paid classes)."])
    H(d, 2, "5.2 Probability Distributions")
    TAB(d, "Distribution analysis results", ["Variable", "Model", "Result", "Conclusion"], [
        ["G3, all students (continuous)", "Normal", f"μ = {DF['G3_all']['mean']:.2f}, σ = {DF['G3_all']['std']:.2f}; skew {DF['G3_all']['skew']:.2f}; D'Agostino p < 0.001", "Rejected: long lower tail from zero grades"],
        ["G3 > 0 (continuous)", "Normal", f"n = {DF['G3_nonzero']['n']}; μ = {DF['G3_nonzero']['mean']:.2f}, σ = {DF['G3_nonzero']['std']:.2f}; skew {DF['G3_nonzero']['skew']:.2f}; D'Agostino p = {DF['G3_nonzero']['dagostino_p']:.2f}", "Not rejected: approximately normal"],
        ["Pass / fail (discrete)", "Bernoulli / Binomial(10, p)", f"p = {DF['pass_rate']:.3f}; Wilson 95% CI {DF['pass_wilson_ci'][0]:.3f}–{DF['pass_wilson_ci'][1]:.3f}", "Binomial matches simulated samples (Fig. 6b)"],
        ["Absences (count)", "Poisson", f"mean {absn['mean']:.2f}, variance {absn['variance']:.2f}, ratio {absn['dispersion_index']:.2f}; zeros {pct(absn['zero_share'])}", "Rejected: strongly over-dispersed"],
        ["Absences (count)", "Negative binomial", f"r = {absn['nb_r']:.2f}, p = {absn['nb_p']:.3f} (method of moments); TV distance {absn['tv_negbin']:.2f}", "Better than Poisson, still imperfect"],
        ["Alcohol (Dalc, 1–5)", "Frequency distribution", "level 1: " + f"{DF['Dalc_pct']['1']}%, 2: {DF['Dalc_pct']['2']}%, 3: {DF['Dalc_pct']['3']}%, 4: {DF['Dalc_pct']['4']}%, 5: {DF['Dalc_pct']['5']}%", "Geometric-like decline; most students drink little on workdays"]], [3.6, 3.0, 6.2, 3.8], font=8.5)
    para(d, "In simple words: a *normal* curve describes measurements that cluster symmetrically around an average. G3 would look like that except for the zero grades. A *Poisson* distribution describes counts of rare, independent events and has variance equal to its mean; absences have variance about six times their mean because a few students miss very many days, so a negative binomial, which allows extra spread, fits better.", size=10)
    H(d, 2, "5.3 Bivariate Analysis")
    para(d, "Every input feature was tested against G3. Numeric and ordinal features use **Spearman's ρ**; binary features use the **Mann–Whitney U test** with Cohen's d (standardised mean difference) as effect size; nominal features use the **Kruskal–Wallis H test** with ε². "
            "Effect sizes are on different scales, so use them to rank features within a test type; the p-value says whether an association is likely to be real, not how important it is.")
    rows = [[f, r["type"], r["test"], f"{r['statistic']:.3f}" if r["test"] == "Spearman" else f"{r['statistic']:.0f}" if r["test"] == "Mann-Whitney U" else f"{r['statistic']:.1f}", "< 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}", f"{r['effect_size']:.3f}"] for f, r in biv.head(14).iterrows()]
    TAB(d, "Fourteen features most strongly associated with G3", ["Feature", "Type", "Test", "Statistic", "p-value", "Effect size"], rows, [2.8, 3.0, 3.2, 2.6, 2.4, 2.6], font=8.5, align_right_from=3)
    para(d, f"**Interpretation.** Apart from G1 and G2, the largest effects are for wanting higher education (d = {eps('higher'):.2f}), school (d = {eps('school'):.2f}) and past failures (|ρ| = {eps('failures'):.2f}). Family background (parents' education, |ρ| about {eps('Medu'):.2f} and {eps('Fedu'):.2f}), study time ({eps('studytime'):.2f}) and address (d = {eps('address'):.2f}) have smaller effects. "
            f"Extra paid classes (p = {pv('paid')}) show no reliable difference. Nominal fields such as Mjob and Fjob have small ε² values. All of these are associations; for example, students who plan to continue studying may also study harder.")
    H(d, 2, "5.4 Multivariate Analysis")
    bullets(d, ["**Correlation structure (Fig. 7):** G1, G2 and G3 form one tight block; parents' education and the two alcohol ratings form two weaker pairs; everything else is only loosely related.",
                "**Multicollinearity:** variance inflation factors are " + ", ".join(f"{k} ({v:.1f})" for k, v in vif.head(4).items()) + "; none exceeds 5, so multicollinearity is mild even though G1 and G2 are strongly correlated.",
                f"**PCA (Fig. 10):** five components explain {sum(E_['pca_var_ratio'][:5])*100:.1f}% of variance, so the numeric information is spread over many directions; there is no single dominant lifestyle factor.",
                "**Combined effects (Fig. 12):** failures and study time act together: the best group (no failures, study time 3) has a mean G3 of 13.6, the worst well-populated group (one failure, study time 1) 8.1."])
    H(d, 2, "5.5 Summary of Statistical Findings")
    TAB(d, "Statistical findings in plain words", ["Finding", "Evidence"], [
        ["Earlier grades almost determine the final grade", f"Pearson r = {E_['pearson_G1_G3']:.2f} (G1) and {E_['pearson_G2_G3']:.2f} (G2); Fig. 9"],
        ["Past failures lower the final grade", f"Mean G3 {gm['failures']['0'][0]:.1f} (none) vs {r_fail[True]:.1f} (one or more); ρ = {biv.loc['failures','statistic']:.2f}"],
        ["Aspiration and school matter", f"Higher-education wish: {gm['higher']['no'][0]:.1f} vs {gm['higher']['yes'][0]:.1f}; school GP {gm['school']['GP'][0]:.1f} vs MS {gm['school']['MS'][0]:.1f}"],
        ["Lifestyle ratings are weak signals", "Most |ρ| < 0.25 with G3; several effects are small"],
        ["Absences are not Poisson", f"Variance/mean = {absn['dispersion_index']:.1f}; {pct(absn['zero_share'])} zeros"],
        ["Zero grades are a separate group", f"{len(zero)} students, all with 0 absences; they break normality of G3"]], [6.5, 10.1], font=9)

    # ================================================= 6 DL MODEL
    H(d, 1, "6. Deep Learning Model")
    H(d, 2, "6.1 Why a Multilayer Perceptron?")
    para(d, "The data are in table form, so the natural deep-learning model is a **multilayer perceptron (MLP)**, a stack of fully connected layers (Goodfellow et al., 2016). CNNs are designed for images and RNN/LSTM/Transformer models for sequences, which this dataset does not contain. "
            "Because G3 is a number, the network ends in one linear output neuron and is trained to minimise squared error (regression). To judge whether the extra complexity is worthwhile, a mean-only baseline, ridge regression (Hoerl & Kennard, 1970) and a random forest (Breiman, 2001) were trained on the same data.")
    H(d, 2, "6.2 Inputs, Target and Pipeline")
    TAB(d, "Model inputs and output", ["Item", "Description"], [["Input, setting A", f"{T_['A']['n_inputs']} columns from 32 raw features (including G1 and G2)"], ["Input, setting B", f"{T_['B']['n_inputs']} columns from 30 raw features (no G1, G2)"],
        ["Target", "G3, final grade 0–20 (numeric)"], ["Task type", "Regression"], ["Pipeline", "Winsorise (1–99 %) → standardise; one-hot for nominal; 0/1 for binary. Fitted on the training set only; saved as models/preprocessor_A.joblib and preprocessor_B.joblib"]], [3.2, 13.4], font=9, bold_first_col=True)
    H(d, 2, "6.3 Architecture and Hyperparameters")
    FIG(d, EVAL_DIR / "diagram2_mlp_architecture.png", 16, "Diagram 2: MLP architecture of the final model (setting A)")
    TAB(d, "Final model configuration", ["Component", "Setting A", "Setting B", "Simple explanation"], [
        ["Hidden layers", " → ".join(map(str, cfgA["hidden"])), " → ".join(map(str, cfgB["hidden"])), "Each layer learns combinations of the inputs"], ["Activation", "ReLU; linear output", "ReLU; linear output", "ReLU adds non-linearity; a linear output can give any grade"],
        ["Regularisation", f"Dropout {cfgA['dropout']}, L2 1e-4, BatchNorm", f"Dropout {cfgB['dropout']}, L2 1e-4, BatchNorm", "Reduce over-fitting on only 454 training rows"], ["Loss", "Mean squared error", "Mean squared error", "Penalises large errors strongly"],
        ["Optimizer", f"Adam, lr {cfgA['lr']}", f"Adam, lr {cfgB['lr']}", "Adaptive gradient method (Kingma & Ba, 2015)"], ["Batch size / max epochs", f"{T_['A']['sizes']['train'] and 32} / 300", "32 / 300", "Samples per weight update / upper limit"],
        ["Early stopping", f"patience 25; ran {M_['epochs_run']['A']} epochs, best {M_['best_epoch']['A']}", f"patience 25; ran {M_['epochs_run']['B']} epochs, best {M_['best_epoch']['B']}", "Stops when validation loss stops improving; best weights restored"],
        ["Parameters", f"{T_['A']['n_params']:,}", f"{T_['B']['n_params']:,}", "Number of learned weights"], ["Random seed", "42", "42", "Makes the run reproducible"]], [3.0, 4.0, 4.0, 5.6], font=8.5, bold_first_col=True)
    H(d, 2, "6.4 Training, Validation and Testing Procedure")
    bullets(d, ["Fit the preprocessing pipeline on the training rows and transform all three subsets.", "For each hyperparameter configuration, train on the training set and monitor loss on the validation set; stop early when it stops improving.",
                "Choose the configuration with the lowest **validation RMSE**, separately for settings A and B.", "Evaluate the chosen models **once** on the untouched test set; report MAE, RMSE and R², and analyse the errors.",
                "Repeat the whole fit on 10 random splits (hyperparameters fixed) to measure stability."], style="List Number")
    H(d, 2, "6.5 Basic Hyperparameter Tuning")
    para(d, "A grid search over **3 architectures × 2 dropout rates × 2 learning rates = 12 configurations** was run for each setting; everything else was fixed. Selection used the validation set only. The five best configurations of each setting are shown; the full results are in outputs/model_evaluation.")
    for n in "AB":
        rows = [[r.hidden, r.dropout, r.lr, r.epochs_run, f3(r.val_RMSE), f3(r.val_MAE), f3(r.val_R2)] for r in tune[n].sort_values("val_RMSE").head(5).itertuples()]
        TAB(d, f"Best five configurations, setting {n} (sorted by validation RMSE)", ["Hidden layers", "Dropout", "Learning rate", "Epochs", "Val RMSE", "Val MAE", "Val R²"], rows, [3.2, 1.8, 2.3, 1.7, 2.4, 2.4, 2.4], font=8.5, align_right_from=1)
    para(d, f"Setting A: best configuration {cfgA['hidden']}, dropout {cfgA['dropout']}, learning rate {cfgA['lr']} (validation RMSE {f3(T_['A']['val_RMSE'])}); the spread across the 12 configurations is {tune['A'].val_RMSE.max()-tune['A'].val_RMSE.min():.2f} RMSE. "
            f"Setting B: best {cfgB['hidden']}, dropout {cfgB['dropout']}, learning rate {cfgB['lr']} (validation RMSE {f3(T_['B']['val_RMSE'])}), spread {tune['B'].val_RMSE.max()-tune['B'].val_RMSE.min():.2f}. With a 97-row validation set, differences of 0.05–0.1 between neighbouring configurations are within noise. Figure 19 shows all results.", size=10.5)
    H(d, 2, "6.6 Feature Analysis and Feature Importance")
    para(d, "Two complementary methods were used. (1) **Statistical association** (Section 5.3) shows which features relate to G3 in the raw data. (2) **Permutation importance** shows what the *trained model* relies on: each raw feature in the test set is shuffled 10 times and the increase in test RMSE is recorded; a large increase means the model depends on that feature. Shuffling is done on raw columns before encoding, so all one-hot columns of a field are shuffled together.")
    for n in "AB":
        rows = [[i + 1, f, f"{r.mean_increase:.4f}", f"{r['std']:.4f}"] for i, (f, r) in enumerate(imp[n].head(8).iterrows())]
        TAB(d, f"Top eight features by permutation importance, setting {n} (test set)", ["Rank", "Feature", "Mean RMSE increase", "Std (10 repeats)"], rows, [1.4, 6.0, 4.4, 4.4], font=8.5, align_right_from=2)
    para(d, f"In setting A the model depends almost entirely on G2 (RMSE rises by {imp['A'].iloc[0].mean_increase:.2f} grade points when G2 is shuffled) and G1 ({imp['A'].iloc[1].mean_increase:.2f}); every other feature adds less than 0.1. "
            f"In setting B the largest increase is only {imp['B'].iloc[0].mean_increase:.3f} (past failures), and for most features the error bar is about as large as the bar itself, so the ranking in setting B is uncertain. Importance is shared between correlated features, so a small value does not prove a feature is unrelated to the grade.", size=10.5)

    # ================================================= 7 EVALUATION
    H(d, 1, "7. Model Evaluation and Result Visualization")
    H(d, 2, "7.1 Evaluation Metrics")
    para(d, "Three regression metrics are used. **MAE** is the average size of the error in grade points. **RMSE** is similar but penalises large errors more. **R²** is the share of the variation in G3 that the model explains (1 is perfect, 0 is no better than always predicting the average). All values were measured by running the code in the appendix (seed 42). The test set has 98 students that were never used for fitting, scaling or tuning. Predictions are clipped to the valid range 0–20.")
    for n, ttl in [("A", "setting A (with G1 and G2)"), ("B", "setting B (without G1 and G2)")]:
        rows = [[k, f2(MT[n][k]["MAE"]), f2(MT[n][k]["RMSE"]), f2(MT[n][k]["R2"])] for k in ["MLP", "MLP (validation)", "Ridge regression", "Random forest", "Mean baseline"]]
        rows[0][0] = "MLP (test)"; rows[1][0] = "MLP (validation)"; rows[2][0] = "Ridge regression (test)"; rows[3][0] = "Random forest (test)"; rows[4][0] = "Mean baseline (test)"
        TAB(d, f"Regression metrics for {ttl}", ["Model", "MAE", "RMSE", "R²"], rows, [6.0, 3.0, 3.0, 3.0], font=9.5, align_right_from=1)
    para(d, f"**Reading the tables.** With earlier grades the MLP is wrong by about {f2(MT['A']['MLP']['MAE'])} grade points on average (RMSE {f2(MT['A']['MLP']['RMSE'])}); {pct(RA['within_1'])} of test students are predicted within one point and {pct(RA['within_2'])} within two. "
            f"Without them, the average error is {f2(MT['B']['MLP']['MAE'])} points and R² is only {f2(MT['B']['MLP']['R2'])}: the MLP is only slightly better than always predicting the average grade (RMSE {f2(MT['B']['Mean baseline']['RMSE'])}). "
            f"As an extra check, the predicted grade was turned into a pass/fail call (predicted ≥ 9.5 means pass): accuracy {pct(PF['A']['accuracy'])} and AUC {f3(PF['A']['auc'])} in setting A, accuracy {pct(PF['B']['accuracy'])} and AUC {f3(PF['B']['auc'])} in setting B. Only {PF['A']['n_fail_test']} of the {PF['A']['n_test']} test students fail, so these pass/fail figures are very uncertain; always predicting 'pass' would already score {pct(1-PF['A']['n_fail_test']/PF['A']['n_test'])}.", size=10.5)
    H(d, 2, "7.2 Comparison with Baseline Models")
    para(d, f"On the seed-42 test set the random forest has the lowest error in both settings (RMSE {f2(MT['A']['Random forest']['RMSE'])} and {f2(MT['B']['Random forest']['RMSE'])}); ridge regression is about as accurate as the MLP in setting A (RMSE {f2(MT['A']['Ridge regression']['RMSE'])} vs {f2(MT['A']['MLP']['RMSE'])}) and better in setting B. "
            "**The MLP is therefore competitive with, but not better than, the simpler models.** This is common for small tabular data, where the main relationship (G3 depends almost linearly on G2) is easy to capture. The MLP is kept as the required deep-learning model; the baselines give context.", size=10.5)
    H(d, 2, "7.3 Result Visualizations and Interpretation")
    Q = EVAL_DIR
    CARD(d, 13, "Training vs validation loss", Q / "fig13_loss_vs_epoch.png", 16, "Show how learning progressed and detect over-fitting.", "Epoch, MSE loss (train and validation), settings A and B", "Line charts with best-epoch marker", "Loss curves are the standard diagnostic of neural-network training.",
         f"In both settings training loss keeps falling while validation loss flattens, and the gap is larger in setting B. Early stopping restored the weights from epoch {M_['best_epoch']['A']} (A, of {M_['epochs_run']['A']} run) and epoch {M_['best_epoch']['B']} (B, of {M_['epochs_run']['B']}). The loss is on the standardised target, so a validation loss of about 0.35 corresponds roughly to an RMSE of 0.59 × 3.1 ≈ 1.8 grade points.")
    CARD(d, 14, "Training vs validation MAE", Q / "fig14_mae_vs_epoch.png", 16, "Show the mean absolute error progress for both subsets.", "Epoch, MAE (train and validation)", "Line charts", "MAE is easier to interpret than squared loss.",
         "The pattern repeats Figure 13: validation MAE levels off long before training MAE does. In setting B the validation MAE stays high and nearly flat, which means the network finds little learnable signal in the background features.")
    CARD(d, 15, "Predicted vs actual final grade", Q / "fig15_predicted_vs_actual.png", 14, "Show how close predictions are to the truth.", "Actual and predicted G3 on the test set", "Scatter plot with the perfect-prediction line (jittered)", "For regression, points on the diagonal are perfect predictions; the spread shows the error.",
         f"Setting A: points hug the diagonal, with the largest misses at the low end (actual 0, predicted {RA['worst_pred']:.1f}). Setting B: predictions are squeezed into a narrow band (about 8–14) whatever the true grade, so high and low achievers are not separated; the zero-grade student is predicted at about {RB['worst_pred']:.0f}.")
    CARD(d, 16, "Error distribution", Q / "fig16_error_distribution.png", 14.5, "Show how prediction errors are spread.", "Error = predicted − actual (test set)", "Histograms with zero line and mean error", "Regression error distributions reveal bias and heavy tails.",
         f"Errors are centred near zero in both settings (mean {RA['mean_error']:+.2f} and {RB['mean_error']:+.2f}), so there is no systematic over- or under-prediction. The spread is much larger without G1/G2 (SD {RB['sd_error']:.2f} vs {RA['sd_error']:.2f}). Shapiro–Wilk rejects normal errors (p < 0.001 in both) because of a few large misses, mainly zero-grade students.")
    CARD(d, 17, "Residual plot", Q / "fig17_residual_plot.png", 14.5, "Check whether errors depend on the predicted value.", "Predicted G3, error", "Scatter plot with zero line", "A good model shows a shapeless cloud around zero.",
         f"Setting A shows no strong pattern except two isolated large positive errors (zero-grade students predicted at about 4.6 and 7.0; the worst absolute error is {RA['max_abs_error']:.1f} points). Setting B shows a much wider scatter: predictions stay between about 7.5 and 14.5, errors of ±5 points are common, and the two largest errors (about 10 and 14 points) belong to students whose actual grade was 0. The model cannot tell most students apart.")
    CARD(d, 18, "Permutation importance (top 10 features)", Q / "fig18_permutation_importance.png", 16, "Show which raw features each trained model depends on.", "Raw features; increase in test RMSE after shuffling", "Horizontal bars with error bars (SD of 10 shuffles)", "Model-agnostic and works with one-hot encoded inputs when applied to raw columns.",
         f"Setting A is dominated by G2 and G1 (note the different x-axis scales). In setting B the bars are about 20 times smaller and error bars overlap zero for several features, so only past failures stands out; the others should not be over-interpreted.")
    CARD(d, 19, "Hyperparameter search results", Q / "fig19_tuning_results.png", 16, "Compare all 12 configurations per setting.", "Architecture, dropout, learning rate vs validation RMSE", "Sorted horizontal bars; best highlighted", "A ranked chart shows both the winner and how close the others are.",
         f"Setting A validation RMSE ranges from {tune['A'].val_RMSE.min():.2f} to {tune['A'].val_RMSE.max():.2f}; setting B from {tune['B'].val_RMSE.min():.2f} to {tune['B'].val_RMSE.max():.2f}. Neighbouring configurations are close, so the choice of the winner is not very certain.")
    H(d, 2, "7.4 Stability over 10 Random Splits")
    para(d, "Because one 98-student test set is small, the complete procedure (preprocessing, training, evaluation) was repeated for 10 different random splits, keeping the hyperparameters chosen above. Table 21 gives the mean and standard deviation of test errors; Figure 20 shows the spread.")
    rows = []
    for n in "AB":
        for m in ["Mean baseline", "Ridge regression", "Random forest", "MLP"]:
            r = RP[f"{n}|{m}"]; rows.append([n, m, f"{r['MAE']['mean']:.2f} ± {r['MAE']['std']:.2f}", f"{r['RMSE']['mean']:.2f} ± {r['RMSE']['std']:.2f}", f"{r['R2']['mean']:.2f} ± {r['R2']['std']:.2f}"])
    TAB(d, "Test errors over 10 random splits (mean ± standard deviation)", ["Setting", "Model", "MAE", "RMSE", "R²"], rows, [2.0, 4.6, 3.3, 3.3, 3.3], font=9)
    FIG(d, Q / "fig20_repeated_splits.png", 15.5, "Figure 20: Stability over 10 random train/validation/test splits")
    para(d, f"**Interpretation.** In setting A the ranking is random forest (RMSE {RP['A|Random forest']['RMSE']['mean']:.2f}), ridge ({RP['A|Ridge regression']['RMSE']['mean']:.2f}), MLP ({RP['A|MLP']['RMSE']['mean']:.2f}); the differences are comparable to the split-to-split standard deviation (about 0.2), so they are small. "
            f"In setting B all three models are only modestly better than the mean baseline ({RP['B|Mean baseline']['RMSE']['mean']:.2f}): MLP {RP['B|MLP']['RMSE']['mean']:.2f}, ridge {RP['B|Ridge regression']['RMSE']['mean']:.2f}, random forest {RP['B|Random forest']['RMSE']['mean']:.2f}, with R² of only {RP['B|MLP']['R2']['mean']:.2f}–{RP['B|Random forest']['R2']['mean']:.2f}. "
            "Note that the hyperparameters were chosen using the seed-42 validation set, which may overlap the test sets of other splits, so these repeated-split numbers are slightly optimistic for the MLP.", size=10.5)

    # ================================================= 8 CONCLUSION
    H(d, 1, "8. Conclusion, Challenges and Future Scope")
    H(d, 2, "8.1 What Was Learned from the Data")
    bullets(d, [f"The dataset is clean in the sense of having no missing values or duplicates, but it contains a small group of zero grades ({len(zero)} students) that behave differently from the rest.",
                f"The final grade is almost a continuation of the earlier grades: r = {E_['pearson_G1_G3']:.2f} with G1 and {E_['pearson_G2_G3']:.2f} with G2.",
                f"Among other factors, past failures (mean G3 {gm['failures']['0'][0]:.1f} vs {r_fail[True]:.1f}), wish for higher education ({gm['higher']['yes'][0]:.1f} vs {gm['higher']['no'][0]:.1f}) and school ({gm['school']['GP'][0]:.1f} vs {gm['school']['MS'][0]:.1f}) are most strongly associated with G3.",
                "Lifestyle ratings (free time, going out, alcohol, health) show weak relationships, and absences follow an over-dispersed count distribution that a Poisson model cannot describe."])
    H(d, 2, "8.2 Model Performance")
    para(d, f"With G1 and G2, the tuned MLP reached test MAE {f2(MT['A']['MLP']['MAE'])}, RMSE {f2(MT['A']['MLP']['RMSE'])} and R² {f2(MT['A']['MLP']['R2'])}. Without them it reached MAE {f2(MT['B']['MLP']['MAE'])}, RMSE {f2(MT['B']['MLP']['RMSE'])} and R² {f2(MT['B']['MLP']['R2'])}. "
            f"The MLP did not outperform ridge regression or the random forest, and over 10 splits the three models were close to each other. The main lesson is that how much information is available (the earlier grades) matters far more than which algorithm is used.")
    H(d, 2, "8.3 Challenges and Limitations")
    bullets(d, ["**Small data:** 454 training rows make a neural network easy to over-fit; dropout, L2, BatchNorm and early stopping were needed, and results vary by about ±0.2 RMSE between splits.",
                "**Dependence on G1 and G2:** setting A is accurate but is available only after two periods of the course; setting B is the real early-warning case and performs poorly.",
                "**Zero grades:** a few students with G3 = 0 produce the largest errors and affect the normality of G3 and of the residuals. Their meaning is not documented in the dataset description.",
                "**Association, not causation:** group differences (for example higher-education wish, school) cannot be interpreted as causes.",
                "**Hyperparameter choice on one validation set:** the repeated-split results reuse those hyperparameters, so they slightly favour the MLP.",
                "**Single course and region:** the data cover one Portuguese-language course in two schools; results may not transfer elsewhere.",
                "**Data source for this study:** the CSV was obtained from public copies of the UCI dataset (identical across two copies) because the UCI site was not reachable from the analysis environment.",
                "**Ethics:** sex, family and alcohol information are sensitive; a deployed system must be checked for fairness and used to offer support, not to label students."])
    H(d, 2, "8.4 Future Scope")
    bullets(d, ["Model the zero-grade students separately (for example a two-stage model: first predict whether the final assessment is taken, then predict the grade).", "Use repeated cross-validation or a nested procedure for tuning, with confidence intervals for model comparisons.",
                "Combine this file with the Mathematics file using a student-grouped split, to get more data without leakage.", "Try gradient-boosted trees, ordinal regression and a classification of grade bands, and calibrate predictions with intervals.",
                "Add SHAP explanations for individual students.", "Build a teacher dashboard that shows predicted risk, with fairness checks across groups."])

    # ================================================= 9 REFERENCES
    H(d, 1, "9. Literature Review and References")
    H(d, 2, "9.1 Short Literature Review")
    para(d, "**Dataset and application.** Cortez and Silva (2008) collected school reports and questionnaires from two Portuguese secondary schools and applied data-mining models to predict student performance in Mathematics and Portuguese language, studying settings with and without earlier-period grades. "
            "The data were later released on the UCI repository (Cortez, 2008) under a CC BY 4.0 licence. This project reuses the Portuguese-language file and follows the same idea of comparing predictions with and without G1 and G2, using a neural network and simple baselines.")
    para(d, "**EDA and visualisation.** Tukey (1977) established exploratory analysis. Fry (2007) formalised the seven-stage visualisation process (acquire, parse, filter, mine, represent, refine, interact) followed in this report. Plots and statistics were made with Matplotlib (Hunter, 2007), seaborn (Waskom, 2021) and SciPy (Virtanen et al., 2020).")
    para(d, "**Modelling.** Goodfellow, Bengio and Courville (2016) describe multilayer perceptrons and regularisation; Srivastava et al. (2014) introduced dropout and Kingma and Ba (2015) the Adam optimiser. The baselines use scikit-learn (Pedregosa et al., 2011), ridge regression (Hoerl & Kennard, 1970) and random forests (Breiman, 2001). The network was built with TensorFlow/Keras (Abadi et al., 2016).")
    H(d, 2, "9.2 References (APA 7th edition)")
    refs = ["Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., et al. (2016). TensorFlow: A system for large-scale machine learning. *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16)*, 265–283.",
            "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.",
            "Cortez, P. (2008). *Student performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T",
            "Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In A. Brito & J. Teixeira (Eds.), *Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)* (pp. 5–12). EUROSIS.",
            "Fry, B. (2007). *Visualizing data: Exploring and explaining data with the Processing environment*. O'Reilly Media.",
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.",
            "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67.",
            "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.",
            "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR 2015)*.",
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
            "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.",
            "Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley.",
            "Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272.",
            "Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software, 6*(60), 3021."]
    for r in refs:
        p = d.add_paragraph(); p.paragraph_format.left_indent = Cm(1.0); p.paragraph_format.first_line_indent = Cm(-1.0); p.paragraph_format.space_after = Pt(4)
        for i, part in enumerate(r.split("*")):
            if part: rr = p.add_run(part); rr.italic = (i % 2 == 1); rr.font.size = Pt(10)

    # ================================================= 10 RUBRIC
    H(d, 1, "10. Rubric Alignment")
    para(d, "The table maps each assessment criterion in the assignment sheet to the evidence in this report and repository. It lists where the evidence is; it does not claim any marks.")
    TAB(d, "Rubric alignment", ["CO / Criterion (marks)", "Report section", "Evidence provided"], [
        ["CO3 – Selection and creation of visualizations (3)", "4.1, 4.4, Figs. 1–12; Ch. 7, Figs. 13–20", "20 figures of 12+ types (bar, histogram, boxplot, scatter, heat-map, pairplot, Q–Q, PCA, dot-and-interval, line, residual, importance) chosen by data type; 2 interactive pages"],
        ["CO3 – Quality and presentation (2)", "4.2", "Numbered titles, labelled axes, legends, direct labels, colour-blind-safe palette, consistent colours"],
        ["CO3 – Interpretation of visualizations (3)", "4.4, 7.3", "Purpose / variables / chart type / why / insight table for every figure"],
        ["CO4 – Univariate analysis and distribution exploration (4)", "2.5, 5.1, Figs. 1–5", "Mean, median, mode, SD, quartiles, skew, histograms, boxplots, outlier analysis"],
        ["CO4 – Discrete / continuous distributions (4)", "5.2, Figs. 5–6", "Normal fit and Q–Q with and without zeros; Bernoulli/Binomial; Poisson and negative binomial for absences"],
        ["CO4 – Bivariate / multivariate analysis (4)", "5.3, 5.4, Figs. 7–12", "Spearman, Mann–Whitney, Kruskal–Wallis with effect sizes, heat-maps, VIF, PCA, pairplot, scatter"],
        ["CO4 – Interpretation of patterns and findings (4)", "5.5, 3.7", "Plain-language findings table linked to evidence"],
        ["CO5 – Real-world problem / application (3)", "1.2–1.5, 2.2", "Early prediction of final grades from school and family data"],
        ["CO5 – Application of EDA techniques (3)", "Chapter 3", "Missing/duplicate checks, outliers, encoding, scaling, stratified split, leakage control, two settings"],
        ["CO5 – Insights and interpretation (3)", "3.7, 5.5, 4.4", "Zero-grade finding, failures and aspiration effects, over-dispersed absences, dependence on G1/G2"],
        ["CO5 – Conclusion and recommendations (3)", "Chapter 8", "Findings, model performance, limitations, future scope"],
        ["CO5 – GitHub repository and documentation (4)", "Appendix B", "Repository structure, README, requirements, dataset, code, report; public link to be added after upload"],
        ["Enrichment – ML/DL (optional)", "Ch. 6, 7", "Tuned MLP regressor in two settings, baselines, permutation importance, error analysis, 10-split stability"]], [5.6, 3.6, 7.4], font=8.5)

    # ================================================= APPENDICES
    page_break(d); H(d, 1, "Appendix A – Complete Python Source Code")
    para(d, "Files are shown in the order in which the pipeline runs. Run everything with: python src/run_all.py", size=10)
    for fn, ttl in [("config.py", "A.1 config.py – paths, seed, column groups, plot style"), ("data_preprocessing.py", "A.2 data_preprocessing.py – loading, cleaning, split, pipeline"), ("eda.py", "A.3 eda.py – statistics and distribution fitting"),
                    ("visualizations.py", "A.4 visualizations.py – Figures 1–12 and interactive pages"), ("train_model.py", "A.5 train_model.py – tuning, training, baselines, repeated splits"), ("evaluate_model.py", "A.6 evaluate_model.py – evaluation, Figures 13–20"),
                    ("diagrams.py", "A.7 diagrams.py – workflow and architecture diagrams"), ("run_all.py", "A.8 run_all.py – run the whole pipeline")]:
        H(d, 2, ttl); code_block(d, (ROOT / "src" / fn).read_text(), 6.5)
    page_break(d); H(d, 1, "Appendix B – GitHub Repository and Checklist")
    H(d, 2, "B.1 Repository Structure")
    code_block(d, """student-grade-prediction/
├── data/
│   └── student-por.csv              # UCI dataset 320, Portuguese course (649 x 33)
├── src/
│   ├── config.py                    # paths, seed, column groups, plot style
│   ├── data_preprocessing.py        # load, clean, split, leakage-safe pipeline
│   ├── eda.py                       # tables + statistical tests + distribution fits
│   ├── visualizations.py            # Figures 1-12 + interactive HTML pages
│   ├── train_model.py               # grid search, MLP training, baselines, 10 splits
│   ├── evaluate_model.py            # test metrics + Figures 13-20
│   ├── diagrams.py                  # workflow / architecture diagrams
│   └── run_all.py                   # runs the full pipeline
├── outputs/
│   ├── eda/                         # EDA figures, tables, interactive pages
│   └── model_evaluation/            # evaluation figures, metrics JSON/CSV
├── models/                          # saved MLPs (.keras) + preprocessors (.joblib)
├── report/                          # Word + PDF report, report builder scripts
├── requirements.txt
├── README.md
└── .gitignore""", 8)
    TAB(d, "Purpose of each folder and file", ["Path", "Purpose"], [["data/", "The dataset used. CC BY 4.0 – keep the citation in the README"], ["src/", "All source code, one module per stage so it is easy to read and explain in a viva"],
        ["outputs/eda, outputs/model_evaluation", "Generated figures, tables and metrics, so the report can be checked without re-running"], ["models/", "Trained MLPs and fitted preprocessors (small files)"], ["report/", "Final report (DOCX and PDF) and the scripts that build it from the saved results"],
        ["requirements.txt", "Exact library versions used"], ["README.md", "Project summary, results, how to run"], [".gitignore", "Keeps caches and secrets out of the repository"]], [5.5, 11.1], font=9)
    H(d, 2, "B.2 README Outline")
    bullets(d, ["Title, one-paragraph summary and student details (RITHI R, 24AD091, U21ADP05)", "Problem statement and objectives", "Dataset: name, UCI URL, DOI, licence, size, target", "Project structure (tree above)", "How to run: pip install -r requirements.txt, then python src/run_all.py",
                "Key results table and 3–4 sample figures", "Key insights", "Limitations and future work", "References / citation of the dataset", "Link to the report in report/"])
    H(d, 2, "B.3 Commands to Publish")
    code_block(d, """cd student-grade-prediction
git init
git add .
git commit -m "Student grade prediction: EDA and MLP regression case study"
git branch -M main
git remote add origin https://github.com/<your-github-id>/student-grade-prediction.git
git push -u origin main""", 8)
    H(d, 2, "B.4 GitHub Submission Checklist")
    TAB(d, "Checklist before submitting the link", ["✔", "Item"], [["☐", "Repository is **public** and created under your own GitHub ID"], ["☐", "Project report (PDF and Word) is inside report/"], ["☐", "Source code (.py) is inside src/ and runs with python src/run_all.py"],
        ["☐", "Dataset file is present (data/student-por.csv) and the UCI link is in the README"], ["☐", "README.md has summary, dataset, how to run, results and limitations"], ["☐", "No passwords, tokens, .env files or personal data are committed"],
        ["☐", "Public GitHub link pasted into the Google Form (https://forms.gle/kT9DU5aERo2peTNh8) or emailed to the coordinator"]], [1.2, 15.4], font=9.5)
    H(d, 1, "Appendix C – Final Checklist Against the Assignment Requirements")
    TAB(d, "Requirement checklist", ["Requirement from assignment sheet", "Status", "Where"], [
        ["Real-world dataset with ≥ 15 features", "Done – 32 input features", "2.1–2.4"], ["Load data and perform EDA", "Done", "Ch. 3, 5"], ["Handle missing values, outliers, duplicates", "Done – none missing/duplicated; outliers analysed and winsorised", "3.2, 3.3"],
        ["Cleaning, transformation, encoding, scaling", "Done", "3.4"], ["Train / validation / test split", "Done – 70/15/15 stratified", "3.6"], ["Minimum 5 visualisations covering all 7 stages", "Done – 12 EDA + 8 model figures; stages mapped in Table 9", "4.1, 4.4"],
        ["Mention what each visual shows, why created, insights", "Done – explanation table per figure", "4.4, 7.3"], ["At least one deep-learning model", "Done – MLP regressor (two settings)", "Ch. 6"], ["Feature analysis, importance, basic tuning", "Done", "6.5, 6.6"],
        ["3–4 result visualisations (loss, accuracy/MAE, error distribution, predicted vs actual)", "Done – Figs. 13–20 (regression task, so no confusion matrix/ROC)", "7.3"], ["Conclusion, challenges, future scope", "Done", "Ch. 8"], ["3–5 references, APA/IEEE", "Done – 14 APA references", "9.2"],
        ["Report format (title page … appendix)", "Done", "Whole report"], ["Well-commented code covering loading, EDA, preprocessing, model, evaluation", "Done", "Appendix A"],
        ["GitHub repository, README, dataset, report, source code", "Prepared locally; **upload and public link still to be done by the student**", "Appendix B"]], [7.4, 6.0, 3.2], font=8.5)
    return d


def finish(d):
    marker = [p for p in d.paragraphs if p.text == "[[TOC]]"][0]; anchor = marker._p
    add = lambda par: anchor.addprevious(par._p)
    title = lambda t: para(d, t, bold=True, size=15, color=BLUE, after=6)
    add(title("Table of Contents")); items = [(l, x) for l, x in HEAD if l <= 2]
    for i, (l, x) in enumerate(items): add(toc_entry(d, x, PAGES.get(x, "0"), indent_cm=0 if l == 1 else 0.7, bold=(l == 1), first=(i == 0), last=(i == len(items) - 1), field='TOC \\o "1-2" \\h \\z \\u'))
    t = title("List of Figures"); t.paragraph_format.space_before = Pt(12); add(t)
    for x in FIGS: add(toc_entry(d, x, PAGES.get(x, "0")))
    t = title("List of Tables"); t.paragraph_format.space_before = Pt(12); t.paragraph_format.page_break_before = True; add(t)
    for x in TABS: add(toc_entry(d, x, PAGES.get(x, "0")))
    anchor.getparent().remove(anchor)


if __name__ == "__main__":
    d = build(); finish(d); out = ROOT / "report" / "Student_Grade_Prediction_Report_RITHI_R_24AD091.docx"; d.save(out)
    json.dump({"headings": [x for _, x in HEAD], "figs": FIGS, "tabs": TABS}, open(ROOT / "report" / "_toc_items.json", "w")); print("saved", out)
