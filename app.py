from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import tempfile, os, math, json, smtplib, ssl
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import plotly.express as px

app = Flask(__name__)
app.secret_key = "CHANGE_ME_SECRET_KEY"  # required for flash()

# ---------- CONFIG ----------
ROW_CHUNK = 100_000               # chunk size for large CSVs
LARGE_FILE_MB = 25                # >= this uses chunked rules-only fast path
ML_SAMPLE_ROWS = 20_000           # sample for ML on small/medium
ML_CONTAMINATION_DEFAULT = 0.03

# Email (auto alert + manual send)
ALERT_THRESHOLD_PCT = 5.0
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_TO   = os.getenv("EMAIL_TO", "")       # used for auto alert only
SMTP_HOST  = os.getenv("SMTP_HOST", "")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER  = os.getenv("SMTP_USER", "")
SMTP_PASS  = os.getenv("SMTP_PASS", "")

FRAUD_HISTORY_FILE = "fraud_history.csv"     # similarity reference (optional)
CUSTOM_RULES_FILE  = "rules.json"            # custom rules (JSON array)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def normalize_cols(cols):
    return [c.strip().replace(" ", "").lower() for c in cols]

def get_col(df, *candidates):
    cols = [c.strip().replace(" ", "").lower() for c in df.columns]
    idx = {c:i for i,c in enumerate(cols)}
    for cand in candidates:
        k = cand.strip().replace(" ", "").lower()
        if k in idx:
            return df.columns[idx[k]]
    return None

def safe_parse_amount(x):
    try: return float(x)
    except Exception: return np.nan

def safe_parse_date(x):
    try: return pd.to_datetime(x, errors="coerce")
    except Exception: return pd.NaT

# --------------------------------------------------------------------------------------
# Email: auto alert (threshold) + manual send to any email
# --------------------------------------------------------------------------------------
def send_email_alert(summary, csv_bytes):
    """Automatic alert to EMAIL_TO when flagged % >= threshold."""
    if not (SMTP_HOST and SMTP_PORT and EMAIL_FROM and EMAIL_TO and SMTP_USER and SMTP_PASS):
        print("[Email] SMTP/Email env vars not set; skipping auto alert.")
        return
    subject = "AI Audit Alert: Anomalies Detected"
    body = f"""
Hi,

Your audit exceeded the threshold.

Total rows: {summary["total"]}
Flagged: {summary["flagged"]} ({summary["flagged_pct"]}%)

By source:
  - Rules: {summary["by_source"].get("rules",0)}
  - ML:    {summary["by_source"].get("ml",0)}
  - Both:  {summary["by_source"].get("both",0)}

The flagged CSV is attached.

Regards,
AI Audit System
"""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename=flagged_transactions.csv")
    msg.attach(part)
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("[Email] Auto alert sent.")
    except Exception as e:
        print("[Email] Auto alert failed:", e)

def send_email_to(recipient_email, summary, csv_bytes):
    """Manual send to any recipient (used by the Send Email button)."""
    sender = EMAIL_FROM or SMTP_USER
    if not (SMTP_HOST and SMTP_PORT and sender and SMTP_USER and SMTP_PASS):
        raise RuntimeError("SMTP credentials not configured (set EMAIL_FROM/SMTP_* env vars).")
    subject = "AI Audit Report Summary"
    body = f"""
AI Audit Summary
----------------
Total Transactions: {summary["total"]}
Flagged Transactions: {summary["flagged"]}
Flagged Percentage: {summary["flagged_pct"]}%

By source:
  - Rules: {summary["by_source"].get("rules",0)}
  - ML:    {summary["by_source"].get("ml",0)}
  - Both:  {summary["by_source"].get("both",0)}
"""
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename=flagged_transactions.csv")
    msg.attach(part)
    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(sender, recipient_email, msg.as_string())
    return True

# --------------------------------------------------------------------------------------
# Custom Rules (sanitized)
# --------------------------------------------------------------------------------------
def sanitize_custom_rules(rules):
    if rules is None:
        return []
    if isinstance(rules, str):
        try:
            rules = json.loads(rules)
        except Exception:
            return []
    if isinstance(rules, dict):
        rules = [rules]
    if not isinstance(rules, list):
        return []
    clean = []
    for r in rules:
        if isinstance(r, dict):
            clean.append(r)
        elif isinstance(r, str):
            try:
                r2 = json.loads(r)
                if isinstance(r2, dict): clean.append(r2)
                elif isinstance(r2, list): clean += [x for x in r2 if isinstance(x, dict)]
            except Exception:
                continue
        elif isinstance(r, list):
            clean += [x for x in r if isinstance(x, dict)]
    return clean

def load_custom_rules():
    if os.path.exists(CUSTOM_RULES_FILE):
        try:
            with open(CUSTOM_RULES_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            return sanitize_custom_rules(raw)
        except Exception:
            return []
    return []

def save_custom_rules(rules):
    with open(CUSTOM_RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

def apply_custom_rules(row, rules, colmap):
    """
    Rule format:
      {"if": {"amount": {">": 50000}, "vendor": {"=": "ABC"}}, "reason": "Custom Rule Triggered"}
    Supported ops: ==, =, !=, >, >=, <, <=, in
    """
    reasons, hits = [], []
    if not rules:
        return reasons, hits
    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            continue
        cond = rule.get("if", {})
        if not isinstance(cond, dict):
            continue
        matched = True
        for field, cmpd in cond.items():
            if not isinstance(cmpd, dict):
                matched = False; break
            col = colmap.get(field)
            if not col or col not in row:
                matched = False; break
            val = row[col]
            try: val_num = float(val)
            except Exception: val_num = None
            for op, target in cmpd.items():
                op = str(op).strip().lower()
                t_str = target if isinstance(target, str) else str(target)
                if op in ("=", "=="):
                    if str(val).strip() != t_str.strip(): matched = False
                elif op == "!=":
                    if str(val).strip() == t_str.strip(): matched = False
                elif op == "in":
                    if t_str.strip().lower() not in str(val).strip().lower(): matched = False
                elif op == ">":
                    if val_num is None or not (val_num > float(target)): matched = False
                elif op == ">=":
                    if val_num is None or not (val_num >= float(target)): matched = False
                elif op == "<":
                    if val_num is None or not (val_num < float(target)): matched = False
                elif op == "<=":
                    if val_num is None or not (val_num <= float(target)): matched = False
                else:
                    matched = False
                if not matched: break
            if not matched: break
        if matched:
            reasons.append(rule.get("reason", f"Custom Rule {idx}"))
            hits.append(f"C{idx}")
    return reasons, hits

# --------------------------------------------------------------------------------------
# Policy checks
# --------------------------------------------------------------------------------------
def apply_policy_checks(row, cols, opts, amt_mean):
    reasons, hits = [], []
    vendor_col = cols.get("vendor")
    date_col   = cols.get("date")
    amount_col = cols.get("amount")
    invoice_col= cols.get("invoice")
    gst_col    = cols.get("gst")

    vendor = str(row[vendor_col]).strip() if vendor_col and pd.notna(row.get(vendor_col)) else None
    dt     = row[date_col] if date_col and isinstance(row.get(date_col), pd.Timestamp) else None
    amt    = float(row[amount_col]) if amount_col and pd.notna(row.get(amount_col)) else None
    invoice= str(row[invoice_col]).strip() if invoice_col and pd.notna(row.get(invoice_col)) else ""
    gst    = None
    if gst_col and pd.notna(row.get(gst_col)):
        try: gst = float(row[gst_col])
        except Exception: gst = None

    # After-hours
    if opts.get("check_time") and dt is not None:
        hr = dt.hour
        start = opts.get("biz_start", 9); end = opts.get("biz_end", 18)
        if hr < start or hr > end:
            reasons.append("Transaction outside business hours"); hits.append("P1")
    # Invoice missing
    if opts.get("check_invoice_missing") and invoice_col:
        if invoice == "" or invoice.lower() in ("nan", "none"):
            reasons.append("Missing invoice number"); hits.append("P2")
    # Vendor WL/BL
    wl = opts.get("vendor_whitelist", set()); bl = opts.get("vendor_blacklist", set())
    if vendor is not None:
        if wl and vendor not in wl:
            reasons.append("Vendor not in whitelist"); hits.append("P3")
        if bl and vendor in bl:
            reasons.append("Vendor in blacklist"); hits.append("P4")
    # GST mismatch
    if opts.get("check_gst") and gst is not None:
        expected = opts.get("gst_expected", None); tol = opts.get("gst_tolerance", 0.5)
        if expected is not None and abs(gst - expected) > tol:
            reasons.append(f"GST mismatch (expected ~{expected})"); hits.append("P5")

    return reasons, hits

# --------------------------------------------------------------------------------------
# Rule engine (streaming) with custom rules + policy + baseline rules
# --------------------------------------------------------------------------------------
def rule_engine_on_chunk(df_chunk, opts, seen_ids, seen_vendor_amt, amt_stats, custom_rules):
    flagged_rows = []
    norm_map = {c: c.strip().replace(" ", "").lower() for c in df_chunk.columns}
    inv_map  = {v: k for k, v in norm_map.items()}

    amount_o  = inv_map.get(get_col(df_chunk, "Amount","Transaction Amount","txn_amount","amt") or "", None)
    vendor_o  = inv_map.get(get_col(df_chunk, "Vendor","Payee","Supplier") or "", None)
    txnid_o   = inv_map.get(get_col(df_chunk, "TransactionID","TxnID","ID","Reference") or "", None)
    date_o    = inv_map.get(get_col(df_chunk, "Date","Txn Date","TransactionDate","datetime","timestamp") or "", None)
    invoice_o = inv_map.get(get_col(df_chunk, "Invoice","InvoiceNo","InvoiceNumber") or "", None)
    gst_o     = inv_map.get(get_col(df_chunk, "GST","GST_Rate","Tax","TaxRate") or "", None)

    amounts = df_chunk[amount_o].map(safe_parse_amount) if amount_o else pd.Series([np.nan]*len(df_chunk), index=df_chunk.index)
    dates   = df_chunk[date_o].map(safe_parse_date)     if date_o   else pd.Series([pd.NaT]*len[df_chunk], index=df_chunk.index) if date_o else pd.Series([pd.NaT]*len(df_chunk), index=df_chunk.index)

    amt_mean, high_thresh = amt_stats
    policy_cols = {"vendor": vendor_o, "date": date_o, "amount": amount_o, "invoice": invoice_o, "gst": gst_o}
    colmap_custom = {"amount": amount_o, "vendor": vendor_o, "transactionid": txnid_o, "date": date_o, "invoice": invoice_o, "gst": gst_o}

    for i, row in df_chunk.iterrows():
        reasons, hits = [], []
        a = amounts[i]; d = dates[i]
        v = row[vendor_o] if vendor_o else None
        t = row[txnid_o]  if txnid_o  else None

        # Baseline rules
        if amount_o and pd.notna(a) and a < 0:
            reasons.append("Negative amount"); hits.append("R1")
        if amount_o and (high_thresh is not None) and pd.notna(a) and a > high_thresh:
            reasons.append("Unusually high amount (outlier by rule)"); hits.append("R2")
        if txnid_o and pd.notna(t):
            key = str(t)
            if key in seen_ids:
                reasons.append("Duplicate Transaction ID"); hits.append("R3")
            else:
                seen_ids.add(key)
        if vendor_o and amount_o and (pd.notna(v) and pd.notna(a)):
            key = (str(v), float(a))
            if key in seen_vendor_amt:
                reasons.append("Duplicate Vendor+Amount combination"); hits.append("R4")
            else:
                seen_vendor_amt.add(key)
        if date_o and amount_o and (pd.notna(d) and pd.notna(a)) and isinstance(d, pd.Timestamp):
            if d.weekday() >= 5 and (amt_mean is not None) and a > (amt_mean * 1.5):
                reasons.append("High-value weekend transaction"); hits.append("R5")

        # Policy checks (inject parsed values)
        temp_row = row.copy()
        if amount_o: temp_row[amount_o] = a
        if date_o:   temp_row[date_o]   = d
        pr, ph = apply_policy_checks(temp_row, policy_cols, opts, amt_mean)
        reasons += pr; hits += ph

        # Custom rules
        cr, ch = apply_custom_rules(temp_row, custom_rules, colmap_custom)
        reasons += cr; hits += ch

        if reasons:
            rowd = row.to_dict()
            rowd["Reasons"] = "; ".join(reasons)
            rowd["RuleHits"] = ",".join(hits)
            flagged_rows.append(rowd)

    return flagged_rows

def compute_amount_stats_iter(file_path, amount_col_name):
    if not amount_col_name:
        return (None, None)
    n = 0; mean = 0.0; M2 = 0.0
    for ch in pd.read_csv(file_path, chunksize=ROW_CHUNK, dtype=str, on_bad_lines='skip'):
        if amount_col_name not in ch.columns: continue
        vals = ch[amount_col_name].map(safe_parse_amount)
        for x in vals.dropna():
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta2 * delta
    if n == 0: return (None, None)
    if n == 1: return (mean, 0.0)
    variance = M2 / n; std = math.sqrt(max(variance, 0.0))
    return (mean, std)

# --------------------------------------------------------------------------------------
# ML + SHAP (small/medium)
# --------------------------------------------------------------------------------------
def ml_on_small_df(df, contamination=ML_CONTAMINATION_DEFAULT, random_state=42):
    import shap
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.shape[1] == 0:
        return pd.DataFrame(columns=list(df.columns) + ["ml_score", "Reasons", "RuleHits"])
    if len(df_num) > ML_SAMPLE_ROWS:
        df_num = df_num.sample(n=ML_SAMPLE_ROWS, random_state=random_state)
    df_num = df_num.fillna(df_num.median(numeric_only=True)).astype(np.float32)

    model = IsolationForest(
        contamination=float(contamination),
        n_estimators=100,
        max_samples=min(10_000, len(df_num)),
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(df_num)
    preds = model.predict(df_num)
    scores = -model.decision_function(df_num)
    anomalies_idx = np.where(preds == -1)[0]
    if len(anomalies_idx) == 0:
        return pd.DataFrame(columns=list(df.columns) + ["ml_score", "Reasons", "RuleHits"])

    # SHAP (TreeExplainer -> fallback to Explainer(decision_function) -> feature_importances_/first cols)
    top3_labels = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_num)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        top3_labels = []
        for i in range(len(df_num)):
            abs_vals = np.abs(vals[i])
            order = np.argsort(abs_vals)[-3:][::-1]
            names = [df_num.columns[j] for j in order]
            top3_labels.append(", ".join(names))
    except Exception:
        try:
            explainer = shap.Explainer(model.decision_function, df_num)
            sv = explainer(df_num)
            vals = np.array(sv.values)
            if vals.ndim == 1: vals = vals.reshape(-1, 1)
            top3_labels = []
            for i in range(len(df_num)):
                abs_vals = np.abs(vals[i])
                order = np.argsort(abs_vals)[-3:][::-1]
                order = order[:min(3, len(df_num.columns))]
                names = [df_num.columns[j] for j in order]
                top3_labels.append(", ".join(names))
        except Exception:
            try:
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    order = np.argsort(importances)[-3:][::-1]
                    default_top = [df_num.columns[j] for j in order]
                else:
                    default_top = list(df_num.columns[:3])
            except Exception:
                default_top = list(df_num.columns[:3])
            top3_labels = [", ".join(default_top) for _ in range(len(df_num))]

    flagged = df.loc[df_num.index[anomalies_idx]].copy()
    flagged["ml_score"] = scores[anomalies_idx]
    flagged["Reasons"] = ["ML anomaly — top factors: " + top3_labels[i] for i in anomalies_idx]
    flagged["RuleHits"] = "ML"
    return flagged

# --------------------------------------------------------------------------------------
# Similarity vs Fraud History
# --------------------------------------------------------------------------------------
def encode_for_similarity(df):
    if df.empty: return None, None
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).fillna(0)
    cat = df.select_dtypes(exclude=[np.number]).fillna("NA")
    if cat.shape[1] > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = enc.fit_transform(cat)
        X = np.hstack([num.values, X_cat])
    else:
        X = num.values
    return X, None

def nearest_similar(flagged_df, history_df, topk=1):
    if flagged_df.empty or history_df.empty:
        return ["N/A"] * len(flagged_df)
    common_cols = list(set(flagged_df.columns).intersection(set(history_df.columns)))
    if not common_cols:
        return ["N/A"] * len(flagged_df)
    X_f, _ = encode_for_similarity(flagged_df[common_cols])
    X_h, _ = encode_for_similarity(history_df[common_cols])
    if X_f is None or X_h is None:
        return ["N/A"] * len(flagged_df)
    sims = cosine_similarity(X_f, X_h)
    res = []
    hist_tid_col = get_col(history_df, "transactionid","id","reference")
    hist_vendor  = get_col(history_df, "vendor","payee","supplier")
    hist_amount  = get_col(history_df, "amount","transactionamount","txn_amount","amt")
    for i in range(sims.shape[0]):
        j = np.argmax(sims[i]); score = sims[i, j]
        label = f"Similarity {score:.2f}"
        if hist_tid_col and pd.notna(history_df.iloc[j].get(hist_tid_col)):
            label += f" to TxnID {history_df.iloc[j][hist_tid_col]}"
        elif hist_vendor and hist_amount:
            label += f" to {history_df.iloc[j][hist_vendor]} (Amt {history_df.iloc[j][hist_amount]})"
        res.append(label)
    return res

# --------------------------------------------------------------------------------------
# Summarize & order
# --------------------------------------------------------------------------------------
def summarize_and_order(df_flagged, total_rows):
    if df_flagged is None or df_flagged.empty:
        return pd.DataFrame(), {
            "total": total_rows,
            "flagged": 0,
            "flagged_pct": 0.0,
            "by_source": {"rules":0,"ml":0,"both":0}
        }
    by = {"rules":0, "ml":0, "both":0}
    if "RuleHits" in df_flagged.columns:
        for v in df_flagged["RuleHits"].fillna(""):
            hits = set(h.strip().upper() for h in v.split(",") if h)
            if "ML" in hits and len(hits) > 1: by["both"] += 1
            elif "ML" in hits: by["ml"] += 1
            elif len(hits) > 0: by["rules"] += 1
    summary = {
        "total": total_rows,
        "flagged": len(df_flagged),
        "flagged_pct": round(100 * len(df_flagged) / max(1, total_rows), 2),
        "by_source": by
    }
    helper_cols = [c for c in ["ml_score","Reasons","RuleHits","SimilarityHint"] if c in df_flagged.columns]
    data_cols = [c for c in df_flagged.columns if c not in helper_cols]
    ordered = df_flagged[data_cols + helper_cols]
    return ordered, summary

# --------------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------------
def process_small_file(file_storage, mode, contamination, policy_opts, custom_rules):
    try:
        df = pd.read_csv(file_storage)
    except UnicodeDecodeError:
        file_storage.stream.seek(0)
        df = pd.read_csv(file_storage, encoding="latin-1")
    df.columns = normalize_cols(df.columns)

    flagged_parts = []
    amt_col = get_col(df, "amount","transactionamount","txn_amount","amt")
    if amt_col is not None and df[amt_col].notna().any():
        s = pd.to_numeric(df[amt_col], errors="coerce")
        mean = s.mean(); std = s.std(ddof=0) if s.count() > 1 else 0.0
        high_thresh = mean + 3 * (std if pd.notna(std) else 0.0)
    else:
        mean = high_thresh = None

    if mode in ("rules","both"):
        df_view = df.copy()
        flagged_rules = rule_engine_on_chunk(
            df_view, policy_opts, seen_ids=set(), seen_vendor_amt=set(),
            amt_stats=(mean, high_thresh), custom_rules=custom_rules
        )
        if flagged_rules: flagged_parts.append(pd.DataFrame(flagged_rules))

    if mode in ("ml","both"):
        flagged_ml = ml_on_small_df(df.copy(), contamination=contamination)
        if not flagged_ml.empty:
            flagged_parts.append(flagged_ml)

    if not flagged_parts:
        return pd.DataFrame(), len(df)

    combined = pd.concat(flagged_parts, ignore_index=True, sort=False).drop_duplicates()

    if os.path.exists(FRAUD_HISTORY_FILE):
        hist = pd.read_csv(FRAUD_HISTORY_FILE)
        hints = nearest_similar(combined.copy(), hist.copy(), topk=1)
        combined["SimilarityHint"] = hints

    return combined, len(df)

def process_large_file_to_temp(file_storage_or_path, mode, contamination, policy_opts, custom_rules):
    # only rules path for large files (fast)
    if isinstance(file_storage_or_path, str):
        temp_path = file_storage_or_path; cleanup = False
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file_storage_or_path.save(tmp.name)
        temp_path = tmp.name; cleanup = True

    first = next(pd.read_csv(temp_path, chunksize=ROW_CHUNK, dtype=str, on_bad_lines='skip'))
    first_norm = normalize_cols(first.columns)
    rename_map = {c:n for c,n in zip(first.columns, first_norm)}
    amount_orig = None
    for oc, nc in zip(first.columns, first_norm):
        if nc in ("amount","transactionamount","txn_amount","amt"):
            amount_orig = oc; break

    mean, std = compute_amount_stats_iter(temp_path, amount_orig)
    high_thresh = mean + 3*std if (mean is not None and std is not None) else None

    seen_ids = set(); seen_vendor_amt = set()
    flagged_all = []; total_rows = 0

    for chunk in pd.read_csv(temp_path, chunksize=ROW_CHUNK, dtype=str, on_bad_lines='skip'):
        total_rows += len(chunk)
        chunk = chunk.rename(columns=rename_map)
        chunk.columns = normalize_cols(chunk.columns)
        flagged_rows = rule_engine_on_chunk(
            chunk, policy_opts, seen_ids, seen_vendor_amt, (mean, high_thresh), custom_rules
        )
        if flagged_rows: flagged_all.extend(flagged_rows)

    if cleanup:
        try: os.remove(temp_path)
        except Exception: pass

    if not flagged_all:
        return pd.DataFrame(), total_rows

    combined = pd.DataFrame(flagged_all).drop_duplicates()

    if os.path.exists(FRAUD_HISTORY_FILE):
        hist = pd.read_csv(FRAUD_HISTORY_FILE)
        hints = nearest_similar(combined.copy(), hist.copy(), topk=1)
        combined["SimilarityHint"] = hints

    return combined, total_rows

# --------------------------------------------------------------------------------------
# Routes (no login, single-user)
# --------------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/manage-rules", methods=["GET","POST"])
def manage_rules():
    if request.method == "POST":
        payload = request.form.get("rules_json","").strip()
        try:
            data = json.loads(payload) if payload else []
            data = sanitize_custom_rules(data)
            save_custom_rules(data)
            flash("Rules saved.")
        except Exception as e:
            flash(f"Invalid JSON: {e}")
        return redirect(url_for("manage_rules"))
    current = load_custom_rules()
    return render_template("manage_rules.html", current=json.dumps(current, indent=2))

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("result.html", message="No file part.", table=None, summary=None, download_available=False)
    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", message="No file selected.", table=None, summary=None, download_available=False)

    mode = request.form.get("mode", "both")
    contamination = request.form.get("contamination", str(ML_CONTAMINATION_DEFAULT))
    try:
        contamination = float(contamination)
        contamination = max(0.001, min(0.2, contamination))
    except Exception:
        contamination = ML_CONTAMINATION_DEFAULT

    vendor_whitelist = set([v.strip() for v in request.form.get("vendor_whitelist","").split(",") if v.strip()]) if request.form.get("vendor_whitelist") else set()
    vendor_blacklist = set([v.strip() for v in request.form.get("vendor_blacklist","").split(",") if v.strip()]) if request.form.get("vendor_blacklist") else set()
    policy_opts = {
        "check_vendor_dup": request.form.get("check_vendor_dup") == "on",
        "check_time": request.form.get("check_time") == "on",
        "check_invoice_missing": request.form.get("check_invoice_missing") == "on",
        "check_gst": request.form.get("check_gst") == "on",
        "biz_start": int(request.form.get("biz_start", 9) or 9),
        "biz_end": int(request.form.get("biz_end", 18) or 18),
        "gst_expected": float(request.form.get("gst_expected")) if request.form.get("gst_expected") else None,
        "gst_tolerance": float(request.form.get("gst_tolerance", 0.5) or 0.5),
        "vendor_whitelist": vendor_whitelist,
        "vendor_blacklist": vendor_blacklist,
    }

    custom_rules = load_custom_rules()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    file.save(tmp.name)
    file_mb = os.path.getsize(tmp.name) / (1024*1024)
    use_large = file_mb >= LARGE_FILE_MB

    if not use_large:
        flagged, total = process_small_file(open(tmp.name, "rb"), mode, contamination, policy_opts, custom_rules)
    else:
        flagged, total = process_large_file_to_temp(tmp.name, mode, contamination, policy_opts, custom_rules)

    try: os.remove(tmp.name)
    except Exception: pass

    ordered, summary = summarize_and_order(flagged, total_rows=total)

    # store for download/dashboard/email
    app.config["LATEST_FLAGGED"] = ordered
    app.config["LATEST_TOTAL"]   = summary["total"]

    if ordered.empty:
        return render_template("result.html",
                               message="✅ No suspicious transactions found.",
                               table=None, summary=summary, download_available=False, show_dashboard=False)

    # Auto email alert if threshold exceeded (uses EMAIL_TO)
    if summary["flagged_pct"] >= ALERT_THRESHOLD_PCT:
        buf = BytesIO(); ordered.to_csv(buf, index=False); buf.seek(0)
        send_email_alert(summary, buf.getvalue())

    html = ordered.to_html(classes="data", index=False, escape=False, justify="center")
    return render_template("result.html",
                           message="⚠️ Suspicious transactions flagged:",
                           table=html, summary=summary,
                           download_available=True, show_dashboard=True)

@app.route("/download")
def download():
    flagged = app.config.get("LATEST_FLAGGED", None)
    if flagged is None or flagged.empty:
        return render_template("result.html", message="No flagged data available to download.", table=None, summary=None, download_available=False)
    buf = BytesIO()
    flagged.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="flagged_transactions.csv", mimetype="text/csv")

@app.route("/dashboard")
def dashboard():
    df = app.config.get("LATEST_FLAGGED", pd.DataFrame())
    if df.empty:
        return render_template("dashboard.html", charts=[])

    charts = []
    if "Reasons" in df.columns and df["Reasons"].notna().any():
        reason_counts = df["Reasons"].value_counts().head(10)
        fig = px.pie(values=reason_counts.values, names=reason_counts.index, title="Top Reasons (Flagged)")
        charts.append(fig.to_html(full_html=False))

    vendor_col = get_col(df, "vendor","payee","supplier")
    if vendor_col and df[vendor_col].notna().any():
        vc = df[vendor_col].value_counts().head(10)
        fig2 = px.bar(x=vc.index, y=vc.values, labels={"x":"Vendor","y":"Flagged Count"}, title="Top Vendors in Flags")
        charts.append(fig2.to_html(full_html=False))

    date_col = get_col(df, "date","txndate","transactiondate","datetime","timestamp")
    if date_col and df[date_col].notna().any():
        dt = pd.to_datetime(df[date_col], errors="coerce")
        series = dt.dt.date.value_counts().sort_index()
        fig3 = px.line(x=series.index, y=series.values, labels={"x":"Date","y":"Flagged Count"}, title="Flagged Trend Over Time")
        charts.append(fig3.to_html(full_html=False))

    return render_template("dashboard.html", charts=charts)

# Manual Send Email button
@app.route("/send-email", methods=["POST"])
def send_email_route():
    recipient = request.form.get("email", "").strip()
    df = app.config.get("LATEST_FLAGGED", pd.DataTable() if False else pd.DataFrame())
    total = app.config.get("LATEST_TOTAL", len(df))
    if not recipient:
        return "❌ No email provided.", 400
    if df.empty:
        return "⚠️ No flagged transactions available to send.", 400

    flagged = len(df)
    pct = round(100 * flagged / max(1, total), 2)
    by = {"rules":0,"ml":0,"both":0}
    if "RuleHits" in df.columns:
        for v in df["RuleHits"].fillna(""):
            hits = set(h.strip().upper() for h in v.split(",") if h)
            if "ML" in hits and len(hits) > 1: by["both"] += 1
            elif "ML" in hits: by["ml"] += 1
            elif len(hits) > 0: by["rules"] += 1
    summary = {"total": total, "flagged": flagged, "flagged_pct": pct, "by_source": by}

    buf = BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    try:
        send_email_to(recipient, summary, buf.getvalue())
        return f"✅ Audit summary sent successfully to {recipient}!"
    except Exception as e:
        print("Email error:", e)
        return f"❌ Failed to send email: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
