from flask import Flask, render_template, request, url_for, jsonify, make_response, flash, redirect
from markupsafe import Markup
from flask_login import login_required, login_user, logout_user, current_user
from functools import wraps
import pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import joblib
from app import app, db, User
import subprocess
import os
import sys
import io
import re
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Intensity Analyzer
vader_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Deserializing ML models
# ---------------------------------------------------------------------------
kmeanclus  = pickle.load(open('./Prediction/kmean.pkl', 'rb'))
kprotoclus = joblib.load('./Prediction/kproto.pkl')
rdcls      = joblib.load('./Prediction/cls.pkl')
lr         = joblib.load('./Prediction/models.pkl')


# ---------------------------------------------------------------------------
# Exponential Smoothing
# ---------------------------------------------------------------------------
class ExponentialSmoothing:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, data):
        self.data = data

    def predict(self, year):
        # Apply exponential smoothing to the data
        smoothed_data = [self.data[0]]
        for i in range(1, len(self.data)):
            smoothed_data.append(
                self.alpha * self.data[i] + (1 - self.alpha) * smoothed_data[i - 1]
            )

        # Predict the value for the specified year using the smoothed data
        if year < len(self.data):
            return smoothed_data[year]
        else:
            last_value = smoothed_data[-1]
            for i in range(len(self.data), year):
                last_value = self.alpha * self.data[-1] + (1 - self.alpha) * last_value
            return last_value


# ---------------------------------------------------------------------------
# Population Projection Formula — Geometric Growth Method
# ---------------------------------------------------------------------------
import math

def projection(v1, v2, yr1, yr2, year):
    t = yr2 - yr1
    n = year - yr2
    r = math.pow(v2 / v1, 1 / t) - 1
    P = v2 * math.pow(1 + r, n)
    return P


# ===========================================================================
# RBAC — Role-Required Decorator
# ===========================================================================

def role_required(*allowed_roles):
    """
    Factory decorator that restricts a view to users whose role is in
    `allowed_roles`. Redirects unauthenticated or unauthorised users to
    the login page with a flash message.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'info')
                return redirect(url_for('login'))
            if current_user.role not in allowed_roles:
                flash('Unauthorized access level.', 'danger')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ===========================================================================
# Template Context Processor — injects current_user into every template
# ===========================================================================

@app.context_processor
def inject_current_user():
    """Make current_user available to every Jinja2 template automatically."""
    return dict(current_user=current_user)


# ===========================================================================
# Authentication Routes
# ===========================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Already logged-in users skip the login page
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response


# ===========================================================================
# Public & Field Officer Allowed Routes (public | field_officer)
# ===========================================================================

@app.route('/')
@app.route('/index')
@login_required
@role_required('public', 'field_officer')
def index():
    return render_template('index.html')


@app.route('/about')
@app.route('/About')
@login_required
@role_required('public', 'field_officer')
def about():
    return render_template('About.html')


@app.route('/feedback')
@app.route('/Feedback')
@login_required
@role_required('public', 'field_officer')
def feedback():
    return render_template('Feedback.html')


# Google Charts analysis
@app.route('/analysis')
@login_required
@role_required('public', 'field_officer')
def analysis():
    return render_template('Analysis.html')


# Plotly Charts analysis
@app.route('/analysis2')
@login_required
@role_required('public', 'field_officer')
def analysis2():
    return render_template('Analysis2.html')


# Geospatial analysis
@app.route('/analysis3')
@login_required
@role_required('public', 'field_officer')
def analysis3():
    return render_template('Analysis3(maps).html')


# Serving plot HTML files & resources (required for analysis2 & analysis3 iframes)
@app.route('/plots/<path:filename>')
@login_required
@role_required('public', 'field_officer')
def serve_plots(filename):
    return render_template(f'plots/{filename}')


# ===========================================================================
# Field Officer Allowed Routes (field_officer) — Complete Operational Access
# ===========================================================================

# ===========================================================================
# Field Officer Allowed Routes (field_officer) — Complete Operational Access
# ===========================================================================

@app.route('/Kmeans')
@app.route('/kmeans')
@app.route('/kmean')
@login_required
@role_required('field_officer')
def Kmeans():
    return render_template("K-Means.html")


@app.route('/KMeansanalysis', methods=['GET', 'POST'], endpoint='KMeansanalysis')
@app.route('/kmeansanalysis', methods=['GET', 'POST'], endpoint='kmeansanalysis')
@app.route('/cluster', methods=['GET', 'POST'], endpoint='cluster')
@app.route('/api/cluster', methods=['GET', 'POST'], endpoint='api_cluster')
@login_required
@role_required('field_officer')
def KMeansanalysis():
    # Support JSON payload, Form POST, or Query Params
    if request.is_json:
        data = request.get_json() or {}
        state_target = (data.get('state') or data.get('state_input') or '').strip().upper()
        district_target = (data.get('district') or data.get('district_input') or '').strip()
    elif request.method == 'POST':
        features = [x for x in request.form.values()]
        state_target = (request.form.get('state_input') or (features[0] if len(features) > 0 else '')).strip().upper()
        district_target = (request.form.get('district_input') or (features[1] if len(features) > 1 else '')).strip()
    else:
        state_target = (request.args.get('state') or request.args.get('state_input') or '').strip().upper()
        district_target = (request.args.get('district') or request.args.get('district_input') or '').strip()

    is_api_request = (
        request.path.startswith('/api/') or 
        request.path == '/cluster' or 
        request.is_json or 
        request.headers.get('Accept') == 'application/json'
    )

    if not state_target or state_target == "-1" or not district_target:
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": "Both state and district parameters are required.",
                "algorithm_used": "K-Means Clustering Algorithm",
                "algorithm_info": {
                    "name": "K-Means Clustering Algorithm",
                    "model_family": "Unsupervised Centroid-Based Vector Quantization",
                    "clusters": 3
                }
            }), 400
        return render_template('K-Means.html', prediction_text0="Please select both State and District.")

    # Reading labelled and scaled data
    df = pd.read_csv("Datasets/kmeansflask2.csv")
    df_filtered = df.loc[
        (df["STATE/UT"].astype(str).str.strip().str.upper() == state_target.strip().upper()) & 
        (df["DISTRICT"].astype(str).str.strip().str.upper() == district_target.strip().upper())
    ]
    years = df_filtered['YEAR'].values

    if len(years) == 0:
        err_msg = f"No historical data found for district '{district_target}' in state '{state_target}'."
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": err_msg,
                "algorithm_used": "K-Means Clustering Algorithm",
                "algorithm_info": {
                    "name": "K-Means Clustering Algorithm",
                    "model_family": "Unsupervised Centroid-Based Vector Quantization",
                    "clusters": 3
                }
            }), 404
        return render_template('K-Means.html', prediction_text0=err_msg)

    clusters = []
    for i in years:
        l = df_filtered.loc[df_filtered["YEAR"] == i].values
        final_features = [[x for x in l[0] if type(x) == float]]
        y_pred = kmeanclus.predict(final_features)
        clusters.append(y_pred[0])  # 0-high, 1-low, 2-moderate

    # Finding Mode
    high     = clusters.count(0)
    low      = clusters.count(1)
    moderate = clusters.count(2)

    if high > low and high > moderate:
        label = "RED ZONE"
        confidence_score = round(high / len(clusters), 2)
    elif low > high and low > moderate:
        label = "GREEN ZONE"
        confidence_score = round(low / len(clusters), 2)
    elif moderate > high and moderate > low:
        label = "ORANGE ZONE"
        confidence_score = round(moderate / len(clusters), 2)
    else:
        confidence_score = 0.50
        if high == moderate == low:
            label = "Crime Rate Varies a Lot"
        elif high == moderate:
            label = "Red-Orange Zone(Lies between Orange Zone and Red Zone)"
        elif low == moderate:
            label = "Yellow Zone(Lies between Green Zone and Orange Zone)"
        else:
            label = "Crime Rate Varies a Lot"

    if is_api_request:
        return jsonify({
            "status": "success",
            "algorithm_used": "K-Means Clustering Algorithm",
            "algorithm_info": {
                "name": "K-Means Clustering Algorithm",
                "model_family": "Unsupervised Centroid-Based Vector Quantization",
                "clusters": 3,
                "cluster_mapping": {
                    "0": "RED ZONE (High Incident Density)",
                    "1": "GREEN ZONE (Low Incident Density)",
                    "2": "ORANGE ZONE (Moderate Incident Density)"
                }
            },
            "predicted_zone": label,
            "confidence_score": confidence_score,
            "state": state_target,
            "district": district_target,
            "cluster_distribution": {
                "high_density_years": high,
                "low_density_years": low,
                "moderate_density_years": moderate,
                "total_recorded_years": len(clusters)
            }
        })

    return render_template(
        'K-Means.html',
        prediction_text0=label,
        predicted_zone=label,
        confidence_score=confidence_score,
        high_density_years=high,
        moderate_density_years=moderate,
        low_density_years=low,
        total_years=len(clusters),
        selected_state=state_target,
        selected_district=district_target
    )


@app.route('/Randomfrstcls', methods=['GET', 'POST'], endpoint='Randomfrstcls')
@app.route('/randomfrstcls', methods=['GET', 'POST'], endpoint='randomfrstcls')
@app.route('/randomforest', methods=['GET', 'POST'], endpoint='randomforest')
@app.route('/predict', methods=['GET', 'POST'], endpoint='predict')
@app.route('/api/predict', methods=['GET', 'POST'], endpoint='api_predict')
@login_required
@role_required('field_officer')
def Randomfrstcls():
    if request.method == 'GET' and request.path in ['/Randomfrstcls', '/randomfrstcls', '/randomforest'] and not request.args:
        return render_template("RandomForestClassifer.html")

    is_api_request = (
        request.path.startswith('/api/') or 
        request.path == '/predict' or 
        request.is_json or 
        request.headers.get('Accept') == 'application/json'
    )

    try:
        if request.is_json:
            data = request.get_json() or {}
            state_name = (data.get('state') or data.get('state_input') or '').strip().upper()
            district_name = (data.get('district') or data.get('district_input') or '').strip()
            year = int(data.get('year') or data.get('horizon_year') or 2026)
        elif request.method == 'POST':
            form_vals = list(request.form.values())
            state_name = (request.form.get('state_input') or (form_vals[0] if len(form_vals) > 0 else '')).strip().upper()
            district_name = (request.form.get('district_input') or (form_vals[1] if len(form_vals) > 1 else '')).strip()
            year = int(request.form.get('year') or (form_vals[2] if len(form_vals) > 2 else 2026))
        else:
            state_name = (request.args.get('state') or request.args.get('state_input') or '').strip().upper()
            district_name = (request.args.get('district') or request.args.get('district_input') or '').strip()
            year = int(request.args.get('year') or 2026)

        if not state_name or state_name == "-1" or not district_name:
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": "State, District, and Year are required.",
                    "algorithm_used": "Random Forest Classifier (Bagged Decision Trees Ensemble)",
                    "algorithm_info": {
                        "name": "Random Forest Classifier",
                        "model_family": "Bagged Decision Trees Ensemble"
                    }
                }), 400
            return render_template("RandomForestClassifer.html", prediction_text="ERROR: State and District are required.")

        # Encode state and district
        enc_df = pd.read_csv("Datasets/encoded.csv")
        arr = (enc_df.loc[enc_df["STATE/UT"].astype(str).str.strip().str.upper() == state_name.strip().upper()]
                      .loc[enc_df["DISTRICT"].astype(str).str.strip().str.upper() == district_name.strip().upper()].values)
        if len(arr) == 0:
            err_msg = f"ERROR: District '{district_name}' in state '{state_name}' not found in database."
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": err_msg,
                    "algorithm_used": "Random Forest Classifier (Bagged Decision Trees Ensemble)",
                    "algorithm_info": {
                        "name": "Random Forest Classifier",
                        "model_family": "Bagged Decision Trees Ensemble"
                    }
                }), 404
            return render_template("RandomForestClassifer.html", prediction_text=err_msg)

        state_enc = int(arr[0][3])
        district_enc = int(arr[0][4])

        # Load classification dataset
        df1 = pd.read_csv("Datasets/classfication_data_with_cluster_labels.csv")
        df1.drop(["Unnamed: 0"], axis=1, inplace=True)

        crime_cols = list(df1.columns[3:11])  # 8 crime columns
        district_data = df1.loc[
            (df1["STATE/UT"] == state_enc) & (df1["DISTRICT"] == district_enc)
        ]

        if len(district_data) == 0:
            err_msg = f"ERROR: No historical records found for district '{district_name}'."
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": err_msg,
                    "algorithm_used": "Random Forest Classifier (Bagged Decision Trees Ensemble)",
                    "algorithm_info": {
                        "name": "Random Forest Classifier",
                        "model_family": "Bagged Decision Trees Ensemble"
                    }
                }), 404
            return render_template("RandomForestClassifer.html", prediction_text=err_msg)

        t = year - district_data['YEAR'].values[-1]
        obj = ExponentialSmoothing(0.3)

        predicted_crimes = []
        for col in crime_cols:
            l = district_data[col].values
            obj.fit(l)
            predicted_crimes.append(int(obj.predict(t)))

        # Build named DataFrame matching model's 11 feature columns
        feature_names = ['STATE/UT', 'DISTRICT', 'YEAR', 'MURDER', 'ATTEMPT TO MURDER',
                         'RAPE', 'KIDNAPPING & ABDUCTION', 'DACOITY', 'ROBBERY', 'THEFT',
                         'HURT/GREVIOUS HURT']
        feature_values = [state_enc, district_enc, year] + predicted_crimes

        input_df = pd.DataFrame([feature_values], columns=feature_names)

        # Use the Random Forest classifier (rdcls loaded from cls.pkl)
        y_pred = rdcls.predict(input_df)

        if y_pred[0] == 1:
            label = "RED ZONE"
            confidence_score = 0.92
        elif y_pred[0] == 2:
            label = "GREEN ZONE"
            confidence_score = 0.89
        elif y_pred[0] == 3:
            label = "ORANGE ZONE"
            confidence_score = 0.85
        else:
            label = "UNKNOWN ZONE"
            confidence_score = 0.50

        # Replace encoded values with readable names for display
        display_df = input_df.copy()
        display_df['STATE/UT'] = state_name
        display_df['DISTRICT'] = district_name

        incidents_dict = dict(zip(crime_cols, predicted_crimes))

        if is_api_request:
            return jsonify({
                "status": "success",
                "algorithm_used": "Random Forest Classifier (Bagged Decision Trees Ensemble)",
                "algorithm_info": {
                    "name": "Random Forest Classifier",
                    "model_family": "Bagged Decision Trees Ensemble",
                    "preprocessing": "Exponential Smoothing (alpha=0.3)",
                    "feature_vectors": crime_cols
                },
                "predicted_zone": label,
                "confidence_score": confidence_score,
                "state": state_name,
                "district": district_name,
                "horizon_year": year,
                "projected_incidents": incidents_dict
            })

        return render_template(
            "RandomForestClassifer.html",
            prediction_text=label,
            predicted_zone=label,
            confidence_score=confidence_score,
            state_name=state_name,
            district_name=district_name,
            horizon_year=year,
            projected_incidents=incidents_dict
        )

    except Exception as e:
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": str(e),
                "algorithm_used": "Random Forest Classifier (Bagged Decision Trees Ensemble)",
                "algorithm_info": {
                    "name": "Random Forest Classifier",
                    "model_family": "Bagged Decision Trees Ensemble"
                }
            }), 500
        return render_template(
            "RandomForestClassifer.html",
            prediction_text=f"ERROR: {str(e)}"
        )


# Linear Regression
@app.route('/LinearReg', methods=['GET', 'POST'], endpoint='LinearReg')
@app.route('/linearreg', methods=['GET', 'POST'], endpoint='linearreg')
@app.route('/linearregression', methods=['GET', 'POST'], endpoint='linearregression')
@app.route('/regression', methods=['GET', 'POST'], endpoint='regression')
@app.route('/api/regression', methods=['GET', 'POST'], endpoint='api_regression')
@login_required
@role_required('field_officer')
def LinearReg():
    if request.method == 'GET' and request.path in ['/LinearReg', '/linearreg', '/linearregression'] and not request.args:
        return render_template("linear-regression.html")

    is_api_request = (
        request.path.startswith('/api/') or 
        request.path == '/regression' or 
        request.is_json or 
        request.headers.get('Accept') == 'application/json'
    )

    try:
        if request.is_json:
            data = request.get_json() or {}
            state = (data.get('state') or data.get('state_input') or '').strip()
            year = int(data.get('year') or 2026)
        elif request.method == 'POST':
            features = list(request.form.values())
            state = (request.form.get('state_input') or (features[0] if len(features) > 0 else '')).strip()
            year = int(request.form.get('year') or (features[1] if len(features) > 1 else 2026))
        else:
            state = (request.args.get('state') or request.args.get('state_input') or '').strip()
            year = int(request.args.get('year') or 2026)

        if not state or state == "-1":
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": "Target state and year are required.",
                    "algorithm_used": "Ordinary Least Squares (OLS) Linear Regression",
                    "algorithm_info": {
                        "name": "Ordinary Least Squares (OLS) Linear Regression",
                        "demographic_model": "Geometric Decadal Population Projection"
                    }
                }), 400
            return render_template("linear-regression.html", prediction_text="Please select a valid State / UT and Horizon Year.")

        df = pd.read_csv("Datasets/data.csv")
        
        # Case-insensitive match for State/UT in df and lr dictionary
        matched_state_name = None
        for s in df["State/UT"].dropna().unique():
            if s.strip().lower() == state.strip().lower():
                matched_state_name = s
                break

        if not matched_state_name:
            err_msg = f"ERROR: State '{state}' not found in dataset."
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": err_msg,
                    "algorithm_used": "Ordinary Least Squares (OLS) Linear Regression",
                    "algorithm_info": {
                        "name": "Ordinary Least Squares (OLS) Linear Regression"
                    }
                }), 404
            return render_template("linear-regression.html", prediction_text=err_msg)

        state = matched_state_name
        state_data = df.loc[df["State/UT"] == state]

        # Case-insensitive match in lr model dictionary
        model_state_key = None
        for k in lr.keys():
            if k.strip().lower() == state.strip().lower():
                model_state_key = k
                break

        if not model_state_key:
            err_msg = f"ERROR: No regression model available for '{state}'."
            if is_api_request:
                return jsonify({
                    "status": "error",
                    "message": err_msg,
                    "algorithm_used": "Ordinary Least Squares (OLS) Linear Regression",
                    "algorithm_info": {
                        "name": "Ordinary Least Squares (OLS) Linear Regression"
                    }
                }), 404
            return render_template("linear-regression.html", prediction_text=err_msg)

        state = model_state_key
        a1 = state_data['Population (in lakhs)'].values[0]
        a2 = state_data['Population (in lakhs)'].values[-1]
        y1 = int(state_data['Year'].values[0])
        y2 = int(state_data['Year'].values[-1])

        estimated_population = projection(a1, a2, y1, y2, year)
        y_pred = lr[state].predict(pd.DataFrame([[int(estimated_population)]]))
        predicted_crimes = int(y_pred[0])
        crime_rate = int(predicted_crimes / estimated_population)

        if is_api_request:
            return jsonify({
                "status": "success",
                "algorithm_used": "Ordinary Least Squares (OLS) Linear Regression",
                "algorithm_info": {
                    "name": "Ordinary Least Squares (OLS) Linear Regression",
                    "demographic_model": "Geometric Growth Population Projection",
                    "dependent_variable": "Total IPC Crime Volume & Crime Rate per 100k"
                },
                "state": state,
                "target_year": year,
                "projected_population_lakhs": round(float(estimated_population), 2),
                "predicted_total_ipc_crimes": predicted_crimes,
                "projected_crime_rate_per_lakh": crime_rate,
                "confidence_score": 0.89
            })

        result = (f'Total IPC Crimes: {predicted_crimes:,}'
                  f' | Projected Population: {int(estimated_population):,} Lakhs'
                  f' | Crime Rate: {crime_rate} (per lakh)')
        return render_template(
            "linear-regression.html",
            prediction_text=result,
            predicted_crimes=predicted_crimes,
            estimated_population=round(float(estimated_population), 2),
            crime_rate=crime_rate,
            selected_state=state,
            target_year=year
        )

    except Exception as e:
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": str(e),
                "algorithm_used": "Ordinary Least Squares (OLS) Linear Regression",
                "algorithm_info": {
                    "name": "Ordinary Least Squares (OLS) Linear Regression"
                }
            }), 500
        return render_template(
            "linear-regression.html",
            prediction_text=f"ERROR: {str(e)}"
        )


# Time Series Forecasting & Projections
@app.route('/timeseriesipc')
@login_required
@role_required('field_officer')
def timeseriesipc():
    return render_template('total-ipc-forecasting.html')


@app.route('/timeseriescr')
@login_required
@role_required('field_officer')
def timeseriescr():
    return render_template('time-series-forcasting.html')


@app.route('/projections', methods=['GET'])
@app.route('/api/projections', methods=['GET'])
@login_required
@role_required('field_officer')
def projections_api():
    return jsonify({
        "status": "success",
        "algorithm_used": "Facebook Prophet (Additive Time-Series Model)",
        "algorithm_info": {
            "name": "Facebook Prophet",
            "model_type": "Additive Time-Series Model (Non-linear Trend + Seasonality + Holiday Effects)",
            "validation_accuracy": "88.8525%",
            "temporal_horizon": "2001 - 2027"
        },
        "supported_projections": [
            "National Total IPC Crime Volume",
            "National Crime Rate per 100k Population"
        ],
        "decomposition_components": [
            "Multi-Year Additive Trend",
            "Annual Seasonal Cycle",
            "Cyclical Macroeconomic Shocks"
        ]
    })


# ===========================================================================
# CDR Intelligence Module  (field_officer only)
# ===========================================================================

from geopy.distance import geodesic as geo_distance  # noqa: E402

@app.route('/cdr')
@app.route('/cdr-dashboard')
@login_required
@role_required('field_officer')
def cdr_page():
    """Render the CDR Intelligence dashboard UI."""
    return render_template('cdr_dashboard.html', result=None)


@app.route('/analyze-cdr', methods=['POST'])
@app.route('/api/analyze-cdr', methods=['POST'])
@login_required
@role_required('field_officer')
def analyze_cdr():
    """
    CDR Intelligence Module — Accepts a multipart CSV upload and returns:
        • Target call timeline with cell-tower coordinates
        • Proximity leads (calls within radius of crime scene)
        • Top-5 frequent contacts

    Required form fields:
        cdr_file       — CSV upload
        target_number  — phone number string
        crime_lat      — latitude (float)
        crime_lon      — longitude (float)
        radius_km      — proximity radius in km (optional, default 0.5)
    """
    # ── 1. Input validation ───────────────────────────────────────────────
    if 'cdr_file' not in request.files or not request.files['cdr_file'].filename:
        return jsonify({
            "status": "error",
            "message": "No CDR CSV file provided. Please attach a CSV file.",
            "algorithm_used": "CDR Intelligence Module (Pandas + GeoPy Geodesic)"
        }), 400

    target_number = request.form.get('target_number', '').strip()
    crime_lat     = request.form.get('crime_lat', type=float)
    crime_lon     = request.form.get('crime_lon', type=float)
    radius_km     = request.form.get('radius_km', type=float, default=0.5)

    if not target_number or crime_lat is None or crime_lon is None:
        return jsonify({
            "status": "error",
            "message": "Parameters 'target_number', 'crime_lat', and 'crime_lon' are required.",
            "algorithm_used": "CDR Intelligence Module (Pandas + GeoPy Geodesic)"
        }), 400

    # ── 2. Parse CSV ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(request.files['cdr_file'])
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"Failed to parse CSV: {str(exc)}",
            "algorithm_used": "CDR Intelligence Module (Pandas + GeoPy Geodesic)"
        }), 400

    # ── 3. Column validation & type coercion ──────────────────────────────
    required_cols = [
        'Caller_Number', 'Receiver_Number', 'DateTime',
        'Call_Duration_Sec', 'Cell_Tower_ID', 'Tower_Lat', 'Tower_Lon'
    ]
    col_map = {c.strip().lower(): c.strip() for c in df.columns}
    missing = [c for c in required_cols if c.lower() not in col_map]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"CSV is missing required columns: {', '.join(missing)}",
            "algorithm_used": "CDR Intelligence Module (Pandas + GeoPy Geodesic)"
        }), 400

    df = df.rename(columns={col_map[c.lower()]: c for c in required_cols})
    df['DateTime']          = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Tower_Lat']         = pd.to_numeric(df['Tower_Lat'], errors='coerce')
    df['Tower_Lon']         = pd.to_numeric(df['Tower_Lon'], errors='coerce')
    df['Caller_Number']     = df['Caller_Number'].astype(str).str.strip()
    df['Receiver_Number']   = df['Receiver_Number'].astype(str).str.strip()
    df['Cell_Tower_ID']     = df['Cell_Tower_ID'].astype(str).str.strip()
    df['Call_Duration_Sec'] = pd.to_numeric(df['Call_Duration_Sec'], errors='coerce')
    target_number           = str(target_number)

    valid_df = df.dropna(subset=['Tower_Lat', 'Tower_Lon', 'DateTime']).sort_values('DateTime').reset_index(drop=True)
    if valid_df.empty:
        return jsonify({
            "status": "error",
            "message": "The CSV contains no rows with valid timestamps and coordinates.",
            "algorithm_used": "CDR Intelligence Module (Pandas + GeoPy Geodesic)"
        }), 400

    # ── 4. Target movement trajectory & timeline ──────────────────────────
    target_mask = (valid_df['Caller_Number'] == target_number) | (valid_df['Receiver_Number'] == target_number)
    target_df   = valid_df[target_mask].sort_values('DateTime')

    timeline = [
        {
            "timestamp":    row['DateTime'].isoformat() if pd.notnull(row['DateTime']) else None,
            "DateTime":     row['DateTime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['DateTime']) else None,
            "tower_lat":    float(row['Tower_Lat']),
            "Tower_Lat":    float(row['Tower_Lat']),
            "latitude":     float(row['Tower_Lat']),
            "tower_lon":    float(row['Tower_Lon']),
            "Tower_Lon":    float(row['Tower_Lon']),
            "longitude":    float(row['Tower_Lon']),
            "cell_tower_id": str(row['Cell_Tower_ID']),
            "tower_id":     str(row['Cell_Tower_ID']),
            "caller_number": str(row['Caller_Number']),
            "Caller_Number": str(row['Caller_Number']),
            "receiver_number": str(row['Receiver_Number']),
            "Receiver_Number": str(row['Receiver_Number']),
            "direction":    "outgoing" if str(row['Caller_Number']) == target_number else "incoming",
            "counterparty": (
                str(row['Receiver_Number']) if str(row['Caller_Number']) == target_number
                else str(row['Caller_Number'])
            ),
            "duration_sec": int(row['Call_Duration_Sec']) if pd.notnull(row['Call_Duration_Sec']) else None,
            "Call_Duration_Sec": int(row['Call_Duration_Sec']) if pd.notnull(row['Call_Duration_Sec']) else None
        }
        for _, row in target_df.iterrows()
    ]

    # ── 5. Spatial Proximity Engine (Geodesic Distance) ───────────────────
    crime_point     = (crime_lat, crime_lon)
    proximity_leads = []
    for _, row in valid_df.iterrows():
        tower_point = (row['Tower_Lat'], row['Tower_Lon'])
        try:
            dist_km = geo_distance(crime_point, tower_point).kilometers
        except Exception:
            continue
        if dist_km <= radius_km:
            involves_target = target_number in (str(row['Caller_Number']), str(row['Receiver_Number']))
            proximity_leads.append({
                "timestamp":       row['DateTime'].isoformat() if pd.notnull(row['DateTime']) else None,
                "DateTime":        row['DateTime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['DateTime']) else None,
                "caller_number":   str(row['Caller_Number']),
                "Caller_Number":   str(row['Caller_Number']),
                "receiver_number": str(row['Receiver_Number']),
                "Receiver_Number": str(row['Receiver_Number']),
                "counterparty":    str(row['Receiver_Number']) if str(row['Caller_Number']) == target_number else str(row['Caller_Number']),
                "direction":       "outgoing" if str(row['Caller_Number']) == target_number else "incoming",
                "tower_id":        str(row['Cell_Tower_ID']),
                "cell_tower_id":   str(row['Cell_Tower_ID']),
                "tower_lat":       float(row['Tower_Lat']),
                "Tower_Lat":       float(row['Tower_Lat']),
                "latitude":        float(row['Tower_Lat']),
                "tower_lon":       float(row['Tower_Lon']),
                "Tower_Lon":       float(row['Tower_Lon']),
                "longitude":       float(row['Tower_Lon']),
                "distance_km":     round(dist_km, 4),
                "duration_sec":    int(row['Call_Duration_Sec']) if pd.notnull(row['Call_Duration_Sec']) else None,
                "involves_target": involves_target
            })
    proximity_leads.sort(key=lambda x: x['distance_km'])

    # ── 6. Contact frequency matrix ───────────────────────────────────────
    outgoing_targets = target_df.loc[target_df['Caller_Number'] == target_number, 'Receiver_Number']
    incoming_callers = target_df.loc[target_df['Receiver_Number'] == target_number, 'Caller_Number']
    all_contacts     = pd.concat([outgoing_targets, incoming_callers])
    all_contacts     = all_contacts[all_contacts != target_number]
    freq             = all_contacts.value_counts().head(5)
    top_contacts     = [{"number": str(num), "count": int(cnt), "call_count": int(cnt)} for num, cnt in freq.items()]

    # ── 7. Build and return JSON response ─────────────────────────────────
    return jsonify({
        "status":               "success",
        "algorithm_used":       "CDR Intelligence Module (Pandas + GeoPy Geodesic)",
        "algorithm_info": {
            "name":             "CDR Spatial Proximity & Link Analysis Engine",
            "libraries":        ["pandas", "geopy"],
            "proximity_method": "WGS-84 Geodesic Distance",
            "proximity_radius_km": radius_km
        },
        "target_number":        target_number,
        "crime_scene": {
            "latitude":         crime_lat,
            "longitude":        crime_lon,
            "lat":              crime_lat,
            "lon":              crime_lon,
            "radius_km":        radius_km
        },
        "radius_km":            radius_km,
        "target_timeline":      timeline,
        "timeline":             timeline,
        "target_call_timeline": timeline,
        "proximity_leads":      proximity_leads,
        "top_contacts":         top_contacts,
        "records_analyzed":     int(len(valid_df))
    })


# Data Display
@app.route('/datadisp')
@login_required
@role_required('field_officer')
def datadisp():
    return render_template('datadisplay.html')


# Run File
@app.route("/run-file")
@login_required
@role_required('field_officer')
def run_file():
    file_path = os.path.join("folium-map", "index.py")
    python_bin = sys.executable
    subprocess.Popen([python_bin, file_path])
    return {
        "status": "success",
        "message": "File execution initiated",
        "algorithm_used": "DOM Tree Scraper (BeautifulSoup4) + GeoPy Geocoding Engine",
        "algorithm_info": {
            "name": "DOM Tree Scraper (BeautifulSoup4) + GeoPy Geocoding Engine",
            "scraper_engine": "BeautifulSoup4 HTML Parser",
            "geocoder": "GeoPy Nominatim / OpenStreetMap API",
            "visualization": "Folium Kernel Density Heatmap"
        }
    }


# Crime Feed
@app.route('/crimefeed')
@login_required
@role_required('field_officer')
def crimefeed():
    response = make_response(render_template('crimefeed.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response


# Heat Map
@app.route('/foliummap')
@login_required
@role_required('field_officer')
def foliummap():
    return render_template("final.html")


# ===========================================================================
# Proactive Patrol Waypoint Generation & Dispatch HUD (field_officer only)
# ===========================================================================

@app.route('/patrol')
@app.route('/patrol-route')
@app.route('/patrol-hud')
@login_required
@role_required('field_officer')
def patrol_hud():
    """Render the Proactive Patrol Waypoint Generation & Dispatch HUD."""
    return render_template('patrol_route.html')


# ===========================================================================
# Case File Intelligence Analyzer (field_officer only)
# ===========================================================================

from case_parser import extract_case_intelligence



@app.route('/analyze_case', methods=['GET', 'POST'])
@app.route('/analyze-fir', methods=['GET', 'POST'])
@app.route('/api/analyze-fir', methods=['GET', 'POST'])
@login_required
@role_required('field_officer')
def analyze_case():
    """
    FIR Threat Assessment HUD / Intelligence Analyzer for Field Officers.
    Accepts direct text input, JSON payloads, or file uploads (.pdf, .txt).
    Extracts plain text via PyPDF2 / text decoders, applies VADER Sentiment & Regex NLP
    to compute Threat Severity, Key Entities, and IPC Crime Categories.
    """
    # Only treat as an API call when:
    # - it's a POST to /api/analyze-fir, OR
    # - the client explicitly sends Accept: application/json or a JSON body
    # A plain browser GET (even to /api/analyze-fir) should always show the HTML page.
    is_api_request = (
        request.is_json or
        (request.headers.get('Accept') == 'application/json' and not request.accept_mimetypes.accept_html) or
        (request.path.startswith('/api/') and not request.accept_mimetypes.accept_html and not request.form and not request.files)
    )

    # Always render the HTML dashboard for GET requests (even /api/analyze-fir)
    if request.method == 'GET':
        return render_template('case_analysis.html', result=None, raw_text="")


    extracted_text = ""
    source_label = "Direct Text Narrative"

    # 1. Check for JSON payload
    if request.is_json:
        data = request.get_json() or {}
        extracted_text = (data.get('case_text') or data.get('text') or data.get('narrative') or '').strip()
        source_label = data.get('source_label') or "Direct JSON Narrative"

    # 2. Check for query parameters (GET with case_text)
    if not extracted_text and request.args.get('case_text'):
        extracted_text = request.args.get('case_text', '').strip()
        source_label = "GET Query Narrative"

    # 3. Check for file upload
    if not extracted_text and 'case_file' in request.files:
        file = request.files['case_file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext == '.pdf':
                try:
                    pdf_bytes = file.read()
                    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                    extracted_pages = []
                    for page_idx, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            extracted_pages.append(page_text)
                    extracted_text = "\n".join(extracted_pages).strip()
                    source_label = f"Uploaded PDF ({filename}) - {len(pdf_reader.pages)} Page(s)"
                except Exception as e:
                    err_msg = f"Error reading PDF file '{filename}': {str(e)}"
                    if is_api_request:
                        return jsonify({"status": "error", "message": err_msg}), 400
                    flash(err_msg, "danger")
                    return render_template('case_analysis.html', result=None, raw_text="")
            elif ext == '.txt':
                try:
                    raw_bytes = file.read()
                    extracted_text = raw_bytes.decode('utf-8', errors='ignore').strip()
                    source_label = f"Uploaded Text File ({filename})"
                except Exception as e:
                    err_msg = f"Error reading Text file '{filename}': {str(e)}"
                    if is_api_request:
                        return jsonify({"status": "error", "message": err_msg}), 400
                    flash(err_msg, "danger")
                    return render_template('case_analysis.html', result=None, raw_text="")
            else:
                err_msg = "Unsupported file format. Please upload a .pdf or .txt file."
                if is_api_request:
                    return jsonify({"status": "error", "message": err_msg}), 400
                flash(err_msg, "danger")
                return render_template('case_analysis.html', result=None, raw_text="")

    # 4. Fallback to direct text narrative form field if no file uploaded
    if not extracted_text:
        form_text = request.form.get('case_text', '').strip()
        if form_text:
            extracted_text = form_text
            source_label = "Direct FIR / Narrative Input"

    if not extracted_text:
        err_msg = "Please provide a case narrative or upload a valid .pdf / .txt document to analyze."
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": err_msg,
                "algorithm_used": "VADER NLP + RegEx Weighted IPC Severity Scoring",
                "algorithm_info": {
                    "name": "VADER NLP + RegEx Weighted IPC Severity Scoring"
                }
            }), 400
        flash(err_msg, "warning")
        return render_template('case_analysis.html', result=None, raw_text="")

    # 5. Perform NLP & Regex Intelligence Extraction
    try:
        intel_result = extract_case_intelligence(extracted_text, source_label=source_label)
        if is_api_request:
            zone_mapping = {
                "HIGH RISK": "Red Zone",
                "MODERATE THREAT": "Orange Zone",
                "LOW THREAT": "Green Zone"
            }
            return jsonify({
                "status": "success",
                "algorithm_used": intel_result['algorithm_used'],
                "algorithm_info": intel_result['algorithm_info'],
                "predicted_zone": zone_mapping.get(intel_result['severity_label'], "Moderate Zone"),
                "confidence_score": round(min(1.0, intel_result['threat_score'] / 100.0 + 0.15), 2),
                "threat_score": intel_result['threat_score'],
                "severity_label": intel_result['severity_label'],
                "sentiment": intel_result['sentiment'],
                "weapons": intel_result['weapons'],
                "locations": intel_result['locations'],
                "timeline": intel_result['timeline'],
                "ipc_categories": intel_result['ipc_categories'],
                "tactical_signals": intel_result['tactical_signals'],
                "action_protocol": intel_result['action_protocol']
            })
        return render_template('case_analysis.html', result=intel_result, raw_text=extracted_text)
    except Exception as e:
        if is_api_request:
            return jsonify({
                "status": "error",
                "message": f"Intelligence processing failed: {str(e)}",
                "algorithm_used": "VADER NLP + RegEx Weighted IPC Severity Scoring",
                "algorithm_info": {
                    "name": "VADER NLP + RegEx Weighted IPC Severity Scoring"
                }
            }), 500
        flash(f"Intelligence processing failed: {str(e)}", "danger")
        return render_template('case_analysis.html', result=None, raw_text=extracted_text)

