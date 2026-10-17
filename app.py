from flask import Flask, render_template, request, url_for, jsonify, abort
from markupsafe import Markup
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from functools import wraps
from spatial_beat_engine import generate_patrol_route

# Initialize the Flask App
app = Flask(__name__)

# --- Security & Database Configuration ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'crime-analysis-dev-secret-key-change-in-prod')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Extension Initialization ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# --- Canonical User Role Definitions ---
ROLE_PUBLIC = 'PUBLIC'
ROLE_FIELD_OFFICER = 'FIELD_OFFICER'
ROLE_SP_ADMIN = 'SP_ADMIN'


class RoleString(str):
    """Case-insensitive string wrapper ensuring backwards-compatibility with legacy lowercase checks."""
    def __eq__(self, other):
        if isinstance(other, str):
            return self.lower() == other.lower()
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.lower())


class RoleType(db.TypeDecorator):
    """SQLAlchemy type decorator that returns RoleString for case-insensitive role checks."""
    impl = db.String(20)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value).upper()
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return RoleString(value)
        return value


# --- User Model ---
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id               = db.Column(db.Integer, primary_key=True)
    username         = db.Column(db.String(80), unique=True, nullable=False)
    password_hash    = db.Column(db.String(256), nullable=False)
    role             = db.Column(RoleType, nullable=False, default='PUBLIC')
    # Roles: 'PUBLIC' | 'FIELD_OFFICER' | 'SP_ADMIN'
    assigned_district = db.Column(db.String(100), nullable=True)

    @property
    def normalized_role(self) -> str:
        return (self.role or '').upper().strip()

    def has_role(self, *roles) -> bool:
        current = self.normalized_role
        return current in [r.upper().strip() for r in roles]

    @property
    def is_field_officer(self) -> bool:
        return self.has_role(ROLE_FIELD_OFFICER)

    @property
    def is_sp_admin(self) -> bool:
        return self.has_role(ROLE_SP_ADMIN)

    def set_password(self, password: str) -> None:
        """Hash and store the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Return True if the given password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username!r} role={self.role!r}>'


# --- Feedback & Debriefing Model ---
class Feedback(db.Model):
    __tablename__ = 'feedbacks'

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    name           = db.Column(db.String(120), nullable=False)
    email          = db.Column(db.String(120), nullable=False)
    role           = db.Column(db.String(40), nullable=True, default='PUBLIC')
    category       = db.Column(db.String(80), nullable=False, default='General Feedback')
    subject        = db.Column(db.String(200), nullable=False)
    police_station = db.Column(db.String(120), nullable=True, default='General / All Stations')
    priority       = db.Column(db.String(30), nullable=False, default='Routine')
    rating         = db.Column(db.Integer, nullable=False, default=5)
    message        = db.Column(db.Text, nullable=False)
    created_at     = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status         = db.Column(db.String(40), nullable=False, default='Transmitted')

    user           = db.relationship('User', backref=db.backref('feedbacks', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'category': self.category,
            'subject': self.subject,
            'police_station': self.police_station or 'General / All Stations',
            'priority': self.priority,
            'rating': self.rating,
            'message': self.message,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'status': self.status
        }

    def __repr__(self):
        return f'<Feedback #{self.id} from {self.name!r} [{self.category}]>'


@login_manager.user_loader
def load_user(user_id: str):
    """Flask-Login callback: load user by primary key."""
    return db.session.get(User, int(user_id))


# --- Database Initialization & Seeding ---
def init_db():
    """Create tables and seed default accounts if needed."""
    with app.app_context():
        db.create_all()

        seed_accounts = [
            {
                'username':          'public_user',
                'password':          'public123',
                'role':              'PUBLIC',
                'assigned_district': None,
            },
            {
                'username':          'field_officer',
                'password':          'officer123',
                'role':              'FIELD_OFFICER',
                'assigned_district': 'BANGALORE COMMR.',
            },
            {
                'username':          'sp_admin',
                'password':          'admin123',
                'role':              'SP_ADMIN',
                'assigned_district': 'STATE POLICE HQ',
            },
        ]

        for account in seed_accounts:
            existing = User.query.filter_by(username=account['username']).first()
            if not existing:
                user = User(
                    username=account['username'],
                    role=account['role'],
                    assigned_district=account['assigned_district'],
                )
                user.set_password(account['password'])
                db.session.add(user)
            else:
                # Sync canonical role and assigned district
                existing.role = account['role']
                if account['assigned_district'] and not existing.assigned_district:
                    existing.assigned_district = account['assigned_district']

        db.session.commit()
        print('[init_db] Verified default user accounts.')

        # Seed initial debriefing/feedback records if empty
        try:
            if not Feedback.query.first():
                sample_feedbacks = [
                    Feedback(
                        name="Inspector R. Kumar",
                        email="officer_kumar@ksp.gov.in",
                        role="FIELD_OFFICER",
                        category="Beat Patrol Route & Waypoint Assessment",
                        subject="Night Patrol Frequency in Koramangala 5th Block",
                        police_station="Koramangala Police Station",
                        priority="Urgent",
                        rating=5,
                        message="K-Means waypoint clustering for night shifts accurately identifies high-footfall bar cluster zones. Recommend adding static surveillance near 80ft Road cross-junction.",
                        created_at=datetime.utcnow(),
                        status="Reviewed"
                    ),
                    Feedback(
                        name="Citizen Safety Forum",
                        email="citizen_forum@bengaluru.org",
                        role="PUBLIC",
                        category="Crime Tip / Field Observation",
                        subject="Inoperative Streetlights near Indiranagar Metro Backlane",
                        police_station="Indiranagar Police Station",
                        priority="Routine",
                        rating=4,
                        message="Several streetlights are non-functional on 100ft road service lane. Increased foot patrol between 10 PM and 1 AM would significantly deter bike-borne mobile snatching.",
                        created_at=datetime.utcnow(),
                        status="Investigating"
                    )
                ]
                db.session.add_all(sample_feedbacks)
                db.session.commit()
                print('[init_db] Verified default feedback records.')
        except Exception as fb_err:
            db.session.rollback()
            print(f'[init_db] Note on feedback seeding: {fb_err}')


# Seed on startup before routes are registered
init_db()


# --- RBAC Role Verification Helpers ---
def verify_user_role(role_name: str, user=None) -> bool:
    """
    Verify if target user (or current_user) holds the specified role (case-insensitive).
    """
    u = user or current_user
    if not u or not getattr(u, 'is_authenticated', False):
        return False
    current = (getattr(u, 'role', '') or '').upper().strip()
    return current == role_name.upper().strip()


def role_required_api(*allowed_roles):
    """
    Decorator to restrict API endpoints to authenticated users with allowed roles.
    Supports current_user authentication and explicit X-User-Role / role payload for testing.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            req_data = request.get_json(silent=True) or {}
            explicit_role = request.headers.get('X-User-Role') or req_data.get('role') or request.args.get('role')
            if explicit_role:
                norm = explicit_role.upper().strip()
                if norm in [r.upper().strip() for r in allowed_roles]:
                    return f(*args, **kwargs)
            if current_user.is_authenticated:
                norm = (current_user.role or '').upper().strip()
                if norm in [r.upper().strip() for r in allowed_roles]:
                    return f(*args, **kwargs)
            return jsonify({'status': 'error', 'message': f'Unauthorized. Required role: {list(allowed_roles)}'}), 403
        return decorated_function
    return decorator

# --- Modus Operandi Vector Search Engine ---
_mo_engine = None


def get_mo_engine():
    """Lazy-load the MOVectorEngine singleton."""
    global _mo_engine
    if _mo_engine is None:
        from mo_vector_service import MOVectorEngine
        _mo_engine = MOVectorEngine()
    return _mo_engine


@app.route('/api/search-mo', methods=['POST'])
def search_mo():
    """
    Search for cases with similar Modus Operandi using dense vector embeddings.
    Accepts 'query' or 'query_text' and optional 'top_k' in JSON body or form data.
    """
    data = request.get_json(silent=True) or {}
    query_text = (
        data.get('query')
        or data.get('query_text')
        or request.form.get('query')
        or request.form.get('query_text')
        or request.args.get('query')
        or ''
    ).strip()

    if not query_text:
        return jsonify({
            'status': 'error',
            'message': 'Query string is required (use "query" or "query_text").'
        }), 400

    try:
        top_k = int(data.get('top_k') or request.form.get('top_k') or request.args.get('top_k') or 5)
    except (ValueError, TypeError):
        top_k = 5

    try:
        engine = get_mo_engine()
        results = engine.search_similar_cases(query_text=query_text, top_k=top_k)
        return jsonify({
            'status': 'success',
            'query': query_text,
            'top_k': top_k,
            'count': len(results),
            'results': results
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# --- Spatio-Temporal Beat Engine Route ---
@app.route('/api/patrol-route', methods=['POST'])
def api_patrol_route():
    """
    Generate an optimized, K-Means clustered high-risk patrol route for a police beat.
    Payload: { "beat_id": "BEAT_101", "shift_window": "Night", "num_waypoints": 5 }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        data = request.form.to_dict() if request.form else {}

    # Clean validation for required beat_id
    raw_beat_id = data.get('beat_id')
    if not raw_beat_id or not str(raw_beat_id).strip():
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: beat_id',
            'message': 'A valid beat_id must be provided in the request body.'
        }), 400

    beat_id = str(raw_beat_id).strip()
    shift_window = str(data.get('shift_window') or data.get('shift') or 'Night').strip()

    try:
        num_waypoints = int(data.get('num_waypoints') or 5)
        if num_waypoints < 1:
            num_waypoints = 5
    except (ValueError, TypeError):
        num_waypoints = 5

    try:
        waypoints = generate_patrol_route(
            beat_id=beat_id,
            shift_window=shift_window,
            num_waypoints=num_waypoints
        )

        if not waypoints:
            return jsonify({
                'status': 'error',
                'error': 'Insufficient coordinate data',
                'message': f"No coordinate data available to generate patrol route for beat '{beat_id}'."
            }), 400

        return jsonify({
            'status': 'success',
            'beat_id': beat_id,
            'shift_window': shift_window,
            'shift': shift_window.lower(),
            'algorithm_used': 'K-Means Spatial Centroids & Nearest-Neighbor TSP Sequencing',
            'total_waypoints': len(waypoints),
            'waypoints': waypoints
        }), 200
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'error': 'Invalid Request or Insufficient Coordinate Data',
            'message': str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'Processing Error',
            'message': str(e)
        }), 500


# --- Generative Case Briefing Engine Route (Week 4) ---
@app.route('/api/case-briefing', methods=['POST'])
def api_case_briefing():
    """
    Generate structured case briefing dossier from FIR narrative and entities.
    Returns: Chronological Timeline, Missing Evidence Gaps, and Suggested Interrogation Questions.
    Payload: { "case_narrative": "...", "entities": { ... } }
    """
    data = request.get_json(silent=True) or request.form or {}
    narrative = data.get('case_narrative') or data.get('case_text') or data.get('narrative') or ''
    entities = data.get('entities') or data.get('entities_dict') or None

    if not narrative.strip():
        return jsonify({'status': 'error', 'message': 'case_narrative is required.'}), 400

    try:
        from case_briefing_copilot import generate_case_brief
        brief = generate_case_brief(case_narrative=narrative, entities_dict=entities)
        return jsonify({
            'status': 'success',
            'case_brief': brief
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- Role-Based Access Control & Dashboard Routing (Week 5) ---
@app.route('/api/role-dashboard', methods=['GET', 'POST'])
@app.route('/api/dashboard-payload', methods=['GET', 'POST'])
def api_role_dashboard():
    """
    Expose role-aware API payloads:
    - 'FIELD_OFFICER': Receives active beat alerts and tactical patrol route waypoints.
    - 'SP_ADMIN': Receives macro-level crime trend analytics, resource allocation, and strategic advisories.
    """
    data = request.get_json(silent=True) or request.form or {}
    requested_role = (
        request.headers.get('X-User-Role')
        or data.get('role')
        or request.args.get('role')
        or (current_user.role if current_user.is_authenticated else None)
        or ''
    ).upper().strip()

    if not requested_role:
        return jsonify({
            'status': 'error',
            'message': 'Authentication or explicit role is required (use header X-User-Role, query ?role=, or login).'
        }), 401

    if requested_role == ROLE_FIELD_OFFICER:
        beat_id = data.get('beat_id') or request.args.get('beat_id') or 'BEAT_101'
        shift = data.get('shift') or request.args.get('shift') or 'night'
        from spatial_beat_engine import SpatialBeatEngine
        engine = SpatialBeatEngine()
        payload = engine.get_field_officer_payload(beat_id=beat_id, shift=shift)
        return jsonify({
            'status': 'success',
            'role': ROLE_FIELD_OFFICER,
            'dashboard_type': 'OPERATIONAL_TACTICAL_FIELD',
            'payload': payload
        }), 200

    elif requested_role == ROLE_SP_ADMIN:
        state = data.get('state') or request.args.get('state') or 'KARNATAKA'
        from spatial_beat_engine import SpatialBeatEngine
        engine = SpatialBeatEngine()
        payload = engine.get_sp_admin_payload(state=state)
        return jsonify({
            'status': 'success',
            'role': ROLE_SP_ADMIN,
            'dashboard_type': 'STRATEGIC_MACRO_ANALYTICS',
            'payload': payload
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f"Access forbidden for role '{requested_role}'. Must be 'FIELD_OFFICER' or 'SP_ADMIN'."
        }), 403


# --- Feedback & Debriefing REST API Route ---
@app.route('/api/feedback', methods=['GET', 'POST'])
def api_feedback():
    """
    API endpoint to submit or retrieve intelligence feedback records.
    GET: Returns list of recent debrief records.
    POST: Creates a new feedback entry.
    """
    if request.method == 'POST':
        req_data = request.get_json(silent=True)
        if not req_data or not isinstance(req_data, dict):
            req_data = request.form.to_dict() if request.form else {}

        message = (req_data.get('message') or req_data.get('feedback') or req_data.get('comments') or '').strip()
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Feedback message / observation is required.'
            }), 400

        name = (req_data.get('name') or (current_user.username if current_user.is_authenticated else 'Anonymous')).strip()
        email = (req_data.get('email') or f"{name.lower().replace(' ', '_')}@ksp.gov.in").strip()
        category = (req_data.get('category') or 'General Feedback').strip()
        subject = (req_data.get('subject') or 'Field Observation Debrief').strip()
        police_station = (req_data.get('police_station') or req_data.get('station') or 'General / All Stations').strip()
        priority = (req_data.get('priority') or 'Routine').strip()
        try:
            rating = int(req_data.get('rating') or 5)
            if rating < 1 or rating > 5:
                rating = 5
        except (ValueError, TypeError):
            rating = 5

        fb = Feedback(
            user_id=current_user.id if current_user.is_authenticated else None,
            name=name,
            email=email,
            role=str(getattr(current_user, 'role', 'PUBLIC')) if current_user.is_authenticated else 'PUBLIC',
            category=category,
            subject=subject,
            police_station=police_station,
            priority=priority,
            rating=rating,
            message=message,
            created_at=datetime.utcnow(),
            status='Transmitted'
        )

        try:
            db.session.add(fb)
            db.session.commit()
            return jsonify({
                'status': 'success',
                'message': f'Feedback registered successfully. Transmission ID: #FB-{fb.id:04d}',
                'feedback_id': fb.id,
                'data': fb.to_dict()
            }), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # GET
    try:
        limit = int(request.args.get('limit') or 20)
    except (ValueError, TypeError):
        limit = 20
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).limit(limit).all()
    return jsonify({
        'status': 'success',
        'count': len(feedbacks),
        'feedbacks': [f.to_dict() for f in feedbacks]
    }), 200


from routes import *

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)