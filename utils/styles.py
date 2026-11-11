"""
Pink Aesthetic Design System & CSS Styling
Theme: Modern, Feminine, Clean, and Professional SaaS Analytics Dashboard
Refined Color Palette for High Contrast and Maximum Readability
"""

COLORS = {
    "primary": "#E879A8",
    "primary_dark": "#C94F83",
    "soft_pink": "#FCE7F0",
    "light_pink": "#FFF5F8",
    "background": "#FFF9FB",
    "white": "#FFFFFF",
    "text_primary": "#352B32",
    "text_secondary": "#6F5965",
    "muted_text": "#806D78",
    "border": "#EBD5DF",
    "success": "#3D8563",
    "error": "#B42345",
    "adaboost": "#E9A1BD",
    "grid": "#F0E5EA",
    "card_shadow": "0 8px 24px rgba(201, 79, 131, 0.08)",
}

CUSTOM_CSS = """
<style>
/* Import Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #352B32 !important;
}

/* App Background */
.stApp {
    background-color: #FFF9FB !important;
}

/* Main Container Spacing */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1200px !important;
}

/* Headings Contrast */
h1, h2, h3, h4, h5, h6 {
    color: #352B32 !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #FFF5F8 !important;
    border-right: 1px solid #EBD5DF !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.8rem !important;
}

/* Sidebar Navigation Items */
div[role="radiogroup"] > label {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD5DF !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin-bottom: 8px !important;
    transition: all 0.2s ease !important;
}

div[role="radiogroup"] > label:hover {
    background-color: #FCE7F0 !important;
    border-color: #E879A8 !important;
}

/* Text inside radio navigation */
div[role="radiogroup"] label p {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #352B32 !important;
}

/* Selected Active Navigation */
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {
    background-color: #FCE7F0 !important;
    border: 1.5px solid #C94F83 !important;
}

div[role="radiogroup"] label[data-checked="true"] p,
div[role="radiogroup"] label:has(input:checked) p {
    color: #C94F83 !important;
    font-weight: 700 !important;
}

/* Custom Card Container */
.pink-card {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD5DF !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 24px rgba(201, 79, 131, 0.06) !important;
    margin-bottom: 20px !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F8 100%) !important;
    border: 1.5px solid #EBD5DF !important;
    border-radius: 16px !important;
    padding: 20px 14px !important;
    box-shadow: 0 4px 16px rgba(201, 79, 131, 0.07) !important;
    text-align: center !important;
}

.metric-card-title {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #6F5965 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 6px !important;
}

.metric-card-value {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #C94F83 !important;
    margin-bottom: 4px !important;
    line-height: 1.1 !important;
}

.metric-card-subtitle {
    font-size: 0.78rem !important;
    color: #806D78 !important;
    font-weight: 500 !important;
}

/* Badges */
.badge-winner {
    background-color: #FCE7F0 !important;
    color: #C94F83 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 5px 12px !important;
    border-radius: 20px !important;
    display: inline-block !important;
    border: 1px solid #E879A8 !important;
}

.badge-info {
    background-color: #EBF5F0 !important;
    color: #3D8563 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 5px 12px !important;
    border-radius: 20px !important;
    display: inline-block !important;
    border: 1px solid #C4E3D4 !important;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #FFF5F8 0%, #FCE7F0 65%, #F8D7E3 100%) !important;
    border: 1px solid #EBD5DF !important;
    border-radius: 20px !important;
    padding: 32px 36px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 10px 30px rgba(201, 79, 131, 0.08) !important;
}

.hero-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #352B32 !important;
    margin-bottom: 6px !important;
    letter-spacing: -0.025em !important;
}

.hero-title span {
    color: #C94F83 !important;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: #6F5965 !important;
    margin-bottom: 12px !important;
    font-weight: 600 !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #E879A8 0%, #C94F83 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(201, 79, 131, 0.28) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ED8CB6 0%, #D95D91 100%) !important;
    box-shadow: 0 6px 18px rgba(201, 79, 131, 0.38) !important;
    transform: translateY(-1px) !important;
    color: #FFFFFF !important;
}

/* Secondary Download Button */
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #C94F83 !important;
    border: 1.5px solid #C94F83 !important;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background-color: #FCE7F0 !important;
    color: #C94F83 !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #E879A8 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background-image: linear-gradient(to bottom, #FFF9FB, #FFFFFF) !important;
}

/* Table and DataFrame Styling (Light Theme) */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD5DF !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(201, 79, 131, 0.04) !important;
}

/* Alert Boxes / Info Callouts */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid #EBD5DF !important;
    background-color: #FFF5F8 !important;
    color: #352B32 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    border-bottom: 1.5px solid #EBD5DF !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    padding: 9px 22px !important;
    font-weight: 600 !important;
    color: #6F5965 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #FCE7F0 !important;
    color: #C94F83 !important;
    border-bottom: 2px solid #C94F83 !important;
    font-weight: 700 !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD5DF !important;
    border-radius: 14px !important;
}

/* Divider */
.pink-divider {
    height: 1px;
    background: linear-gradient(90deg, #EBD5DF 0%, #E879A8 50%, #EBD5DF 100%);
    margin: 24px 0;
    border: none;
}
</style>
"""
