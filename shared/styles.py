import streamlit as st

def apply_global_style():
    """
    Applies the professional AegisLife 'Modern Clinical Slate' theme 
    and smooth fade-in animations to any page.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* 1. Global Typography & Background */
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    .stApp { 
        background-color: #F8FAFC; 
        animation: fadeIn 0.5s ease-in; /* Smooth Fade-In Animation */
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(5px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* 2. Professional Dark Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #0A1628 !important; 
        border-right: 1px solid #1E293B; 
    }
    [data-testid="stSidebar"] * { 
        color: #F8FAFC !important; 
    }

    /* 3. Sidebar Navigation 'Button' Tiles */
    [data-testid="stSidebarNav"] { padding-top: 1.5rem; }
    
    div[data-testid="stSidebarNav"] li {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin: 8px 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Hover Effect */
    div[data-testid="stSidebarNav"] li:hover {
        background: rgba(37, 99, 235, 0.2) !important;
        border-color: #3B82F6 !important;
        transform: translateX(6px) !important;
    }

    /* Active Page Highlight (Blue Gradient) */
    div[data-testid="stSidebarNav"] li:has(a[aria-current="page"]) {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    }

    /* 4. Hide Default Streamlit Elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)