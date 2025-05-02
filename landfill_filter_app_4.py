# Add this code at the top of your landfill_filter_app_4.py file

import streamlit as st

# Password Protection Function
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "GenLFG25":  # Replace with your desired password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.title("Galactic Landfill Explorer")
        st.write("Please enter the password to access this application.")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.title("Galactic Landfill Explorer")
        st.write("Please enter the password to access this application.")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Use the check_password function right at the beginning of your app
if not check_password():
    st.stop()  # Stop execution here if password is incorrect

import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from scipy.spatial import cKDTree
import time
import random
import hashlib

# Set page config first
st.set_page_config(
    page_title="Galactic Landfill Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS to improve performance
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stDataFrame {max-height: 400px;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.map_created = False
    st.session_state.filter_hash = ""
    st.session_state.need_map_update = True

# Function definitions
def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between points on earth."""
    R = 6371.0  # Radius of earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def assign_nearest_city_fast(landfill_df, cities_df):
    """Efficiently find nearest city for each landfill using KDTree."""
    if landfill_df.empty or cities_df.empty:
        return landfill_df
    
    try:
        # Convert to float first to ensure proper calculation
        landfill_coords = np.radians(landfill_df[["Latitude", "Longitude"]].astype(float).values)
        city_coords = np.radians(cities_df[["Latitude", "Longitude"]].astype(float).values)
        
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(city_coords)
        _, idxs = tree.query(landfill_coords, k=1)
        
        # Match cities to landfills
        matched = cities_df.iloc[idxs].reset_index(drop=True)
        landfill_df = landfill_df.copy()  # Create copy to avoid SettingWithCopyWarning
        landfill_df["Nearest City"] = matched["City"].values
        landfill_df["Nearest State"] = matched["State"].values
        
        # Calculate distances
        landfill_df["Distance to City (km)"] = haversine_np(
            landfill_df["Longitude"].astype(float), 
            landfill_df["Latitude"].astype(float),
            matched["Longitude"].astype(float), 
            matched["Latitude"].astype(float)
        ).round(1)
        
        return landfill_df
    except Exception as e:
        st.error(f"Error assigning cities: {e}")
        return landfill_df

@st.cache_data(ttl=3600)
def load_landfills():
    """Load and prepare landfill data."""
    try:
        # Load only needed columns for better performance
        needed_columns = ["Landfill Name", "State", "City", "County", "Latitude", "Longitude", 
                         "Percent Methane", "LFG Collected (mmscfd)", "Ownership Type",
                         "Current Landfill Status", "LFG Collection System In Place?",
                         "Flares in Place?", "Passive Venting/Flaring?", "Landfill Closure Year",
                         "Waste in Place (tons)", "Waste in Place Year", "Annual Waste Acceptance Year",
                         "Current Landfill Depth (feet)"]
        
        df = pd.read_excel("landfilllmopdata.xlsx", sheet_name="LMOP Database")
        
        # Clean data
        df.columns = df.columns.astype(str).str.strip()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["Percent Methane"] = pd.to_numeric(df["Percent Methane"], errors="coerce")
        df["LFG Collected (mmscfd)"] = pd.to_numeric(df["LFG Collected (mmscfd)"], errors="coerce")
        df["Waste in Place (tons)"] = pd.to_numeric(df["Waste in Place (tons)"], errors="coerce")
        df["Annual Waste Acceptance Year"] = pd.to_numeric(df["Annual Waste Acceptance Year"], errors="coerce")
        df["Current Landfill Depth (feet)"] = pd.to_numeric(df["Current Landfill Depth (feet)"], errors="coerce")
        
        return df.dropna(subset=["Latitude", "Longitude"])
    except Exception as e:
        st.error(f"Error loading landfill data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_cities():
    """Load and prepare city data."""
    try:
        cities = pd.read_csv("us_cities_over_25k.csv", encoding="latin1")
        cities = cities[cities["Population"] > 20000]
        return cities[["City", "State", "Latitude", "Longitude", "Population"]]
    except Exception as e:
        st.error(f"Error loading city data: {e}")
        return pd.DataFrame()

# Create hash of filter settings to detect changes
def get_filter_hash(filters):
    return hashlib.md5(str(filters).encode()).hexdigest()

# MAIN APP FUNCTION
def main():
    st.title("Galactic Landfill Explorer")
    
    # Load data
    cities_df = load_cities()
    landfill_df = load_landfills()
    
    if landfill_df.empty or cities_df.empty:
        st.error("Could not load necessary data. Please check your data files.")
        return
    
    # SIDEBAR CONTROLS
    st.sidebar.title("Controls")
    
    assign_cities = st.sidebar.checkbox(
        "Calculate distance to nearest city", 
        value=False, 
        key="assign_cities",
        help="When checked, finds the nearest city to each landfill. This calculation may slow down performance, but enables distance-based analysis."
    )
    
    # Map display options
    st.sidebar.header("Map Display")
    exclude_missing = st.sidebar.checkbox(
        "Exclude landfills with missing key fields", 
        value=True, 
        key="exclude_missing",
        help="When checked, landfills missing data for key fields (like methane percentage or LFG collection) will be hidden from the map."
    )
    enable_clustering = st.sidebar.checkbox(
        "Enable Marker Clustering", 
        value=True, 
        key="enable_clustering",
        help="When checked, nearby landfills are grouped together as clusters when zoomed out. This improves map performance and reduces visual clutter."
    )
    
    # Add full data toggle to sidebar
    show_all_data = st.sidebar.checkbox(
        "Show all landfills (may be slower)", 
        value=st.session_state.get('show_all_landfills', False), 
        key="show_all_data",
        help="When checked, displays ALL landfills on the map. This can significantly impact performance on slower computers."
    )
    
    # Update session state based on checkbox
    st.session_state.show_all_landfills = show_all_data
    if 'prev_all_data_state' not in st.session_state or st.session_state.prev_all_data_state != show_all_data:
        st.session_state.prev_all_data_state = show_all_data
        st.session_state.need_map_update = True
    
    # Market settings
    st.sidebar.header("Market Settings")
    efficiency = st.sidebar.number_input(
        "Electrical Efficiency (%)", 
        min_value=1.0, 
        max_value=100.0, 
        value=40.0, 
        key="efficiency",
        help="The percentage of landfill gas energy that can be converted to electricity. Higher values increase projected revenue."
    ) / 100
    
    elec_price = st.sidebar.number_input(
        "Electricity Price ($/kWh)", 
        min_value=0.01, 
        max_value=2.0, 
        value=0.06, 
        key="elec_price",
        help="The selling price of electricity in dollars per kilowatt-hour. This directly impacts revenue calculations."
    )
    
    days_per_year = st.sidebar.number_input(
        "Operational Days/year", 
        min_value=1, 
        max_value=365, 
        value=365, 
        key="days_per_year",
        help="Number of days per year the facility operates. Affects annual revenue projections."
    )
    
    years = st.sidebar.number_input(
        "Project lifetime (years)", 
        min_value=1, 
        max_value=50, 
        value=20, 
        key="years",
        help="Expected operational lifetime of the project in years. Used to calculate lifetime revenue."
    )
    
    # Sorting options
    sort_by = st.sidebar.selectbox(
        "Sort by", 
        ["None", "Percent Methane", "LFG Collected (mmscfd)"], 
        key="sort_by",
        help="Sort the landfill data based on a specific criterion. Higher values will appear first in the data tables."
    )
    
    # Process data - only assign cities if checkbox is selected
    if assign_cities and "Nearest City" not in landfill_df.columns:
        with st.spinner("Assigning nearest cities to landfills..."):
            df_all = assign_nearest_city_fast(landfill_df, cities_df)
    else:
        df_all = landfill_df.copy()
    
    # Create deep copy to prevent modification warnings
    df = df_all.copy()
    
    # Apply sort if selected
    if sort_by != "None" and sort_by in df.columns:
        df[sort_by] = pd.to_numeric(df[sort_by], errors="coerce")
        df = df.sort_values(by=sort_by, ascending=False)
    
    # LOCATION FILTERS SECTION
    st.sidebar.header("Location Filters")
    
    # State filter
    all_states = sorted(df["State"].dropna().unique())
    selected_states = st.sidebar.multiselect(
        "Filter by State", 
        all_states,
        key="state_filter",
        help="Select one or more states to filter landfills. Leave empty to include all states."
    )
    
    # Apply state filter
    if selected_states:
        df = df[df["State"].isin(selected_states)]
        # Update available cities based on selected states
        available_cities = sorted(df["City"].dropna().unique())
    else:
        available_cities = sorted(df["City"].dropna().unique())
    
    # City filter - only show cities from selected states
    selected_cities = st.sidebar.multiselect(
        "Filter by City", 
        available_cities,
        key="city_filter",
        help="Select one or more cities to filter landfills. Leave empty to include all cities in the selected states."
    )
    
    # Apply city filter
    if selected_cities:
        df = df[df["City"].isin(selected_cities)]
    
    # Categorical filters
    st.sidebar.header("Categorical Filters")
    categorical_selections = {}
    
    for col in ["County", "Ownership Type", "Current Landfill Status",
                "LFG Collection System In Place?", "Flares in Place?", "Passive Venting/Flaring?"]:
        if col in df.columns:
            unique_values = sorted(df[col].dropna().unique())
            if len(unique_values) <= 30:  # Only show if not too many options
                opts = st.sidebar.multiselect(col, unique_values, key=f"cat_{col}")
                categorical_selections[col] = opts
                if opts:
                    df = df[df[col].isin(opts)]
    
    # Function for numeric sliders
    numeric_selections = {}
    
    def optional_slider(df, col, label, help_text):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid_data = df[col].dropna()
            if not valid_data.empty and st.sidebar.checkbox(
                f"Filter {label}", 
                key=f"check_{col}",
                help=help_text
            ):
                try:
                    lo, hi = float(valid_data.min()), float(valid_data.max())
                    if lo == hi:  # Handle edge case
                        st.sidebar.warning(f"All values for {label} are {lo}, can't filter.")
                        return df, None
                    minval, maxval = st.sidebar.slider(
                        label, lo, hi, (lo, hi), key=f"slider_{col}"
                    )
                    numeric_selections[col] = (minval, maxval)
                    return df[df[col].between(minval, maxval)], (minval, maxval)
                except Exception as e:
                    st.sidebar.error(f"Error creating slider for {label}: {e}")
            return df, None
        return df, None
    
    # Numeric filters
    st.sidebar.header("Numeric Filters")
    df, _ = optional_slider(df, "Percent Methane", "% Methane", 
                           "Filter landfills based on their methane percentage. Higher values typically indicate better energy potential.")
    df, _ = optional_slider(df, "LFG Collected (mmscfd)", "LFG Collected", 
                           "Filter by Landfill Gas collected in million standard cubic feet per day. Higher values indicate greater production capacity.")
    # Only include Distance to City slider if we're calculating it
    # Only include Distance to City slider if we're calculating it
    if assign_cities and "Distance to City (km)" in df.columns:
        if st.sidebar.checkbox(
            f"Filter Distance to City", 
            key=f"check_Distance to City (km)",
            help="Filter landfills based on their proximity to the nearest city in kilometers."
        ):
            # Create a custom range slider for distance (1-1000 km)
            minval, maxval = st.sidebar.slider(
                "Distance to City (km)",
                min_value=1.0,
                max_value=1000.0,
                value=(1.0, 1000.0),  # Default range
                step=1.0,
                key=f"slider_Distance to City (km)"
            )
            numeric_selections["Distance to City (km)"] = (minval, maxval)
            df = df[df["Distance to City (km)"].between(minval, maxval)]
    df, _ = optional_slider(df, "Landfill Closure Year", "Closure Year",
                           "Filter landfills based on when they closed or are projected to close. Impacts long-term gas production potential.")
    
    # New numeric filters for additional fields
    df, _ = optional_slider(df, "Waste in Place (tons)", "Waste in Place", 
                           "Filter landfills based on the total amount of waste in the landfill in tons.")
    df, _ = optional_slider(df, "Annual Waste Acceptance (tons per year)", "Annual Waste Acceptance", 
                           "Filter landfills based on how much waste they accept annually in tons per year.")
    df, _ = optional_slider(df, "Design Landfill Depth (feet)", "Landfill Depth", 
                           "Filter landfills based on their designed depth in feet.")
    
    # Create filter hash to detect changes
    current_filters = {
        'exclude_missing': exclude_missing,
        'enable_clustering': enable_clustering,
        'sort_by': sort_by,
        'assign_cities': assign_cities,
        'categorical': categorical_selections,
        'numeric': numeric_selections,
        'selected_states': selected_states,
        'selected_cities': selected_cities
    }
    
    new_filter_hash = get_filter_hash(current_filters)
    
    # Check if filters changed
    if new_filter_hash != st.session_state.filter_hash:
        st.session_state.filter_hash = new_filter_hash
        st.session_state.need_map_update = True
    
    # Reset button with help text
    if st.sidebar.button(
        "Reset All Filters", 
        help="Clear all filters and return to the default view. This will reset all your selections."
    ):
        st.session_state.clear()
        st.experimental_rerun()
    
    # SUMMARY STATS SECTION
    st.header("Summary Stats")

    # Create a calculation dataframe with needed columns
    calc_df = df.copy()
    calc_df["LFG Collected (mmscfd)"] = pd.to_numeric(calc_df["LFG Collected (mmscfd)"], errors="coerce")
    calc_df["Percent Methane"] = pd.to_numeric(calc_df["Percent Methane"], errors="coerce")

    # Get total LFG collected
    total_lfg = calc_df["LFG Collected (mmscfd)"].sum()

    # Fill missing methane percentages with a default value (50%)
    calc_df["Percent Methane"].fillna(50, inplace=True)

    # Calculate methane volume for each landfill
    calc_df["Methane Volume (scfd)"] = calc_df["LFG Collected (mmscfd)"] * (calc_df["Percent Methane"]/100) * 1e6

    # Calculate total methane collected (in mmscfd for consistency)
    total_methane = (calc_df["Methane Volume (scfd)"].sum() / 1e6)

    # Energy content of methane (BTU to kWh conversion)
    energy_content_kwh = 1011 * 0.000293071  # kWh per scf of methane (≈ 0.296 kWh/scf)

    # Calculate energy for each landfill
    calc_df["Energy Potential (kWh)"] = calc_df["Methane Volume (scfd)"] * energy_content_kwh
    calc_df["Energy Output (kWh)"] = calc_df["Energy Potential (kWh)"] * efficiency
    calc_df["Energy Output (MWh)"] = calc_df["Energy Output (kWh)"] / 1000

    # Sum the energy output across all landfills - used for revenue calculation but not displayed
    total_energy_mwh = calc_df["Energy Output (MWh)"].sum()

    # Calculate revenue based on total energy
    revenue_day = total_energy_mwh * 1000 * elec_price  # Convert MWh back to kWh for pricing
    annual_revenue = revenue_day * days_per_year
    lifetime_revenue = annual_revenue * years

    # Display metrics in columns for better UI
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Landfills", f"{len(df):,}")
        st.metric("Total LFG Collected", f"{total_lfg:.2f} mmscfd")
    with col2:
        st.metric("Total Methane Collected", f"{total_methane:.2f} mmscfd")
        st.metric("Daily Revenue", f"${revenue_day:,.0f}")
    with col3:
        st.metric("Annual Revenue", f"${annual_revenue:,.0f}")
        st.metric("Lifetime Revenue", f"${lifetime_revenue:,.0f}")

    # Download button - include the calculation columns in the download
    calc_df_download = calc_df.copy()
    download_columns = list(df.columns) + ["Methane Volume (scfd)", "Energy Output (MWh)"]
    st.download_button(
        "Download Filtered Data with Energy Calculations", 
        calc_df_download[download_columns].to_csv(index=False), 
        file_name="filtered_landfills_with_energy.csv"
    )
    
    # Download button
    st.download_button(
        "Download Filtered Data", 
        df.to_csv(index=False), 
        file_name="filtered_landfills.csv"
    )
    
    # MAP SECTION
    st.header("Filtered Landfill Map")
    
    # Map configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        color_options = ["Percent Methane", "LFG Collected (mmscfd)"]
        if assign_cities and "Distance to City (km)" in df.columns:
            color_options.append("Distance to City (km)")
        color_by = st.selectbox(
            "Color markers by:", 
            color_options, 
            key="color_by",
            help="Choose which data attribute determines the marker colors on the map."
        )
    
    with col2:
        map_height = st.slider(
            "Map Height", 
            min_value=400, 
            max_value=800, 
            value=600, 
            step=50, 
            key="map_height",
            help="Adjust the height of the map display in pixels."
        )
    
    with col3:
        if st.button("Refresh Map", key="refresh_map", help="Force the map to update with current settings."):
            st.session_state.need_map_update = True
                
    # Track zoom state with consistent key naming
    if 'current_zoom' not in st.session_state:
        st.session_state.current_zoom = 5
    if 'current_center' not in st.session_state:
        st.session_state.current_center = [0, 0]
    if 'show_all_landfills' not in st.session_state:
        st.session_state.show_all_landfills = False
        
    # Always enable zoom detail (not optional anymore)
    st.session_state.enable_zoom_detail = True
    
    # Apply filtering for map display
    df_map = df.copy()
    
    # Define columns to check for missing values based on what's available
    missing_check_cols = ["Percent Methane", "LFG Collected (mmscfd)"]
    
    # Only include Distance to City if we're calculating it
    if assign_cities and "Distance to City (km)" in df_map.columns:
        missing_check_cols.append("Distance to City (km)")
    
    if exclude_missing:
        df_map = df_map.dropna(subset=missing_check_cols)
    
    # Early exit if no data
    if df_map.empty:
        st.warning("No landfills match your current filters. Try adjusting your criteria.")
        return
    
    # MAP CREATION
    # Only create or update the map when needed
    if st.session_state.need_map_update:
        with st.spinner("Creating map... please wait"):
            # Create base map
            m = folium.Map(
                location=[df_map["Latitude"].mean(), df_map["Longitude"].mean()], 
                zoom_start=5,
                control_scale=True
            )
            
            # Check if the selected color column exists
            if color_by in df_map.columns:
                # Ensure all values are numeric
                df_map[color_by] = pd.to_numeric(df_map[color_by], errors="coerce")
                
                # Get min/max for color scaling
                valid_values = df_map[color_by].dropna()
                if not valid_values.empty:
                    vmin = valid_values.min()
                    vmax = valid_values.max()
                else:
                    vmin, vmax = 0, 1
            else:
                # Default if column doesn't exist
                st.warning(f"Selected column '{color_by}' not available. Using default coloring.")
                vmin, vmax = 0, 1
            
            # Create appropriate container
            marker_container = MarkerCluster() if enable_clustering else m
            
            # Define max markers value
            initial_max_markers = 2000
            
            # Check if we're showing all landfills based on checkbox
            if st.session_state.get('show_all_landfills', False):
                st.success(f"Showing all {len(df_map)} landfills. Uncheck 'Show all landfills' in the sidebar if performance slows down.")
                # Keep the full dataset - no sampling
            elif len(df_map) > initial_max_markers:
                # Clear message about data sampling - zoom detail always enabled
                st.info(f"Sampling {initial_max_markers} of {len(df_map)} landfills for better performance. Zoom in to see all landfills in specific areas.")
                
                # Save the full dataset for later use
                st.session_state.full_landfill_data = df_map.copy()
                
                # For initial view, sample strategically but preserve highest value points
                # Sort by LFG collected and take top 500
                if "LFG Collected (mmscfd)" in df_map.columns:
                    top_lfg = df_map.dropna(subset=["LFG Collected (mmscfd)"])
                    top_lfg = top_lfg.nlargest(min(500, len(top_lfg)), "LFG Collected (mmscfd)")
                    
                    # Random sample the rest to reach max_markers
                    remaining = df_map[~df_map.index.isin(top_lfg.index)]
                    remaining_sample = remaining.sample(min(initial_max_markers - len(top_lfg), len(remaining)))
                    
                    # Combine top values with random sample
                    df_map = pd.concat([top_lfg, remaining_sample])
                else:
                    # If no LFG data, just sample randomly
                    df_map = df_map.sample(initial_max_markers)
            
            # Show progress
            progress_bar = st.progress(0)
            
            # Pre-compute values for performance
            if color_by in df_map.columns:
                df_map['val'] = pd.to_numeric(df_map[color_by], errors="coerce")
                
                def get_color(val):
                    if pd.isna(val) or vmin == vmax:
                        return "gray"
                    scale = (val - vmin) / (vmax - vmin)
                    return "green" if scale < 0.33 else "orange" if scale < 0.66 else "red"
                
                df_map['color'] = df_map['val'].apply(get_color)
            else:
                df_map['color'] = "blue"
            
            # Add markers in batches
            total_rows = len(df_map)
            for i, (_, row) in enumerate(df_map.iterrows()):
                # Update progress bar
                if i % 10 == 0:
                    progress_bar.progress(min(i/total_rows, 1.0))
                
                try:
                    lat, lon = float(row["Latitude"]), float(row["Longitude"])
                    
                    # Skip invalid coordinates
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Create popup content
                    popup_parts = [
                        f"<b>{row.get('Landfill Name', 'Unnamed')}</b>"
                    ]

                    # Add location information
                    address_parts = []
                    if 'Address' in row and not pd.isna(row.get('Address')):
                        address_parts.append(f"{row.get('Address')}")
                    if 'City' in row and not pd.isna(row.get('City')):
                        address_parts.append(f"{row.get('City')}")
                    if 'State' in row and not pd.isna(row.get('State')):
                        address_parts.append(f"{row.get('State')}")

                    # Join address parts with commas
                    if address_parts:
                        popup_parts.append(", ".join(address_parts))

                    # Add value if available
                    if color_by in df_map.columns and 'val' in row:
                        val = row['val']
                        if not pd.isna(val):
                            popup_parts.append(f"{color_by}: {val:.2f}")

                    # Create popup
                    popup = "<br>".join(popup_parts)
                    
                    # Add marker
                    folium.Marker(
                        [lat, lon], 
                        popup=folium.Popup(popup, max_width=300),
                        icon=folium.Icon(color=row.get('color', 'blue'), icon="leaf")
                    ).add_to(marker_container if enable_clustering else m)
                    
                except Exception:
                    continue
            
            # Add cluster to map if enabled
            if enable_clustering:
                marker_container.add_to(m)
            
            # Complete progress
            progress_bar.progress(1.0)
            time.sleep(0.5)  # Short delay to show complete progress
            progress_bar.empty()
            
            # Save map to session state
            st.session_state.map = m
            st.session_state.need_map_update = False
            st.session_state.map_created = True
            
            # Success message
            st.success("Map created successfully!")
    
    # Display map from session state with zoom handling
    if st.session_state.map_created:
        try:
            # Create container for map
            map_container = st.container()
            
            # Add zoom functionality
            st.checkbox("Enable zoom-based detail", value=True, key="enable_zoom_detail", 
                       help="When checked, zooming in will show all landfills in the current view")
            
            # Use folium_static with zoom detection features
            with map_container:
                # Store the map HTML for later reference
                if 'map_html' not in st.session_state:
                    st.session_state.map_html = st.session_state.map._repr_html_()
                
                # Display the map
                folium_static(st.session_state.map, width=1000, height=map_height)
            
            # Add zoom-based detail explanation and controls
            if st.session_state.get("show_all_landfills", False):
                reset_col1, reset_col2 = st.columns([3, 1])
                with reset_col1:
                    st.warning("⚠️ Showing all landfills may affect performance on slower computers.")
                with reset_col2:
                    if st.button("Switch to Sampled Mode"):
                        st.session_state.show_all_landfills = False
                        st.session_state.need_map_update = True
                        st.experimental_rerun()
            elif st.session_state.get("enable_zoom_detail", True):
                st.info("💡 Tip: Use 'Show All Data' to see all landfills or zoom to areas of interest.")
        except Exception as e:
            st.error(f"Error displaying map: {e}")
            st.session_state.need_map_update = True
    
    # ANALYSIS DASHBOARD
    with st.expander("Analysis Dashboard", expanded=False):
        st.markdown("## 📊 **Analysis Dashboard**")
        df_dash = df.copy()
        
        # Create tabs with the added visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Missing Data", "Nearest Finder", "Core Visualizations", "Advanced Visualizations"])
        
        with tab1:
            st.subheader("Missing Data Overview")
            missing_stats = df_dash.isna().sum().sort_values(ascending=False)
            st.dataframe(missing_stats[missing_stats > 0].rename("Missing Count"))
        
        with tab2:
            st.subheader("Lat/Lon Nearest Landfill Finder")
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input(
                    "Latitude", 
                    min_value=-90.0, 
                    max_value=90.0, 
                    value=40.0, 
                    key="lat_input"
                )
            with col2:
                lon = st.number_input(
                    "Longitude", 
                    min_value=-180.0, 
                    max_value=180.0, 
                    value=-100.0, 
                    key="lon_input"
                )
            
            if st.button("Find Nearest", key="find_nearest"):
                try:
                    # Calculate distances
                    dists = haversine_np(
                        df_dash["Longitude"].astype(float), 
                        df_dash["Latitude"].astype(float), 
                        lon, lat
                    )
                    
                    if dists.empty:
                        st.warning("No data available for nearest calculation.")
                    else:
                        # Find nearest landfill
                        nearest_idx = dists.idxmin()
                        nearest = df_dash.loc[nearest_idx]
                        
                        # Display results
                        st.success(f"Nearest landfill: {nearest.get('Landfill Name', 'Unnamed')}")
                        
                        # Show details
                        result_cols = ["Landfill Name", "City", "State", "Percent Methane", 
                                     "LFG Collected (mmscfd)"]
                        if "Distance to City (km)" in nearest:
                            result_cols.append("Distance to City (km)")
                            
                        st.dataframe(nearest[result_cols])
                        
                        # Create mini map for nearest landfill
                        mini_map = folium.Map(
                            location=[float(nearest["Latitude"]), float(nearest["Longitude"])], 
                            zoom_start=10
                        )
                        
                        # Add landfill marker
                        folium.Marker(
                            [float(nearest["Latitude"]), float(nearest["Longitude"])],
                            popup=nearest.get("Landfill Name", "Unnamed"),
                            icon=folium.Icon(color="red")
                        ).add_to(mini_map)
                        
                        # Add user location marker
                        folium.Marker(
                            [lat, lon],
                            popup="Your Location",
                            icon=folium.Icon(color="blue", icon="info-sign")
                        ).add_to(mini_map)
                        
                        # Show mini map with unique key
                        folium_static(mini_map, width=700, height=400, key="nearest_landfill_map")
                        
                except Exception as e:
                    st.error(f"Error finding nearest landfill: {e}")
        
        with tab3:
            st.subheader("Core Visualizations")
            viz_type = st.selectbox(
                "Select Visualization", 
                ["Methane vs LFG", "Closure Year vs LFG", "Distance Histogram", "Top 10 Tables"],
                key="viz_type"
            )
            
            if viz_type == "Methane vs LFG":
                try:
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    scatter_data = df_dash.dropna(subset=["Percent Methane", "LFG Collected (mmscfd)"])
                    
                    if not scatter_data.empty:
                        sns.scatterplot(
                            data=scatter_data, 
                            x="Percent Methane", 
                            y="LFG Collected (mmscfd)",
                            alpha=0.7,
                            ax=ax1
                        )
                        ax1.set_title("Percent Methane vs LFG Collected")
                        ax1.grid(True, alpha=0.3)
                    else:
                        ax1.text(0.5, 0.5, "Insufficient data for plot", 
                                ha='center', va='center', fontsize=12)
                        
                    st.pyplot(fig1)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {e}")
                    
            elif viz_type == "Closure Year vs LFG":
                try:
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    scatter_data2 = df_dash.dropna(subset=["Landfill Closure Year", "LFG Collected (mmscfd)"])
                    
                    if not scatter_data2.empty:
                        sns.scatterplot(
                            data=scatter_data2, 
                            x="Landfill Closure Year", 
                            y="LFG Collected (mmscfd)",
                            alpha=0.7,
                            ax=ax2
                        )
                        # Set sensible x-limits to handle outliers
                        min_year = max(1950, scatter_data2["Landfill Closure Year"].min() - 5)
                        max_year = min(2050, scatter_data2["Landfill Closure Year"].max() + 5)
                        ax2.set_xlim(min_year, max_year)
                        ax2.set_title("Closure Year vs LFG Collected")
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, "Insufficient data for plot", 
                                ha='center', va='center', fontsize=12)
                        
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Error creating closure year plot: {e}")
                    
            elif viz_type == "Distance Histogram":
                if "Distance to City (km)" in df_dash.columns:
                    try:
                        # Safely convert to numeric with error handling
                        dist = pd.to_numeric(df_dash["Distance to City (km)"], errors="coerce")
                        
                        # Filter for reasonable distances (0-1000 km)
                        dist = dist[(dist >= 0) & (dist <= 1000)]
                        
                        if not dist.empty:
                            fig3, ax3 = plt.subplots(figsize=(8, 5))
                            sns.histplot(dist, bins=30, kde=True, ax=ax3)
                            ax3.set_title("Distance to Nearest City")
                            ax3.set_xlabel("Distance (km)")
                            ax3.grid(True, alpha=0.3)
                            st.pyplot(fig3)
                        else:
                            st.warning("No distance data available in the selected range")
                    except Exception as e:
                        st.error(f"Error creating histogram: {e}")
                else:
                    st.warning("Distance to City data not available. Enable city calculations to see this visualization.")
                    
            elif viz_type == "Top 10 Tables":
                try:
                    # Select relevant columns
                    cols = ["Landfill Name", "City", "State", "Percent Methane", "LFG Collected (mmscfd)"]
                    if "Distance to City (km)" in df_dash.columns:
                        cols.append("Distance to City (km)")
                    
                    # Create a copy to avoid SettingWithCopyWarning
                    df_top = df_dash[cols].copy()
                    
                    # Convert numeric columns
                    for col in ["Percent Methane", "LFG Collected (mmscfd)"]:
                        df_top[col] = pd.to_numeric(df_top[col], errors="coerce")
                    
                    if "Distance to City (km)" in df_top.columns:
                        df_top["Distance to City (km)"] = pd.to_numeric(df_top["Distance to City (km)"], errors="coerce")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top 10 by Methane %**")
                        top_methane = df_top.dropna(subset=["Percent Methane"]) \
                            .sort_values("Percent Methane", ascending=False) \
                            .head(10)[["Landfill Name", "State", "Percent Methane"]]
                        st.dataframe(top_methane)
                    
                    with col2:
                        st.markdown("**Top 10 by LFG Collected**")
                        top_lfg = df_top.dropna(subset=["LFG Collected (mmscfd)"]) \
                            .sort_values("LFG Collected (mmscfd)", ascending=False) \
                            .head(10)[["Landfill Name", "State", "LFG Collected (mmscfd)"]]
                        st.dataframe(top_lfg)
                    
                    if "Distance to City (km)" in df_top.columns:
                        st.markdown("**Top 10 Closest to Cities**")
                        top_close = df_top.dropna(subset=["Distance to City (km)"]) \
                            .sort_values("Distance to City (km)") \
                            .head(10)[["Landfill Name", "State", "Distance to City (km)"]]
                        st.dataframe(top_close)
                except Exception as e:
                    st.error(f"Error displaying top 10 tables: {e}")
        
        # NEW TAB FOR ADVANCED VISUALIZATIONS
        with tab4:
            st.subheader("Advanced Visualizations")
            advanced_viz_type = st.selectbox(
                "Select Advanced Visualization", 
                ["Waste in Place vs LFG Collected", "Landfill Depth vs LFG Collected", "Annual Waste Acceptance Year Histogram", 
                 "Multiple Factors Analysis"],
                key="advanced_viz_type"
            )
            
            if advanced_viz_type == "Waste in Place vs LFG Collected":
                try:
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    scatter_data4 = df_dash.dropna(subset=["Waste in Place (tons)", "LFG Collected (mmscfd)"])
                    
                    if not scatter_data4.empty:
                        # Convert to millions of tons for better readability
                        scatter_data4["Waste in Place (millions of tons)"] = scatter_data4["Waste in Place (tons)"] / 1e6
                        
                        # Create scatter plot with regression line
                        sns.regplot(
                            data=scatter_data4,
                            x="Waste in Place (millions of tons)",
                            y="LFG Collected (mmscfd)",
                            scatter_kws={"alpha": 0.6},
                            line_kws={"color": "red"},
                            ax=ax4
                        )
                        
                        ax4.set_title("Waste in Place vs LFG Collected")
                        ax4.set_xlabel("Waste in Place (Millions of Tons)")
                        ax4.set_ylabel("LFG Collected (mmscfd)")
                        ax4.grid(True, alpha=0.3)
                        
                        # Calculate correlation coefficient
                        corr = scatter_data4["Waste in Place (tons)"].corr(scatter_data4["LFG Collected (mmscfd)"])
                        ax4.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", 
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                        
                        st.pyplot(fig4)
                        
                        # Show additional statistics
                        st.write("### Statistical Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Waste in Place Statistics (millions of tons)**")
                            waste_stats = scatter_data4["Waste in Place (millions of tons)"].describe()
                            st.dataframe(waste_stats)
                            
                        with col2:
                            st.write("**Relationship Insights**")
                            st.write(f"- Correlation coefficient: {corr:.2f}")
                            st.write(f"- Sample size: {len(scatter_data4)} landfills")
                            st.write(f"- Average LFG per million tons: {scatter_data4['LFG Collected (mmscfd)'].sum() / scatter_data4['Waste in Place (millions of tons)'].sum():.3f} mmscfd")
                    else:
                        ax4.text(0.5, 0.5, "Insufficient data for plot", 
                                ha='center', va='center', fontsize=12)
                        st.pyplot(fig4)
                        st.warning("Not enough data with both Waste in Place and LFG Collected values to create visualization.")
                except Exception as e:
                    st.error(f"Error creating Waste vs LFG plot: {e}")
            
            elif advanced_viz_type == "Landfill Depth vs LFG Collected":
                try:
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    scatter_data5 = df_dash.dropna(subset=["Current Landfill Depth (feet)", "LFG Collected (mmscfd)"])
                    
                    if not scatter_data5.empty:
                        # Create scatter plot with hue by methane percent if available
                        if "Percent Methane" in scatter_data5.columns:
                            scatter_data5["Percent Methane"] = pd.to_numeric(scatter_data5["Percent Methane"], errors="coerce")
                            valid_data = scatter_data5.dropna(subset=["Percent Methane"])
                            
                            if not valid_data.empty:
                                scatter = sns.scatterplot(
                                    data=valid_data,
                                    x="Current Landfill Depth (feet)",
                                    y="LFG Collected (mmscfd)",
                                    hue="Percent Methane",
                                    palette="viridis",
                                    alpha=0.7,
                                    ax=ax5
                                )
                                # Add colorbar legend
                                norm = plt.Normalize(valid_data["Percent Methane"].min(), valid_data["Percent Methane"].max())
                                sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                                sm.set_array([])
                                cbar = fig5.colorbar(sm, ax=ax5)
                                cbar.set_label("Percent Methane")
                                
                                # Remove the default legend
                                if ax5.get_legend():
                                    ax5.get_legend().remove()
                            else:
                                # Fallback to simple scatter plot if no methane data
                                sns.scatterplot(
                                    data=scatter_data5,
                                    x="Current Landfill Depth (feet)",
                                    y="LFG Collected (mmscfd)",
                                    ax=ax5
                                )
                        else:
                            # Default scatter plot
                            sns.scatterplot(
                                data=scatter_data5,
                                x="Current Landfill Depth (feet)",
                                y="LFG Collected (mmscfd)",
                                ax=ax5
                            )
                        
                        # Add regression line
                        sns.regplot(
                            data=scatter_data5,
                            x="Current Landfill Depth (feet)",
                            y="LFG Collected (mmscfd)",
                            scatter=False,
                            line_kws={"color": "red"},
                            ax=ax5
                        )
                        
                        ax5.set_title("Landfill Depth vs LFG Collected")
                        ax5.set_xlabel("Current Landfill Depth (feet)")
                        ax5.set_ylabel("LFG Collected (mmscfd)")
                        ax5.grid(True, alpha=0.3)
                        
                        # Calculate correlation coefficient
                        corr = scatter_data5["Current Landfill Depth (feet)"].corr(scatter_data5["LFG Collected (mmscfd)"])
                        ax5.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", 
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                        
                        st.pyplot(fig5)
                        
                        # Additional depth analysis
                        st.write("### Depth Analysis")
                        
                        # Create depth bins
                        bins = [0, 50, 100, 150, 200, 250, 300, float('inf')]
                        labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '300+']
                        scatter_data5['Depth Range (feet)'] = pd.cut(scatter_data5['Current Landfill Depth (feet)'], bins=bins, labels=labels)
                        
                        # Group by depth range
                        depth_analysis = scatter_data5.groupby('Depth Range (feet)').agg({
                            'LFG Collected (mmscfd)': ['count', 'mean', 'std', 'min', 'max'],
                            'Current Landfill Depth (feet)': ['mean']
                        }).reset_index()
                        
                        # Format for display
                        depth_analysis.columns = ['Depth Range (feet)', 'Count', 'Mean LFG', 'Std Dev LFG', 'Min LFG', 'Max LFG', 'Mean Depth']
                        depth_analysis = depth_analysis.sort_values('Mean Depth')
                        
                        # Show the table
                        st.dataframe(depth_analysis)
                        
                        # Create a bar chart of LFG by depth range
                        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                        bar_data = depth_analysis[depth_analysis['Count'] > 0]  # Only plot where we have data
                        bar_data.plot(kind='bar', x='Depth Range (feet)', y='Mean LFG', yerr='Std Dev LFG', ax=ax_bar, color='skyblue')
                        ax_bar.set_title('Average LFG Collected by Landfill Depth Range')
                        ax_bar.set_ylabel('LFG Collected (mmscfd)')
                        ax_bar.set_xlabel('Depth Range (feet)')
                        plt.xticks(rotation=45)
                        st.pyplot(fig_bar)
                        
                    else:
                        ax5.text(0.5, 0.5, "Insufficient data for plot", 
                                ha='center', va='center', fontsize=12)
                        st.pyplot(fig5)
                        st.warning("Not enough data with both Landfill Depth and LFG Collected values to create visualization.")
                except Exception as e:
                    st.error(f"Error creating Depth vs LFG plot: {e}")
            
            elif advanced_viz_type == "Annual Waste Acceptance Year Histogram":
                try:
                    # Convert to numeric and handle missing values
                    df_dash["Annual Waste Acceptance Year"] = pd.to_numeric(
                        df_dash["Annual Waste Acceptance Year"], errors="coerce"
                    )
                    
                    waste_data = df_dash.dropna(subset=["Annual Waste Acceptance Year"])
                    
                    if not waste_data.empty:
                        # Create figure with histogram
                        fig6, ax6 = plt.subplots(figsize=(12, 6))
                        
                        # Histogram of Annual Waste Acceptance Year
                        sns.histplot(
                            data=waste_data,
                            x="Annual Waste Acceptance Year",
                            kde=True,
                            bins=20,
                            ax=ax6
                        )
                        ax6.set_title("Annual Waste Acceptance Year Distribution")
                        ax6.set_xlabel("Year")
                        ax6.set_ylabel("Count")
                        ax6.grid(True, alpha=0.3)
                        
                        # Set x-axis to show years clearly
                        min_year = int(waste_data["Annual Waste Acceptance Year"].min())
                        max_year = int(waste_data["Annual Waste Acceptance Year"].max())
                        ax6.set_xticks(range(min_year, max_year + 1, 2))  # Show every other year
                        plt.xticks(rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig6)
                        
                        # Display summary statistics
                        st.write("### Annual Waste Acceptance Year Statistics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Basic Statistics**")
                            waste_stats = waste_data["Annual Waste Acceptance Year"].describe().astype(int)
                            st.dataframe(waste_stats)
                        
                        with col2:
                            # Calculate percentiles
                            percentiles = [10, 25, 50, 75, 90, 95, 99]
                            percentile_values = [int(waste_data["Annual Waste Acceptance Year"].quantile(p/100)) for p in percentiles]
                            
                            st.write("**Percentiles**")
                            percentile_df = pd.DataFrame({
                                'Percentile': percentiles,
                                'Year': percentile_values
                            })
                            st.dataframe(percentile_df)
                            
                        # Additional analysis by state
                        if "State" in waste_data.columns:
                            st.write("### Annual Waste Acceptance Year by State")
                            
                            # Calculate average year by state
                            state_data = waste_data.groupby("State")["Annual Waste Acceptance Year"].agg(['mean', 'count', 'min', 'max']).reset_index()
                            state_data.columns = ['State', 'Average Year', 'Number of Landfills', 'Earliest Year', 'Latest Year']
                            
                            # Round years to integers
                            state_data['Average Year'] = state_data['Average Year'].round().astype(int)
                            state_data['Earliest Year'] = state_data['Earliest Year'].astype(int)
                            state_data['Latest Year'] = state_data['Latest Year'].astype(int)
                            
                            # Sort by average year
                            state_data = state_data.sort_values('Average Year', ascending=False)
                            
                            # Show top states with most recent data
                            top_states = state_data.head(10)
                            
                            # Create bar chart
                            fig_state, ax_state = plt.subplots(figsize=(10, 6))
                            sns.barplot(
                                data=top_states,
                                x='State',
                                y='Average Year',
                                ax=ax_state
                            )
                            ax_state.set_title('Top 10 States by Most Recent Waste Acceptance Data')
                            ax_state.set_ylabel('Average Year of Data')
                            ax_state.set_xlabel('State')
                            plt.xticks(rotation=45)
                            
                            for i, v in enumerate(top_states['Average Year']):
                                ax_state.text(i, v - 3, f"{int(v)}", ha='center')
                                
                            plt.tight_layout()
                            st.pyplot(fig_state)
                            
                            # Show full state data in an expandable section
                            with st.expander("View data for all states"):
                                st.dataframe(state_data)
                    else:
                        st.warning("No Annual Waste Acceptance Year data available for visualization.")
                except Exception as e:
                    st.error(f"Error creating Annual Waste Acceptance Year histogram: {e}")
            
            elif advanced_viz_type == "Multiple Factors Analysis":
                try:
                    # Select relevant columns for analysis
                    analysis_cols = [
                        "LFG Collected (mmscfd)", 
                        "Percent Methane", 
                        "Waste in Place (tons)", 
                        "Annual Waste Acceptance Year",
                        "Current Landfill Depth (feet)"
                    ]
                    
                    # Create a copy and convert to numeric
                    multi_df = df_dash[analysis_cols].copy()
                    for col in multi_df.columns:
                        multi_df[col] = pd.to_numeric(multi_df[col], errors="coerce")
                    
                    # Drop rows with all NaN values
                    multi_df = multi_df.dropna(how="all")
                    
                    if not multi_df.empty:
                        st.write("### Correlation Analysis")
                        
                        # Calculate and display correlation matrix
                        corr_matrix = multi_df.corr()
                        
                        # Create heatmap
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        cmap = sns.diverging_palette(220, 10, as_cmap=True)
                        
                        sns.heatmap(
                            corr_matrix, 
                            mask=mask,
                            cmap=cmap,
                            annot=True,
                            fmt=".2f",
                            center=0,
                            square=True,
                            linewidths=.5,
                            ax=ax_corr
                        )
                        
                        ax_corr.set_title("Correlation Matrix of Key Metrics")
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        
                        # Create pairplot for deeper analysis
                        st.write("### Pairwise Relationships")
                        
                        # Sample data if more than 500 points to keep plot responsive
                        plot_df = multi_df
                        if len(multi_df) > 500:
                            plot_df = multi_df.sample(500)
                            st.info(f"Showing a random sample of 500 landfills (out of {len(multi_df)}) for faster rendering.")
                        
                        # Scale waste values for better visualization
                        if "Waste in Place (tons)" in plot_df.columns:
                            plot_df["Waste in Place (M tons)"] = plot_df["Waste in Place (tons)"] / 1e6
                            plot_df = plot_df.drop("Waste in Place (tons)", axis=1)
                            
                        if "Annual Waste Acceptance Year" in plot_df.columns:
                            # Keep the year as is, no need to scale
                            plot_df = plot_df
                        
                        # Create pairplot
                        with st.spinner("Generating pairplot... this may take a moment"):
                            pair_fig = sns.pairplot(
                                plot_df.dropna(), 
                                diag_kind="kde",
                                plot_kws={"alpha": 0.6}
                            )
                            pair_fig.fig.suptitle("Pairwise Relationships Between Key Metrics", y=1.02)
                            st.pyplot(pair_fig.fig)
                        
                        # Key insights section
                        st.write("### Key Insights")
                        
                        # Find strongest correlations
                        strong_correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                col1 = corr_matrix.columns[i]
                                col2 = corr_matrix.columns[j]
                                corr_value = corr_matrix.iloc[i, j]
                                if abs(corr_value) > 0.3:  # Arbitrary threshold for moderate correlation
                                    strong_correlations.append({
                                        'Variable 1': col1,
                                        'Variable 2': col2,
                                        'Correlation': corr_value
                                    })
                        
                        strong_corr_df = pd.DataFrame(strong_correlations).sort_values('Correlation', key=abs, ascending=False)
                        if not strong_corr_df.empty:
                            st.write("**Strongest Variable Relationships:**")
                            st.dataframe(strong_corr_df)
                            
                            # Write some interpretations
                            st.write("**Interpretation:**")
                            top_corr = strong_corr_df.iloc[0]
                            
                            if abs(top_corr['Correlation']) > 0.7:
                                strength = "very strong"
                            elif abs(top_corr['Correlation']) > 0.5:
                                strength = "strong"
                            else:
                                strength = "moderate"
                                
                            direction = "positive" if top_corr['Correlation'] > 0 else "negative"
                            
                            st.write(f"- There is a {strength} {direction} correlation ({top_corr['Correlation']:.2f}) between {top_corr['Variable 1']} and {top_corr['Variable 2']}.")
                            
                            if len(strong_corr_df) > 1:
                                second_corr = strong_corr_df.iloc[1]
                                if abs(second_corr['Correlation']) > 0.7:
                                    strength = "very strong"
                                elif abs(second_corr['Correlation']) > 0.5:
                                    strength = "strong"
                                else:
                                    strength = "moderate"
                                    
                                direction = "positive" if second_corr['Correlation'] > 0 else "negative"
                                
                                st.write(f"- There is a {strength} {direction} correlation ({second_corr['Correlation']:.2f}) between {second_corr['Variable 1']} and {second_corr['Variable 2']}.")
                        else:
                            st.write("No strong correlations found between the variables.")
                            
                    else:
                        st.warning("Insufficient data for multi-factor analysis. More data points with complete information are needed.")
                except Exception as e:
                    st.error(f"Error creating multiple factors analysis: {e}")

# Run the app
if __name__ == "__main__":
    main()