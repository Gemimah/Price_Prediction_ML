import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

def create_rwanda_district_map(df):
    """
    Creates a Rwanda map showing district boundaries and number of vehicle clients per district
    """
    
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    
    district_coords = {
        'Kigali City': {'lat': -1.9536, 'lon': 30.0606},
        'Nyarugenge': {'lat': -1.9833, 'lon': 30.0611},
        'Kicukiro': {'lat': -1.9333, 'lon': 30.0833},
        'Gasabo': {'lat': -1.9167, 'lon': 30.1333},
        'Northern Province': {
            'Burera': {'lat': -1.4667, 'lon': 29.8833},
            'Gicumbi': {'lat': -1.5500, 'lon': 30.0167},
            'Musanze': {'lat': -1.5000, 'lon': 29.6500},
            'Rulindo': {'lat': -1.6833, 'lon': 30.0333}
        },
        'Southern Province': {
            'Gisagara': {'lat': -2.6167, 'lon': 29.7500},
            'Huye': {'lat': -2.6000, 'lon': 29.7500},
            'Kamonyi': {'lat': -2.0500, 'lon': 30.0167},
            'Karongi': {'lat': -2.1500, 'lon': 29.3833},
            'Muhanga': {'lat': -2.2000, 'lon': 29.9500},
            'Nyamagabe': {'lat': -2.8500, 'lon': 29.5500},
            'Nyanza': {'lat': -2.3500, 'lon': 29.7000},
            'Nyaruhande': {'lat': -2.4500, 'lon': 29.6500},
            'Ruhango': {'lat': -2.1500, 'lon': 29.8500}
        },
        'Eastern Province': {
            'Bugesera': {'lat': -2.2000, 'lon': 30.2167},
            'Gatsibo': {'lat': -1.6167, 'lon': 30.4333},
            'Kayonza': {'lat': -1.9500, 'lon': 30.3500},
            'Kirehe': {'lat': -2.2833, 'lon': 30.7000},
            'Ngoma': {'lat': -2.1333, 'lon': 30.2167},
            'Nyagatare': {'lat': -1.3000, 'lon': 30.2500},
            'Rwamagana': {'lat': -1.9500, 'lon': 30.3833}
        },
        'Western Province': {
            'Karongi': {'lat': -2.1500, 'lon': 29.3833},
            'Ngororero': {'lat': -1.8500, 'lon': 29.6500},
        'Nyabihu': {'lat': -1.7500, 'lon': 29.8000},
            'Nyamasheke': {'lat': -2.2833, 'lon': 29.1500},
            'Rubavu': {'lat': -1.6833, 'lon': 29.2500},
            'Rusizi': {'lat': -2.4667, 'lon': 29.1500},
            'Rutsiro': {'lat': -1.9500, 'lon': 29.4500}
        }
    }
    
    # Flatten the coordinates and prepare data
    map_data = []
    for district, count in district_counts.values:
        # Find coordinates for this district
        coords = None
        for province, districts in district_coords.items():
            if isinstance(districts, dict) and district in districts:
                coords = districts[district]
                break
            elif province == district:  # Handle province-level entries
                coords = district_coords[province]
                break
        
        if coords:
            map_data.append({
                'district': district,
                'client_count': count,
                'lat': coords['lat'],
                'lon': coords['lon']
            })
    
    map_df = pd.DataFrame(map_data)
    
    # Create the map
    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="client_count",
        hover_name="district",
        hover_data=["client_count"],
        color="client_count",
        color_continuous_scale="Viridis",
        size_max=50,
        zoom=8,
        center={"lat": -1.9403, "lon": 30.0444},  # Center of Rwanda
        mapbox_style="open-street-map",
        title="Vehicle Clients by District in Rwanda"
    )
    
    fig.update_layout(
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def get_district_statistics(df):
    """Get statistics about vehicle clients by district"""
    district_stats = df.groupby('district').agg({
        'client_name': 'count',
        'selling_price': ['mean', 'sum'],
        'estimated_income': 'mean'
    }).round(2)
    
    district_stats.columns = ['Client Count', 'Avg Price', 'Total Sales', 'Avg Income']
    district_stats = district_stats.sort_values('Client Count', ascending=False)
    
    return district_stats.head(10).to_html(classes="table table-bordered table-striped table-sm")
