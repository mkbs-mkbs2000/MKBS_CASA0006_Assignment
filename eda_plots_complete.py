"""
This script generates plots on overall KSI:Slight split in the gpd_full_flagged
and the  for the EDA section.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.stats import pointbiserialr

# Set consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme for severity classes
COLORS = {
    'KSI': '#E74C3C',      # Red-orange for KSI
    'Slight': '#3498DB',   # Blue for Slight
    'Injuries': '#2ECC71', # Green for total injuries
}

# Font sizes for consistency
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# ============================================================================
# SECTION 1: DATASET OVERVIEW & CLASS DISTRIBUTION
# ============================================================================

def plot_class_distribution(df, target_col='casualty_severity'):
    """
    Bar chart showing KSI vs Slight distribution with percentages
    
    Parameters:
    -----------
    df : DataFrame with casualty_severity column (0=Slight, 1=KSI)
    target_col : name of target column
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Count classes
    class_counts = df[target_col].value_counts().sort_index()
    labels = ['Slight', 'KSI']
    counts = class_counts.values
    percentages = (counts / counts.sum() * 100)
    
    # Create bars
    bars = ax.bar(labels, counts, color=[COLORS['Slight'], COLORS['KSI']], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add count and percentage labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=LABEL_SIZE, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Number of Casualties', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel('Casualty Severity', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Class Distribution (N={len(df):,} casualties)', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_ylim(0, max(counts) * 1.15)  # Add headroom for labels
    
    # Add imbalance ratio annotation
    ratio = counts[0] / counts[1]
    ax.text(0.98, 0.95, f'Imbalance Ratio\n{ratio:.2f}:1', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 2: TEMPORAL PATTERNS
# ============================================================================

def plot_hourly_ksi_rate(df, hour_col='hour', target_col='casualty_severity'):
    """
    Line graph: KSI rate by hour with shaded lighting periods
    
    Parameters:
    -----------
    df : DataFrame with hour (0-23) and casualty_severity columns
    hour_col : name of hour column
    target_col : name of target column
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate KSI rate by hour
    hourly_stats = df.groupby(hour_col)[target_col].agg(['sum', 'count', 'mean'])
    hourly_stats['ksi_rate'] = hourly_stats['mean'] * 100
    
    # Plot line
    ax.plot(hourly_stats.index, hourly_stats['ksi_rate'], 
            marker='o', linewidth=2.5, markersize=6, color=COLORS['KSI'],
            label='KSI Rate (%)')
    
    # Add shaded regions for lighting conditions
    # Night: 0-6, Dusk/Dawn: 6-8 and 18-20, Day: 8-18, Night: 20-24
    ax.axvspan(0, 6, alpha=0.15, color='navy', label='Night')
    ax.axvspan(6, 12, alpha=0.1, color='orange', label='Morning')
    ax.axvspan(12, 18, alpha=0.05, color='yellow', label='Afternoon')
    ax.axvspan(18, 24, alpha=0.15, color='navy', label='Evening')
    
    # Formatting
    ax.set_xlabel('Hour of Day (0-23)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('KSI Rate (%)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Cyclist KSI Risk by Hour of Day (N={len(df):,})', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    # Add sample size annotation
    ax.text(0.98, 0.02, f'Mean KSI Rate: {hourly_stats["ksi_rate"].mean():.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_weekly_temporal_heatmap(df, hour_col='hour', day_col='day_of_week', 
                                   target_col='casualty_severity'):
    """
    Heatmap: KSI rate by day of week × time period
    
    Parameters:
    -----------
    df : DataFrame with hour, day_of_week, and casualty_severity columns
    hour_col : name of hour column (0-23)
    day_col : name of day column (1=Sunday, 7=Saturday per STATS19)
    target_col : name of target column
    """
    # Create time period bins
    time_periods = pd.cut(df[hour_col], 
                          bins=[0, 6, 12, 18, 24],
                          labels=['Night\n(0-6)', 'Morning\n(6-12)', 
                                  'Afternoon\n(12-18)', 'Evening\n(18-24)'],
                          include_lowest=True)
    
    df_temp = df.copy()
    df_temp['time_period'] = time_periods
    
    # Map day numbers to names (STATS19: 1=Sunday)
    day_names = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
                 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    df_temp['day_name'] = df_temp[day_col].map(day_names)
    
    # Calculate KSI rate for each day × time period
    pivot_data = df_temp.groupby(['day_name', 'time_period'])[target_col].agg(['mean', 'count'])
    pivot_data['ksi_rate'] = pivot_data['mean'] * 100
    
    # Reshape for heatmap
    heatmap_data = pivot_data['ksi_rate'].unstack()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'KSI Rate (%)'}, linewidths=0.5,
                vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max(),
                ax=ax)
    
    ax.set_xlabel('Time Period', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Day of Week', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('KSI Risk Patterns: Day of Week x Time Period', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.show()


def plot_monthly_trends(df, date_col='date', target_col='casualty_severity'):
    """
    Line graph: Monthly KSI rate trends (2020-2024)
    
    Parameters:
    -----------
    df : DataFrame with date column and casualty_severity
    date_col : name of date column (should be datetime)
    target_col : name of target column
    """
    df_temp = df.copy()
    
    # Extract year-month if date_col is datetime
    if pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
        df_temp['year_month'] = df_temp[date_col].dt.to_period('M')
    else:
        df_temp['year_month'] = pd.to_datetime(df_temp[date_col]).dt.to_period('M')
    
    # Calculate monthly statistics
    monthly_stats = df_temp.groupby('year_month')[target_col].agg(['sum', 'count', 'mean'])
    monthly_stats['ksi_rate'] = monthly_stats['mean'] * 100
    monthly_stats.index = monthly_stats.index.to_timestamp()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(monthly_stats.index, monthly_stats['ksi_rate'], 
            marker='o', linewidth=2, markersize=4, color=COLORS['KSI'])
    
    # Add COVID lockdown shading (March-June 2020)
    if monthly_stats.index.min().year == 2020:
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2022-01-27'), 
                   alpha=0.2, color='gray', label='COVID-19 Measures')
    
    # Formatting
    ax.set_xlabel('Month', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('KSI Rate (%)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Monthly KSI Rate Trends (2020-2024)', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')

    # Add sample size annotation
    ax.text(0.98, 0.02, f'Mean KSI Rate: {monthly_stats["ksi_rate"].mean():.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()



# ============================================================================
# SECTION 3: ENVIRONMENTAL/COLLISION CONTEXT
# ============================================================================

def plot_ksi_by_categorical_feature(df, feature_col, feature_labels, 
                                     target_col='casualty_severity',
                                     title='', xlabel=''):
    """
    Bar chart: Injury counts by categorical feature, putting both KSI and Slight injuries together
    
    Parameters:
    -----------
    df : DataFrame
    feature_col : name of categorical column
    feature_labels : dict mapping codes to readable labels
    target_col : name of target column
    title : plot title
    xlabel : x-axis label
    """
    # Calculate injuries by category
    counts = df[feature_col].value_counts().sort_index()
    
    # Map labels
    counts.index = counts.index.map(feature_labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    x_pos = np.arange(len(counts))
    bars = ax.bar(x_pos, counts.values, color=COLORS['Injuries'], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + counts.max() * 0.01,
            f'{int(height):,}',
            ha='center',
        va='bottom',
        fontsize=TICK_SIZE
    )

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel('Total Injuries', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig, ax


def plot_all_environmental_features(df, target_col='casualty_severity'):
    """
    Generate all environmental feature plots
    """
    # Light conditions
    light_labels = {
        1: 'Daylight',
        4: 'Darkness:\nLights lit',
        5: 'Darkness:\nLights unlit',
        6: 'Darkness:\nNo lighting',
        7: 'Darkness:\nLighting unknown',
        -1: 'Data missing'
    }
    
    fig, ax = plot_ksi_by_categorical_feature(
        df, 'light_conditions', light_labels,
        target_col=target_col,
        title='Injuries by Lighting Conditions',
    )
    plt.show()
    
    # Weather conditions
    weather_labels = {
        1: 'Fine\nno wind',
        2: 'Raining\nno wind',
        3: 'Snowing\nno wind',
        4: 'Fine +\nhigh winds',
        5: 'Raining +\nhigh winds',
        6: 'Snowing +\nhigh winds',
        7: 'Fog/mist',
        8: 'Other',
        9: 'Unknown',
        -1: 'Data missing'
    }
    
    fig, ax = plot_ksi_by_categorical_feature(
        df, 'weather_conditions', weather_labels,
        target_col=target_col,
        title='Injuries by Weather Conditions',
    )
    plt.show()
    
    # Road surface conditions
    surface_labels = {
        1: 'Dry',
        2: 'Wet/Damp',
        3: 'Snow',
        4: 'Frost/Ice',
        5: 'Flood',
        6: 'Oil or diesel',
        7: 'Mud',
        9: 'Unknown',
        -1: 'Data missing'
    }
    
    fig, ax = plot_ksi_by_categorical_feature(
        df, 'road_surface_conditions', surface_labels,
        target_col=target_col,
        title='Injuries by Road Surface Conditions',
    )
    plt.show()
    
    # Road type
    road_labels = {
        1: 'Roundabout',
        2: 'One way\nstreet',
        3: 'Dual\ncarriageway',
        6: 'Single\ncarriageway',
        7: 'Slip road',
        9: 'Unknown',
        12: 'One way street',
        -1: 'Data missing'
    }
    
    fig, ax = plot_ksi_by_categorical_feature(
        df, 'road_type', road_labels,
        target_col=target_col,
        title='Injuries by Road Type',
    )
    plt.show()

    # First road class
    road_class_labels = {
        1: 'Motorway',
        2: 'A(M)',
        3: 'A',
        4: 'B',
        5: 'C',
        6: 'Unclassified'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'first_road_class', road_class_labels,
        target_col=target_col,
        title='Injuries by First Road Class',
    )
    plt.show()

    # Junction detail
    junction_det_labels = {
        0: 'Not at/near junction',
        13: 'T/Staggered \njunction',
        16: 'Standard \njunction',
        17: 'Non-standard \njunction',
        18: 'Private driveway',
        19: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'junction_detail', junction_det_labels,
        target_col=target_col,
        title='Injuries by Junction Detail',
    )
    plt.show()
    
    # Carriageway Hazards
    carriageway_hzds_labels = {
        0: 'None',
        11: 'Defective \ntraffic signals',
        12: 'Road signing/marking \nobscure/defective/inadequate',
        13: 'Roadworks',
        14: 'Oil or diesel',
        15: 'Mud',
        16: 'Dislodged vehicle load',
        17: 'Other objects on road',
        18: 'Involvement with \nprevious collision',
        19: 'Pedestrian on road',
        20: 'Animal on road',
        21: 'Poor/defective \nroad surface',
        99: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'carriageway_hazards', carriageway_hzds_labels,
        target_col=target_col,
        title='Injuries by Carriageway Hazards',
    )
    plt.show()
    
    # Special Conditions
    special_con_labels = {
        0: 'None',
        1: 'Auto traffic \nsignal out',
        2: 'Auto signal \npart defective',
        3: 'Road sign/marking \ndefective/obscured',
        4: 'Roadworks',
        5: 'Road surface defective',
        6: 'Oil or diesel',
        7: 'Mud',
        9: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'special_conditions_at_site', special_con_labels,
        target_col=target_col,
        title='Injuries by Special Conditions',
    )
    plt.show()

    # skidding
    skidding_labels = {
        0: 'None',
        1: 'Skidded',
        2: 'Skidded \nand overturned',
        3: 'Jackknifed',
        4: 'Jackknifed \nand overturned',
        5: 'Overturned',
        9: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'skidding_and_overturning', skidding_labels,
        target_col=target_col,
        title='Injuries by Skidding',
    )
    plt.show()

    # hit object on
    hit_onroad_labels = {
        0: 'None',
        1: 'Previous accident',
        2: 'Road works',
        4: 'Parked vehicle',
        5: 'Bridge (roof)',
        6: 'Bridge (side)',
        7: 'Bollard/refuge',
        8: 'Open door of vehicle',
        9: 'Central island/\nroundabout',
        10: 'Kerb',
        11: 'Other objects',
        12: 'Animal',
        99: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'hit_object_in_carriageway', hit_onroad_labels,
        target_col=target_col,
        title='Injuries by Hitting Obj On Road',
    )
    plt.show()

    # hit object off
    hit_offroad_labels = {
        0: 'None',
        1: 'Road sign/\ntraffic signal',
        2: 'Lamp post',
        3: 'Telegraph/\nelectricity pole',
        4: 'Tree',
        5: 'Bus stop/\nshelter',
        6: 'Central crash barrier',
        7: 'Offside crash barrier',
        8: 'Submerged',
        9: 'Entered ditch',
        10: 'Other objects',
        11: 'Wall/fence',
        99: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'hit_object_off_carriageway', hit_offroad_labels,
        target_col=target_col,
        title='Injuries by Hitting Obj Off Road',
    )
    plt.show()

    # first pt of impact
    first_impact_labels = {
        0: 'No impact',
        1: 'Front',
        2: 'Back',
        3: 'Offside',
        4: 'Nearside',
        9: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'first_point_of_impact', first_impact_labels,
        target_col=target_col,
        title='Injuries by First Point of Impact',
    )
    plt.show()

    # IN RESTRICTED LANE?
    restricted_labels = {
        0: 'On normal road',
        1: 'On tram track',
        2: 'In bus lane',
        4: 'In on-road cycle lane',
        5: 'In off-road cycle lane',
        6: 'In emergency lane',
        9: 'On footpath',
        99: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'vehicle_location_restricted_lane', restricted_labels,
        target_col=target_col,
        title='Injuries by Presence in Restricted Lane',
    )
    plt.show()

    # WHERE IT IS ON THE JUNCTION?
    junc_loc_labels = {
        0: 'Not at/near junction',
        1: 'At junction approach',
        2: 'At junction exit',
        3: 'Leaving roundabout',
        4: 'Entering roundabout',
        5: 'Leaving main road',
        6: 'Entering main road',
        7: 'Entering from \nslip road',
        8: 'Mid junction', 
        9: 'Unknown',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'junction_location', junc_loc_labels,
        target_col=target_col,
        title='Injuries by Junction Location',
    )
    plt.show()

    # WHY WAS IT TRAVELLING?
    purpose_labels = {
        1: 'Part of work',
        2: 'To/from work',
        6: 'Unknown',
        7: 'School/\nschool run',
        8: 'Emergency',
        9: 'Leisure',
        -1: 'Data missing'
    }

    fig, ax = plot_ksi_by_categorical_feature(
        df, 'journey_purpose_of_driver', purpose_labels,
        target_col=target_col,
        title='Injuries by Journey Purpose of Cyclist',
    )
    plt.show()


# ============================================================================
# MASTER EXECUTION FUNCTION
# ============================================================================

def run_complete_eda(df_filtered):
    """
    Execute complete EDA pipeline
    
    Parameters:
    -----------
    df_filtered : DataFrame after filtering (for all other analyses)

    """    
    # Section 1: Dataset Overview
    print(">>> SECTION 1: Dataset Overview & Class Distribution")
    plot_class_distribution(df_filtered)

    # Section 2: Temporal Patterns
    print("\n>>> SECTION 2: Temporal Patterns")
    plot_hourly_ksi_rate(df_filtered)
    plot_weekly_temporal_heatmap(df_filtered)
    plot_monthly_trends(df_filtered)
    
    # Section 3: Environmental Features
    print("\n>>> SECTION 3: Data from STATS19")
    plot_all_environmental_features(df_filtered)