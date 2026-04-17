import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px

from shapely.geometry import Polygon
from scipy.spatial import Voronoi

SUB_FILE = "sub.csv"
BUS2SUB_FILE = "bus2sub.csv"
US_STATES_FILE = "ne_10m_admin_1_states_provinces.shp"
LAKES_FILE = "ne_10m_lakes.shp"
TEXAS_VALUES_FILE = "texas_v0.csv"

POLYGON_OPACITY = 0.35
POLYGON_LINE_WIDTH = 0.06
MAP_ZOOM = 3.2

sub = pd.read_csv(SUB_FILE)
bus2sub = pd.read_csv(BUS2SUB_FILE)
texas_vals = pd.read_csv(TEXAS_VALUES_FILE)


bus_geo = bus2sub.merge(
    sub[["sub_id", "name", "lat", "lon", "interconnect"]],
    on="sub_id",
    how="left",
    suffixes=("_bus", "_sub")
).dropna(subset=["lat", "lon"]).copy()

bus_geo["lat"] = pd.to_numeric(bus_geo["lat"], errors="coerce")
bus_geo["lon"] = pd.to_numeric(bus_geo["lon"], errors="coerce")
bus_geo["bus_id"] = pd.to_numeric(bus_geo["bus_id"], errors="coerce")
bus_geo = bus_geo.dropna(subset=["lat", "lon", "bus_id"]).copy()

texas_vals["bus_id"] = pd.to_numeric(texas_vals["bus_id"], errors="coerce")
texas_vals = texas_vals.dropna(subset=["bus_id"]).copy()


np.random.seed(42)
bus_geo["hc_random"] = np.random.rand(len(bus_geo)) * 100

bus_geo = bus_geo.merge(
    texas_vals[["bus_id", "HC_aggregate_MW", "HC_network_MW"]],
    on="bus_id",
    how="left"
)

bus_geo["hc_texas_bus"] = bus_geo["HC_network_MW"]

sub_metric = (
    bus_geo.groupby(
        ["sub_id", "name", "lat", "lon", "interconnect_bus"],
        as_index=False
    )
    .agg(
        hc_random=("hc_random", "mean"),
        hc_texas=("hc_texas_bus", "mean"),
        bus_count=("bus_id", "count")
    )
)


gdf = gpd.GeoDataFrame(
    sub_metric,
    geometry=gpd.points_from_xy(sub_metric["lon"], sub_metric["lat"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")


coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
vor = Voronoi(coords)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

regions, vertices = voronoi_finite_polygons_2d(vor)


polygons = [Polygon(vertices[region]) for region in regions]

vor_gdf = gpd.GeoDataFrame(
    gdf.drop(columns="geometry").copy(),
    geometry=polygons,
    crs="EPSG:3857"
)


us_states = gpd.read_file(US_STATES_FILE)
us_states = us_states[us_states["admin"] == "United States of America"].copy()

exclude = {"Alaska", "Hawaii", "Puerto Rico"}
conus = us_states[~us_states["name"].isin(exclude)].copy()
conus = conus.to_crs("EPSG:3857")
conus_mask = conus.dissolve()

lakes = gpd.read_file(LAKES_FILE)
lakes = lakes.to_crs("EPSG:3857")

great_lake_names = {
    "Lake Superior",
    "Lake Michigan",
    "Lake Huron",
    "Lake Erie",
    "Lake Ontario"
}


great_lakes = lakes[lakes["name"].isin(great_lake_names)].copy()
great_lakes_mask = great_lakes.dissolve()


conus_land_no_lakes = gpd.overlay(conus_mask, great_lakes_mask, how="difference")


vor_gdf = gpd.clip(vor_gdf, conus_land_no_lakes)


vor_gdf = vor_gdf.to_crs("EPSG:4326")
vor_gdf["sub_id_str"] = vor_gdf["sub_id"].astype(str)

geojson = vor_gdf.__geo_interface__



vor_gdf["interconnect_bus"] = vor_gdf["interconnect_bus"].astype(str)

view_options = [
    ("Random values — All", "hc_random", None),
    ("Random values — Eastern", "hc_random", "Eastern"),
    ("Random values — Western", "hc_random", "Western"),
    ("Texas real values", "hc_texas", "Texas"),
]

def build_view_arrays(df, metric_col, interconnect_filter=None):
    if interconnect_filter is None:
        mask = pd.Series(True, index=df.index)
    else:
        mask = df["interconnect_bus"] == interconnect_filter

    z_vals = df[metric_col].where(mask, np.nan)

    customdata = np.array([
        [
            row["name"],
            row["interconnect_bus"],
            row["bus_count"],
            row[metric_col]
        ]
        for _, row in df.iterrows()
    ], dtype=object)

    visible_nonnull = z_vals.dropna()
    if len(visible_nonnull) > 0:
        zmin = float(visible_nonnull.min())
        zmax = float(visible_nonnull.max())
        if zmin == zmax:
            zmax = zmin + 1e-9
    else:
        zmin, zmax = 0.0, 1.0

    return z_vals, customdata, zmin, zmax

view_cache = {}
for label, metric_col, inter_filter in view_options:
    view_cache[label] = build_view_arrays(vor_gdf, metric_col, inter_filter)

initial_label = "Random values — All"
initial_z, initial_customdata, initial_zmin, initial_zmax = view_cache[initial_label]

# map
fig = px.choropleth_map(
    vor_gdf,
    geojson=geojson,
    locations="sub_id_str",
    featureidkey="properties.sub_id_str",
    color=initial_z,
    color_continuous_scale="Viridis",
    center={"lat": 39.5, "lon": -98.35},
    zoom=MAP_ZOOM,
    opacity=POLYGON_OPACITY,
    title=f"Adaptive One-Polygon-Per-Substation Voronoi Map — {initial_label}"
)

fig.update_traces(
    marker_line_width=POLYGON_LINE_WIDTH,
    z=initial_z,
    zmin=initial_zmin,
    zmax=initial_zmax,
    customdata=initial_customdata,
    hovertemplate=(
        "Substation: %{customdata[0]}<br>"
        "Interconnect: %{customdata[1]}<br>"
        "Bus count: %{customdata[2]}<br>"
        "Value: %{customdata[3]:.2f}<extra></extra>"
    )
)


buttons = []
for label, _, _ in view_options:
    z_vals, customdata, zmin, zmax = view_cache[label]

    buttons.append(
        dict(
            label=label,
            method="update",
            args=[
                {
                    "z": [z_vals],
                    "customdata": [customdata],
                    "zmin": [zmin],
                    "zmax": [zmax],
                },
                {
                    "title": f"Adaptive One-Polygon-Per-Substation Voronoi Map — {label}"
                }
            ]
        )
    )

fig.update_layout(
    map_style="carto-positron",
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top"
        )
    ],
    annotations=[
        dict(
            text="View",
            x=0.01,
            y=1.04,
            xref="paper",
            yref="paper",
            showarrow=False
        )
    ]
)

fig.show()

fig.write_html("substation_voronoi_interconnect_dropdown.html")