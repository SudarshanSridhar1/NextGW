# Substation Voronoi Hosting Capacity Visualization
## Objective

The goal of this script is to create a scalable and interpretable map-based framework for visualizing grid-related metrics at the substation level. Rather than plotting substations as points, the script partitions geographic space into adaptive polygonal regions using a Voronoi tessellation, which provides a more continuous and visually expressive representation of spatial patterns.
---

## Methodology

The script follows the workflow below:

### 1. Load input data
The script reads:
- substation metadata
- bus-to-substation mappings
- Texas hosting capacity values
- U.S. state boundary shapefiles
- lake shapefiles

### 2. Join bus and substation data
Each bus is associated with:
- a substation identifier
- a substation name
- latitude and longitude
- an interconnection label

This produces a georeferenced bus-level dataset.

### 3. Attach metric values
Two types of metric values are incorporated:
- **random placeholder values**, used for visualization testing
- **Texas real values**, pulled from the Texas hosting capacity dataset

At the bus level, the script currently uses:
- `HC_network_MW` as the Texas real-value metric

### 4. Aggregate from bus level to substation level
Because one substation may serve multiple buses, the script aggregates bus-level values to one record per substation using mean values. It also stores the number of buses associated with each substation.

### 5. Construct Voronoi polygons
The substation coordinates are used to build a Voronoi tessellation. This creates one polygonal region per substation, representing a nearest-neighbor geographic partition of space.

Infinite Voronoi regions are converted into finite polygons so they can be plotted and clipped properly.

### 6. Clip polygons to the continental U.S.
The Voronoi polygons are clipped to a land-only mask representing the continental United States. Alaska, Hawaii, and Puerto Rico are excluded. The Great Lakes are also removed from the mask so polygons do not extend into those water bodies.

### 7. Render an interactive map
The clipped polygons are converted to GeoJSON and displayed with Plotly as an interactive choropleth map. A dropdown menu allows the user to switch between different views without rebuilding the geometry.

### 8. Export to HTML
The finished interactive map is saved as a standalone HTML file for sharing and reuse.

---

## Current Views

The map currently supports the following dropdown views:

### Random values — All
Displays randomly generated placeholder values across all substations. -> future should be real values


### Random values — Eastern
Displays placeholder values only for substations labeled as part of the Eastern interconnection. -> future should be real values


### Random values — Western
Displays placeholder values only for substations labeled as part of the Western interconnection. -> future should be real values

### Texas real values
Displays Texas hosting capacity values using `HC_network_MW` for substations labeled as `Texas`.

---

## Input Data Requirements

The script expects the following files to be present in the working directory.

### `sub.csv`
Substation-level metadata.

### `bus2sub.csv`
Bus-to-substation mapping data.

### `texas_v0.csv`
Texas bus-level hosting capacity data.

### `ne_10m_admin_1_states_provinces.shp`
Natural Earth state/province shapefile used to define the U.S. clipping boundary.

### `ne_10m_lakes.shp`
Natural Earth lake shapefile used to remove the Great Lakes from the land mask.

---

## Output

The script produces two outputs:

### Interactive map display
The map is displayed directly through Plotly using:
- `fig.show()`

### HTML export
The map is saved as:
- `substation_voronoi_interconnect_dropdown.html`

This HTML file can be opened in any browser and shared independently of the Python environment.

--
## Required Pyton Packages 

Run "pip install pandas numpy geopandas plotly shapely scipy" to install required pyton packages before running the script.