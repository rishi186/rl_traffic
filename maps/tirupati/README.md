# Tirupati SUMO scenario

Central Tirupati, Andhra Pradesh, around the railway station and APSRTC bus
station. Use `sumo-gui -c maps/tirupati/tirupati.sumocfg` from the repository root.
The map includes roads that continue beyond the requested OpenStreetMap extract.

Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
[Extract request](https://api.openstreetmap.org/api/0.6/map?bbox=79.408,13.616,79.442,13.643)
was retrieved on 2026-11-25. Its SHA-256 was
`e44fff5bb1903e4d95df14f7a51f2253552e98133e62f0f92fd4ca1aab39abac`.
The API returns current data, so a later download can differ.

The bundled network was built with SUMO 1.26.0 and left-hand traffic. The
extract contained zero OSM nodes tagged `highway=traffic_signals`. SUMO inferred
all ten controllers. The 450 passenger trips are seeded synthetic demand,
scheduled every three seconds for the first 1,350 seconds. Signal placement,
timing and traffic volumes have no field calibration.

Standalone SUMO ran all 450 trips to completion by 1,800 simulated seconds.
The benchmark smoke run exercised fixed time, actuated, DQN and PPO.
SUMO reported two phases without controlling detectors at one actuated
controller. Treat the actuated replay as a simulation diagnostic.

To rebuild from the same OSM extract saved as `source.osm`, run these commands
from its directory and gzip the network XML as `tirupati.net.xml.gz`:

```powershell
netconvert --osm-files source.osm --lefthand --keep-edges.by-vclass passenger --keep-edges.by-type highway.motorway,highway.motorway_link,highway.trunk,highway.trunk_link,highway.primary,highway.primary_link,highway.secondary,highway.secondary_link,highway.tertiary,highway.tertiary_link,highway.unclassified,highway.residential,highway.living_street --keep-edges.components 1 --no-turnarounds --geometry.remove --junctions.join --tls.guess --tls.discard-simple --output-file tirupati.net.xml
python "$env:SUMO_HOME/tools/randomTrips.py" -n tirupati.net.xml -r tirupati.rou.xml -o tirupati.trips.xml --edge-permission passenger --min-distance 650 --max-distance 2700 --seed 42 --validate --fringe-factor 2 -e 1350 -p 3
```
