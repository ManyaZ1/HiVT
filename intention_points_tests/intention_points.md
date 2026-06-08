## 1 KMeans on local-frame final displacement vectors

Cluster the vector from agent’s current position to its final future position, in the agent-aligned frame.
**Pros:** Simple, captures maneuver endpoints, robust to rotation.
**Cons:** Only encodes final goal, not the path shape.

## 2 KMeans on full future displacement vectors

Cluster the entire future trajectory (e.g., flatten all 30 future 
[x,y]
[x,y] steps into a 
60
60-dim vector).
**Pros:** Captures full maneuver shape, not just endpoint.
**Cons:** Higher dimensionality, clusters may be less stable, more sensitive to noise.
## 3 KMeans on lane-relative goal points

Project agent’s final position onto candidate lane centerlines, cluster those projected points.
**Pros:** Anchors are semantically meaningful (e.g., “left lane end”), robust to map geometry.
**Cons:** Requires map/lane processing, more complex to implement.

## 4 KMeans on lane-following trajectory shapes

Cluster full future trajectories, but only those that follow a specific lane or maneuver type.
**Pros:** Most semantically grounded, can directly correspond to “turn left,” “go straight,” etc.
**Cons:** Requires lane assignment and maneuver labeling, most complex.


## Summary:

For a first implementation, option 1 is the simplest and most robust.
Option 2 is good if you want to capture maneuver shape, but may need more clusters.
Options 3 and 4 are best for semantic diversity, but require map/lane processing.