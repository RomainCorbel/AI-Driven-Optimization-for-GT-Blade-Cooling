"""
Visualisation interactive de la géométrie de l'aube GT avec les 4 labels :
  Label 0 — Intérieur (volume)   → points bleus transparents
  Label 1 — Entrée  (inlet)      → surface bleue vive
  Label 2 — Parois  (walls)      → surface orange semi-transparente
  Label 3 — Sortie  (outlet)     → surface rouge vive

Ouvre le fichier HTML généré dans un navigateur pour une vue 3D interactive.
"""
import os
import trimesh
import numpy as np
import plotly.graph_objects as go


STL_PATH = os.path.join(os.path.dirname(__file__), "Baseline_ML4Science.stl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "run", "geometry_labels.html")

# ── Couleurs et opacités par label ──────────────────────────────────────────
LABELS = {
    0: dict(name="Label 0 — Inside",  color="#64B5F6", opacity=0.30),
    1: dict(name="Label 1 — Inlet",  color="#1E88E5", opacity=0.90),
    2: dict(name="Label 2 — Walls",  color="#FB8C00", opacity=0.30),
    3: dict(name="Label 3 — Outlet", color="#E53935", opacity=0.90),
}


def classify_faces(obj: trimesh.Trimesh):
    threshold = 1e-5
    x_max = obj.vertices[:, 0].max()

    inlet, outlet, walls = [], [], []
    for i, face in enumerate(obj.faces):
        xs = obj.vertices[face, 0]
        if np.all(np.abs(xs) < threshold):
            inlet.append(i)
        elif np.all(np.abs(xs - x_max) < threshold):
            outlet.append(i)
        else:
            walls.append(i)

    return inlet, outlet, walls


def mesh_trace(obj: trimesh.Trimesh, face_indices: list, label: int) -> go.Mesh3d:
    cfg = LABELS[label]
    sub = obj.submesh([face_indices], only_watertight=False)[0]
    v, f = sub.vertices, sub.faces

    # Intensité de lighting pour donner du relief
    intensity = np.linalg.norm(sub.vertex_normals, axis=1)

    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=cfg["color"],
        opacity=cfg["opacity"],
        name=cfg["name"],
        showlegend=True,
        flatshading=False,
        lighting=dict(
            ambient=0.5,
            diffuse=0.9,
            specular=0.4,
            roughness=0.4,
            fresnel=0.3,
        ),
        lightposition=dict(x=200, y=400, z=300),
    )


def interior_scatter(obj: trimesh.Trimesh, n: int = 8_000) -> go.Scatter3d:
    """Points intérieurs échantillonnés pour représenter le volume (label 0)."""
    pts = trimesh.sample.volume_mesh(obj, n)
    cfg = LABELS[0]
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=1.8,
            color=cfg["color"],
            opacity=cfg["opacity"],
        ),
        name=cfg["name"],
        showlegend=True,
    )


def main():
    obj = trimesh.load(STL_PATH)

    print("Classification des faces …")
    inlet_idx, outlet_idx, wall_idx = classify_faces(obj)
    print(f"  Inlet  : {len(inlet_idx):>6} faces")
    print(f"  Outlet  : {len(outlet_idx):>6} faces")
    print(f"  Walls  : {len(wall_idx):>6} faces")
    traces = [
        interior_scatter(obj),
        mesh_trace(obj, inlet_idx,  1),
        mesh_trace(obj, wall_idx,   2),
        mesh_trace(obj, outlet_idx, 3),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Geometry Labels",
            font=dict(size=18, color="white"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X — flow direction", color="white",
                       backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Y", color="white",
                       backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Z", color="white",
                       backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)"),
            aspectmode="data",
            bgcolor="rgb(15, 15, 25)",
        ),
        paper_bgcolor="rgb(15, 15, 25)",
        font=dict(color="white"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(30, 30, 50, 0.85)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    fig.write_html(OUTPUT_PATH)


if __name__ == "__main__":
    main()
