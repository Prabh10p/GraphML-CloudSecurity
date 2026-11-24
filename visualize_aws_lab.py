
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT = Path("results/aws_lab")

with open(OUTPUT / "aws_lab_graph.json") as f:
    data = json.load(f)

G = nx.DiGraph()

for node in data["nodes"]:
    G.add_node(node["id"], **node)

for edge in data["edges"]:
    G.add_edge(
        edge["source"],
        edge["target"],
        **edge
    )

positions = {}
labels = {}
colors = []

for node, attributes in G.nodes(data=True):

    resource_type = attributes["resource_type"]
    name = attributes.get("name", node)

    if resource_type == "ec2":

        risky = "risky" in name.lower()

        positions[node] = (
            2 if risky else 0,
            4
        )

        labels[node] = (
            "Risky EC2" if risky else "Secure EC2"
        )

        colors.append("#E74C3C" if risky else "#27AE60")

    elif resource_type == "iam":

        risky = attributes.get("privilege") == 1

        positions[node] = (
            2 if "risky" in name.lower() else 0,
            2
        )

        labels[node] = (
            f"{'Risky' if 'risky' in name.lower() else 'Secure'} IAM\n"
            f"{attributes.get('allowed_permission_count', '?')} permissions"
        )

        colors.append("#E74C3C" if risky else "#27AE60")

    elif resource_type == "s3":

        critical = attributes.get("criticality") == 1

        positions[node] = (
            2 if critical else 0,
            0
        )

        labels[node] = (
            "Critical S3" if critical
            else "Noncritical S3"
        )

        colors.append("#E74C3C" if critical else "#3498DB")

fig, ax = plt.subplots(figsize=(12, 9))

nx.draw_networkx_nodes(
    G,
    positions,
    node_color=colors,
    node_size=6500,
    edgecolors="#263238",
    linewidths=1.5,
    ax=ax
)

nx.draw_networkx_labels(
    G,
    positions,
    labels=labels,
    font_size=11,
    font_weight="bold",
    font_color="white",
    ax=ax
)

edge_labels = {}

for source, target, attributes in G.edges(data=True):

    relationship = attributes["relationship"]

    if relationship == "instance_profile_role":

        edge_labels[(source, target)] = "IAM attachment"
        edge_color = "#34495E"

    else:

        actions = attributes.get("actions", [])

        short_actions = [
            action.replace("s3:", "")
            for action in actions
        ]

        edge_labels[(source, target)] = "\n".join(short_actions)

        edge_color = (
            "#C0392B"
            if len(actions) > 1
            else "#27AE60"
        )

    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=[(source, target)],
        edge_color=edge_color,
        width=2.5,
        arrows=True,
        arrowsize=25,
        node_size=6500,
        ax=ax
    )

nx.draw_networkx_edge_labels(
    G,
    positions,
    edge_labels=edge_labels,
    font_size=9,
    rotate=False,
    ax=ax
)

legend = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Secure configuration",
        markerfacecolor="#27AE60",
        markersize=13
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Research-design risk or criticality",
        markerfacecolor="#E74C3C",
        markersize=13
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Noncritical asset",
        markerfacecolor="#3498DB",
        markersize=13
    )
]

ax.legend(
    handles=legend,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=3
)

ax.set_title(
    "Real AWS Cloud Security Graph\n"
    "Configuration and IAM Permission Relationships",
    fontsize=18,
    fontweight="bold",
    pad=25
)

ax.set_xlim(-0.8, 2.8)
ax.set_ylim(-0.7, 4.7)
ax.axis("off")

fig.text(
    0.5,
    0.02,
    "Source: AWS API discovery and IAM policy simulation. "
    "Edges do not establish runtime attack paths.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.savefig(
    OUTPUT / "real_aws_security_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT / "real_aws_security_graph.svg",
    bbox_inches="tight"
)

print("=" * 50)
print("AWS GRAPH VISUALIZATION COMPLETE")
print("=" * 50)
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("PNG:", OUTPUT / "real_aws_security_graph.png")
print("SVG:", OUTPUT / "real_aws_security_graph.svg")

plt.show()
