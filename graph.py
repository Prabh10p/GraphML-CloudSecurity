import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
nodes = {
    'Internet': {'kind':'entry', 'criticality':0, 'cvss':0},
    'Web EC2': {'kind':'compute', 'criticality':2, 'cvss':7.2},
    'Admin IAM Role': {'kind':'identity', 'criticality':3, 'cvss':0},
    'Private Database': {'kind':'data', 'criticality':10, 'cvss':0},
    'Sensitive S3': {'kind':'data', 'criticality':9, 'cvss':0},
    'Isolated EC2': {'kind':'compute', 'criticality':1, 'cvss':9.8},
}
G.add_nodes_from((n, a) for n,a in nodes.items())
G.add_edges_from([('Internet','Web EC2'),('Web EC2','Admin IAM Role'),('Admin IAM Role','Private Database'),('Admin IAM Role','Sensitive S3')])
assets = ['Private Database','Sensitive S3']

def exposure(graph):
    return sum(nodes[a]['criticality'] for a in assets if nx.has_path(graph, 'Internet', a))

def paths(graph):
    return [nx.shortest_path(graph,'Internet',a) for a in assets if nx.has_path(graph,'Internet',a)]

print('BASELINE: severity-only priority')
for n in sorted(nodes, key=lambda x:nodes[x]['cvss'], reverse=True):
    if nodes[n]['cvss']:
        print(f"  {n}: CVSS {nodes[n]['cvss']}")
print('\nGRAPH: reachable critical assets')
for p in paths(G): print('  ' + ' -> '.join(p))
base = exposure(G)
print(f'Baseline modeled exposure: {base}')
actions = [('Restrict public web access', ('Internet','Web EC2'), 3),('Remove excessive IAM permission',('Web EC2','Admin IAM Role'),2),('Revoke database permission',('Admin IAM Role','Private Database'),1),('Revoke S3 permission',('Admin IAM Role','Sensitive S3'),1)]
print('\nCOUNTERFACTUAL REMEDIATIONS (illustrative costs)')
for name, edge, cost in sorted(actions, key=lambda x: (base-exposure(nx.restricted_view(G,[],[x[1]])))/x[2], reverse=True):
    edited = G.copy(); edited.remove_edge(*edge)
    reduction = base-exposure(edited)
    print(f'  {name}: exposure reduction={reduction}, cost={cost}, benefit/cost={reduction/cost:.2f}')

pos={'Internet':(0,3),'Web EC2':(0,2),'Admin IAM Role':(0,1),'Private Database':(-1,0),'Sensitive S3':(1,0),'Isolated EC2':(2,2)}
colors=['#9ca3af','#eab308','#8b5cf6','#ef4444','#ef4444','#22c55e']
plt.figure(figsize=(10,6))
nx.draw_networkx(G,pos,node_color=colors,node_size=2500,font_size=9,arrows=True,arrowsize=20,edge_color='#64748b')
plt.title('Synthetic AWS Attack Graph — Demonstration')
plt.axis('off');plt.tight_layout();plt.savefig('attack_graph.png',dpi=180)
