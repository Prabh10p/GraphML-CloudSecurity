import json
from pathlib import Path
import boto3
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

session = boto3.Session(profile_name='cloud-project', region_name='us-east-1')
s3, ec2, iam = (session.client(x) for x in ('s3','ec2','iam'))
G = nx.DiGraph()
output = Path('aws_graph_output')
output.mkdir(exist_ok=True)

for page in s3.get_paginator('list_buckets').paginate():
    for bucket in page.get('Buckets', []):
        name = bucket['Name']
        node = f's3:{name}'
        G.add_node(node, label=name, type='S3')
        try:
            cfg = s3.get_public_access_block(Bucket=name)['PublicAccessBlockConfiguration']
            G.nodes[node]['public_access_block_enabled'] = all(cfg.get(k,False) for k in ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets'))
        except s3.exceptions.NoSuchPublicAccessBlockConfiguration:
            G.nodes[node]['public_access_block_enabled'] = False
        except Exception as exc:
            G.nodes[node]['public_access_block_status'] = 'unknown'
            print(f'Unable to inspect public access block for {name}: {exc}')

for page in ec2.get_paginator('describe_security_groups').paginate():
    for sg in page.get('SecurityGroups', []):
        sg_id = sg['GroupId']
        G.add_node(f'sg:{sg_id}', label=sg_id, type='SecurityGroup')
        for rule in sg.get('IpPermissions', []):
            protocol, start, end = rule.get('IpProtocol'), rule.get('FromPort'), rule.get('ToPort')
            sensitive = protocol == '-1' or (protocol == 'tcp' and start is not None and end is not None and (start <= 22 <= end or start <= 3389 <= end))
            if sensitive and (any(r.get('CidrIp') == '0.0.0.0/0' for r in rule.get('IpRanges', [])) or any(r.get('CidrIpv6') == '::/0' for r in rule.get('Ipv6Ranges', []))):
                G.add_node('internet', label='Internet', type='External')
                G.add_edge('internet', f'sg:{sg_id}', relationship='potential_exposure')

for page in ec2.get_paginator('describe_instances').paginate():
    for reservation in page.get('Reservations', []):
        for instance in reservation.get('Instances', []):
            iid = instance['InstanceId']
            G.add_node(f'ec2:{iid}', label=iid, type='EC2')
            for sg in instance.get('SecurityGroups', []):
                G.add_edge(f'sg:{sg["GroupId"]}', f'ec2:{iid}', relationship='attached_to')

role_name = 'Snowflakerole'
role = iam.get_role(RoleName=role_name)['Role']
role_id = f'role:{role_name}'
G.add_node(role_id, label=role_name, type='IAMRole')

for page in iam.get_paginator('list_attached_role_policies').paginate(RoleName=role_name):
    for policy in page.get('AttachedPolicies', []):
        arn = policy['PolicyArn']
        policy_id = f'policy:{arn}'
        G.add_node(policy_id, label=policy['PolicyName'], type='IAMPolicy')
        G.add_edge(role_id, policy_id, relationship='attached_policy')
        # Resource-scope relationships only; these are NOT verified effective access.
        meta = iam.get_policy(PolicyArn=arn)['Policy']
        doc = iam.get_policy_version(PolicyArn=arn, VersionId=meta['DefaultVersionId'])['PolicyVersion']['Document']
        statements = doc.get('Statement', [])
        if isinstance(statements, dict):
            statements = [statements]
        for stmt in statements:
            if stmt.get('Effect') != 'Allow' or 'NotAction' in stmt or 'NotResource' in stmt:
                continue
            actions = stmt.get('Action', [])
            resources = stmt.get('Resource', [])
            if isinstance(actions, str): actions = [actions]
            if isinstance(resources, str): resources = [resources]
            if not any(a in ('s3:*', '*') or a.startswith('s3:') for a in actions):
                continue
            for bucket in [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'S3']:
                name = bucket.removeprefix('s3:')
                bucket_arn = f'arn:aws:s3:::{name}'
                object_arn = bucket_arn + '/*'
                if '*' in resources or bucket_arn in resources or object_arn in resources or 'arn:aws:s3:::*' in resources:
                    G.add_edge(policy_id, bucket, relationship='policy_resource_scope', effective_access_unverified=True)

trust = role['AssumeRolePolicyDocument']
statements = trust.get('Statement', [])
if isinstance(statements, dict): statements = [statements]
for stmt in statements:
    if stmt.get('Effect') != 'Allow': continue
    actions = stmt.get('Action', [])
    if isinstance(actions, str): actions = [actions]
    if 'sts:AssumeRole' not in actions and '*' not in actions: continue
    principal = stmt.get('Principal', {})
    aws_principals = principal.get('AWS', []) if isinstance(principal, dict) else []
    if isinstance(aws_principals, str): aws_principals = [aws_principals]
    for arn in aws_principals:
        principal_id = f'principal:{arn}'
        G.add_node(principal_id, label='External principal', type='ExternalPrincipal')
        G.add_edge(principal_id, role_id, relationship='trust_policy_allows', condition_present=bool(stmt.get('Condition')), effective_assumption_unverified=True)

payload = nx.node_link_data(G, edges='links')
(output / 'aws_graph.json').write_text(json.dumps(payload, indent=2, default=str))
nx.write_graphml(G, output / 'aws_graph.graphml')

plt.figure(figsize=(15, 9))
pos = nx.spring_layout(G, seed=42, k=1.5)
colors = {'S3':'#f5a623','IAMRole':'#8b5cf6','IAMPolicy':'#ef4444','ExternalPrincipal':'#64748b','SecurityGroup':'#22a6b3','EC2':'#2ecc71','External':'#34495e'}
nx.draw_networkx_nodes(G, pos, node_color=[colors.get(d.get('type'), '#aaa') for _, d in G.nodes(data=True)], node_size=2200)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, node_size=2200, edge_color='#777', connectionstyle='arc3,rad=0.06')
nx.draw_networkx_labels(G, pos, labels={n:d.get('label',n) for n,d in G.nodes(data=True)}, font_size=7)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['relationship'] for u,v,d in G.edges(data=True)}, font_size=6)
plt.title('AWS Configuration Graph (policy scope is not effective access)')
plt.axis('off')
plt.tight_layout()
plt.savefig(output / 'aws_graph.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'Graph saved to: {output.resolve()}')
print(f'Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}')
print('Policy-to-bucket edges indicate policy resource scope, not confirmed effective access.')
