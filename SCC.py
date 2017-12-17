
def arcs_to_graph(Arcs):
	"""
	Convert Arc's set to graph representation
	"""
	graph = {}
	for k,v in Arcs.items():
		try:
			graph[k[0]].append(k[1])
		except  KeyError:
			graph[k[0]] =[]
			graph[k[0]].append(k[1])
		try:
			graph[k[1]]
		except  KeyError:
			graph[k[1]] =[]
			
	
	return graph

def dfs_stack(graph,node,visited,stack):
	"""
	1st step of Kosaraju's algorithm. Create an empty stack and 
	perform DFS on the original graph
	"""
	visited[node]= True
	for neighbour in graph[node]:
		if visited[neighbour] ==  False:
			dfs_stack(graph,neighbour,visited,stack)
	stack = stack.append(node)

def transpose_graph(graph):
	"""
	2nd step of Kosaraju's algorithm. Create the transpose of the
	graph(reverse edge directions)
	"""
	rev_graph = {}
	for k,v in graph.items():
		for node in v:
			try:
				rev_graph[node].append(k)
			except  KeyError:
				rev_graph[node] =[]
				rev_graph[node].append(k)

	for k,v in graph.items():
		try:
			rev_graph[k]
		except  KeyError:
			rev_graph[k] =[]
	
	return rev_graph


def dfs(graph,node,visited,scc_list):
	"""
	3d step of Kosaraju's algorithm. Perform DFS on reversed
	graph by poping one by one elemrnts of the stack
	"""
	visited[node]= True
	scc_list.append(node)
	for neighbour in graph[node]:
		if visited[neighbour] ==  False:
			scc_list = dfs(graph,neighbour,visited,scc_list)
	
	return scc_list

def get_SCC(Arcs):
	"""
	Get a list of strongly connected components(SCC) of a given graph,
	represented as a set of arcs
	"""
	
	graph = arcs_to_graph(Arcs)
	stack =[]
	visited = visited_false(graph)
	for k,v in graph.items():
		if visited[k] == False:
			dfs_stack(graph,k,visited,stack)

	rev_graph = transpose_graph(graph)
	visited = visited_false(graph)
	SCC_list =[]
	
	while (len(stack) > 0):
            node = stack.pop()
            if visited[node]==False:
            	SCC_list.append(dfs(rev_graph,node,visited,[]))
	return SCC_list


def visited_false(graph): 
	#Initialize all nodes to visited = False
	visited = {}
	for k,v in graph.items():
		visited[k] = False
	return visited


