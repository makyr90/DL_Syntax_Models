import numpy as np
import SCC
import copy
import random



def parse_proj(scores_matrix, gold=None):
	'''
    Parse using Chu-Liu-Edmonds algorithm(non-projective).
    '''
	nr, nc = np.shape(scores_matrix)
	if nr != nc:
		raise ValueError("scores must be a squared matrix with nw+1 rows")

	Arcs = {}
	for idx in range(scores_matrix.shape[0]):
		for jdx in range(scores_matrix.shape[1]):
			# ADD 	+1.0 to all edges that are not in gold tree, to enable learning via loss function
			Arcs[(idx,jdx)] = scores_matrix[idx,jdx] + (0.0 if gold is not None and gold[jdx]==idx else 1.0)

	#Delete self-edges and edges that points to root
	Arcs_copy = copy.deepcopy(Arcs)
	for k,v in Arcs.items():

		if (k[1] == 0):
			del Arcs_copy[k]
		if (k[0]== k[1] and k[0]!=0):
			del Arcs_copy[k]

	newArcs =  Chu_Liu_Edmonds(Arcs_copy)

	#Get the initial weights of selected arcs
	for k,v in newArcs.items():
		newArcs[k] = Arcs[k]

	heads = [0 for _ in range(nr)]
	for k,v in newArcs.items():
		heads[k[1]] = k[0]

	#detect multiple roots
	rootCount = 0
	for head in heads[1:]:
		if head == 0:
			rootCount += 1

	if (rootCount == 1):
		return heads

	else:
		root_indices = [idx+1 for idx,root in enumerate(heads[1:]) if root==0]
		root_scores = [scores_matrix[0,root] for root in root_indices]
		highest_scored_roots_idx= [idx for idx, score in zip(root_indices,root_scores) if score == max(root_scores)]
		best_root_idx = random.choice(highest_scored_roots_idx)
		for jdx in range(1,scores_matrix.shape[1]):
			if (jdx!= best_root_idx):
				Arcs[(0,jdx)] = -float("inf")

		Arcs_copy2 = copy.deepcopy(Arcs)

		#Delete self-edges and edges that points to root
		for k,v in Arcs.items():

			if (k[1] == 0):
				del Arcs_copy2[k]
			if (k[0]== k[1] and k[0]!=0):
				del Arcs_copy2[k]

		newArcs2 =  Chu_Liu_Edmonds(Arcs_copy2)

		#Get the initial weights of selected arcs
		for k,v in newArcs2.items():
			newArcs2[k] = Arcs[k]

		heads = [0 for _ in range(nr)]
		for k,v in newArcs2.items():
			heads[k[1]] = k[0]

		#Eliminate multiple roots
		rootCount = 0
		for head in heads[1:]:
			if head == 0:
				rootCount += 1
		if (rootCount > 1):
			print("Error!! MST contains more than one root!!")


		return heads



def Chu_Liu_Edmonds(Arcs):

	#greedy choice
	newArcs = highest_incoming_arcs(Arcs)
	scc = SCC.get_SCC(newArcs)
	for component in scc:
		if (len(component)>1):
			cycle = component
			break

	else:
		return newArcs

	Cycle_arcs = {}
	for k,v in newArcs.items():
		if ((k[0] in cycle) and  (k[1] in cycle)):
			Cycle_arcs[k] = v

	contracted_Arcs,wc,ep,predecessor = contract(Arcs,cycle,Cycle_arcs)
	newArcs = Chu_Liu_Edmonds(contracted_Arcs)

	#Final arc set
	del_keys = []
	insert_keys = []
	for k,v in newArcs.items():
		if (k[0] == wc):
			endpoint = ep[(wc,k[1])]
			del_keys.append((wc,k[1]))
			insert_keys.append((endpoint,k[1],v))

	for key in del_keys:
		del newArcs[key]

	for key in insert_keys:
		newArcs[(key[0],key[1])] = key[2]


	for k,v in newArcs.items():
		if (k[1] == wc):
			endpoint = ep[(k[0],wc)]
			in_edge = (k[0],v)
			del Cycle_arcs[(predecessor[endpoint],endpoint)]
			break

	newArcs[(in_edge[0],endpoint)] = in_edge[1]
	del newArcs[in_edge[0],wc]

	for k,v in Cycle_arcs.items():
		newArcs[k] = v

	return  newArcs




def contract(Arcs,cycle,Cycle_arcs):

	#Contract cycle & recompute arcs from/to the cycle
	wc =hash(tuple(cycle))
	contracted_Arcs = {}
	ep = {}
	for k,v in Arcs.items():
		if (not((k[0] in cycle) and  (k[1] in cycle))):
			contracted_Arcs[k] = v


	#out-degree edges of cycle
	del_keys = []
	insert_keys = []
	for k,v in contracted_Arcs.items():
		if k[0] in cycle:
			max_edge = -float("inf")
			argmax = None
			for edge in cycle:
				try:
					edge_value = Arcs[(edge,k[1])]
					if (edge_value > max_edge):
						max_edge = edge_value
						argmax = edge
					del_keys.append((edge,k[1]))

				except  KeyError:
					continue
			ep[(wc,k[1])] = argmax
			insert_keys.append((wc,k[1],max_edge))

	for key in del_keys:
		try:
			del contracted_Arcs[key]
		except  KeyError:
					continue

	for key in insert_keys:
		contracted_Arcs[(key[0],key[1])] = key[2]

	#in-degree edges of cycle
	score = sum(Cycle_arcs.values())
	#find predecessors
	predecessor = {}
	for node in cycle:
		for k,v in Cycle_arcs.items():
			if k[1]==node:
				predecessor[node] = k[0]

	del_keys = []
	insert_keys = []
	for k,v in contracted_Arcs.items():

		if k[1] in cycle:
			max_edge = -float("inf")
			argmax = None
			for node in cycle:
				try:
					candidate_edge = Arcs[k[0],node]- Cycle_arcs[(predecessor[node],node)]
					if (candidate_edge > max_edge):
						max_edge = candidate_edge
						argmax = node
					del_keys.append((k[0],node))
				except  KeyError:
					continue

			ep[(k[0],wc)] = argmax
			insert_keys.append((k[0],wc,max_edge + score))


	for key in del_keys:
		try:
			del contracted_Arcs[key]
		except  KeyError:
					continue

	for key in insert_keys:
		contracted_Arcs[(key[0],key[1])] = key[2]


	return contracted_Arcs,wc,ep,predecessor




def highest_incoming_arcs(Arcs):

	#For each vertex(except root) find the incoming arc with highest value
	greedy_arcs = {}
	graph = SCC.arcs_to_graph(Arcs)
	rev_graph = SCC.transpose_graph(graph)

	for k,v in rev_graph.items():
		if (not (k == 0)):

			max_edge = -float("inf")
			argmax = None

			for neighbor in v:
				if (Arcs[(neighbor,k)] > max_edge):
					max_edge =  Arcs[(neighbor,k)]
					argmax = neighbor
			greedy_arcs[(argmax,k)] = Arcs[(argmax,k)]

	return greedy_arcs
