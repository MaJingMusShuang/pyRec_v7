import random
from collections import defaultdict
from time import time
from six import iterkeys

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def make_consistent(self):
      for k in iterkeys(self):
          self[k] = list(sorted(set(self[k])))
      self.remove_self_loops()
      return self

  def remove_self_loops(self):
      removed = 0
      for x in self:
          if x in self[x]:
              self[x].remove(x)
              removed += 1

      print('remove_self_loops: removed {}'.format(removed))
      return self

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [node for node in path]




def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    return walks


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G
