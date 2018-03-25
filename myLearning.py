from collections import defaultdict
itemIdx_freq = defaultdict(int)

walk = [1]
walk.extend([1,2])
print(walk)
for node in walk:
    itemIdx_freq[node] += 1
print(itemIdx_freq)
sorted_res = sorted(itemIdx_freq.items(), key=lambda item:item[1], reverse=True)
topN = sorted_res[0:2]
topN = [item_freq[0] for item_freq in sorted_res[0:2]]
print(topN)