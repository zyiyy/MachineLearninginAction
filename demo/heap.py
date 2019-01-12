import heapq

h = []
heapq.heappush(h, (5, 'write code'))
heapq.heappush(h, (7, 'release product'))
heapq.heappush(h, (1, 'write spec'))
heapq.heappush(h, (3, 'create tests'))
heapq.heappush(h, (9, 'create'))
heapq.heappush(h, (2, 'write'))

print(h)

l = heapq.nsmallest(10, h)
print(l)