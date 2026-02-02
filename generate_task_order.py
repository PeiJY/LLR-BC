import random
random.seed(1)
items = ['U', 'GM', 'E', 'C', 'Ring', 'Grid']
orders = []

for _ in range(5):
    shuffled = items.copy()
    random.shuffle(shuffled)
    orders.append(shuffled)

for i, order in enumerate(orders):
    print(f"Order {i+1}: {order}")